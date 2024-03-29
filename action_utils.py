import numpy as np
import torch
from torch.distributions import Categorical

from torch.autograd import Variable

def parse_action_args(args):
    if args.num_actions[0] > 0:
        # environment takes discrete action
        args.continuous = False
        # assert args.dim_actions == 1
        # support multi action
        args.naction_heads = [int(args.num_actions[i]) for i in range(args.dim_actions)]
    else:
        # environment takes continuous action
        actions_heads = args.nactions.split(':')
        if len(actions_heads) == 1 and int(actions_heads[0]) == 1:
            args.continuous = True
        elif len(actions_heads) == 1 and int(actions_heads[0]) > 1:
            args.continuous = False
            args.naction_heads = [int(actions_heads[0]) for _ in range(args.dim_actions)]
        elif len(actions_heads) > 1:
            args.continuous = False
            args.naction_heads = [int(i) for i in actions_heads]
        else:
            raise RuntimeError("--nactions wrong format!")


def select_action(args, action_out):
    if args.hetgat:
        # sample for P
        nagents = args.nfriendly_P + args.nfriendly_A

        x = torch.zeros((1, nagents, 1))
        # TODO: fix this to output actions like [1,,23]
        mp = Categorical(logits = action_out['P'])
        p_idx = mp.sample() # size N

        for i in range(args.nfriendly_P):
            x[0][i] = p_idx.cpu().numpy()[i]

        # save log_prob of P agents
        # self.batch_P_log_probs[i_b].append(mp.log_prob(p_idx)) # size N

        # sample for A
        if args.nfriendly_A !=0:
            ma = Categorical(logits = action_out['A'])
            a_idx = ma.sample()

            for i in range(args.nfriendly_A):
                x[0][args.nfriendly_P + i] = a_idx.cpu().numpy()[i]
        # save log_prob of A agents
        # self.batch_A_log_probs[i_b].append(ma.log_prob(a_idx))
        return x
    if args.continuous:
        action_mean, _, action_std = action_out
        action = torch.normal(action_mean, action_std)
        return action.detach()
    else:
        log_p_a = action_out
        p_a = [[z.exp() for z in x] for x in log_p_a]
        # if args.eval:
            # ret = torch.argmax(p_a[0][0], dim=1).reshape(1, 1, 3, 1)
        # else:
        ret = torch.stack([torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_a])
        return ret

def translate_action(args, env, action):
    if args.num_actions[0] > 0:
        # environment takes discrete action
        action = [x.squeeze().data.numpy() for x in action]
        actual = action
        return action, actual
    else:
        if args.continuous:
            action = action.data[0].numpy()
            cp_action = action.copy()
            # clip and scale action to correct range
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                cp_action[i] = cp_action[i] * args.action_scale
                cp_action[i] = max(-1.0, min(cp_action[i], 1.0))
                cp_action[i] = 0.5 * (cp_action[i] + 1.0) * (high - low) + low
            return action, cp_action
        else:
            actual = np.zeros(len(action))
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                actual[i] = action[i].data.squeeze()[0] * (high - low) / (args.naction_heads[i] - 1) + low
            action = [x.squeeze().data[0] for x in action]
            return action, actual
