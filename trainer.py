import tracemalloc
from collections import namedtuple
from inspect import getargspec

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from action_utils import *
from utils import *

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

class Trainer(object):
    def __init__(self, args, policy_net, env, policy=None):
        # self.entropy_buffer_p = []
        # self.entropy_buffer_a = []
        self.start_time = time.time()
        self.args = args
        if args.hetcomm:
            self.policy_perception_net = policy_net[0]
            self.policy_action_net = policy_net[1]
        else:
            self.policy_net = policy_net
        self.policy_net.tensor_obs = args.tensor_obs
        self.env = env
        self.display = False
        self.last_step = False
        self.policy = policy
        if args.hetcomm:
            self.optimizer_perception = optim.RMSprop(self.policy_perception_net.parameters(),
                                           lr=args.lrate, alpha=0.97, eps=1e-6)
            self.params_perception = [p for p in self.policy_perception_net.parameters()]
            self.optimizer_action = optim.RMSprop(self.policy_action_net.parameters(),
                                                      lr=args.lrate, alpha=0.97, eps=1e-6)
            self.params_action = [p for p in self.policy_action_net.parameters()]
        else:
            self.optimizer = optim.RMSprop(policy_net.parameters(),
                lr = args.lrate, alpha=0.97, eps=1e-6)
            self.params = [p for p in self.policy_net.parameters()]
        self.episode_counter = 0

        tracemalloc.start()
        self.reset_memory_peak()


    def get_episode(self, epoch):
        if self.args.hetcomm:
            episode_action = []
            episode_perception = []
        else:
            episode = []

        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()

        stat = dict()
        info = dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        for t in range(self.args.max_steps):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    if self.args.hetcomm:
                        prev_hid_action = self.policy_action_net.init_hidden(batch_size=state.shape[0])
                        prev_hid_perception = self.policy_perception_net.init_hidden(batch_size=state.shape[0])

                    else:
                        prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                if self.args.hetcomm:
                    # need to fix prev_hid to be full size before input
                    hids = torch.cat((prev_hid_perception[0], prev_hid_action[0]), dim=0)
                    cels = torch.cat((prev_hid_perception[1], prev_hid_action[1]), dim=0)
                    prev_hid =(hids, cels)
                    x_action = state[:, 2, :].reshape(1, 1, 29)
                    x_perception = state[:, :2, :]
                    x_action = [x_action, prev_hid]
                    x_perception = [x_perception, prev_hid]
                    action_out_action, value_action, prev_hid_action = self.policy_action_net(x_action, info)
                    action_out_perception, value_perception, prev_hid_perception = self.policy_perception_net(
                        x_perception, info)
                    action_out =[torch.cat((action_out_perception[0], action_out_action[0]),dim=1)]
                else:
                    x = [state, prev_hid]
                    action_out, value, prev_hid = self.policy_net(x, info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            elif self.args.hetgat:
                if self.args.hetgat_a2c:
                    if t == 0:
                        # TODO: fix prev_hid size
                        if self.args.tensor_obs:
                            prev_hid = self.policy_net.init_hidden(batch_size=1)
                        else:
                            prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                    # x = [state, prev_hid]
                    if self.args.tensor_obs:
                        old_form_state = torch.tensor(self.env.env.get_obs()).reshape((1,self.args.nfriendly_P + self.args.nfriendly_A,-1))
                        x = [old_form_state, state, prev_hid]
                    else:
                        x = [state, prev_hid]
                    action_out, value, prev_hid = self.policy.batch_select_action_universal(x, self.stats['num_episodes'])
                    # value hardcoded for trainer bookkeeping
                    value = torch.tensor([[0], [0], [0]])
                    # if (t + 1) % self.args.detach_gap == 0:
                    #     prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    if (t + 1) % self.args.detach_gap == 0:
                        prev_hid['P_s'] = (prev_hid['P_s'][0].detach(), prev_hid['P_s'][1].detach())
                        prev_hid['P_o'] = (prev_hid['P_o'][0].detach(), prev_hid['P_o'][1].detach())
                        prev_hid['A_s'] = (prev_hid['A_s'][0].detach(), prev_hid['A_s'][1].detach())

                else:
                    if t == 0:
                        prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])
                    x = [state, prev_hid]
                    action_out, prev_hid = self.policy.batch_select_action_universal(x, self.stats['num_episodes'])
                    value = torch.tensor([[0], [0], [0]])
                    # if (t + 1) % self.args.detach_gap == 0:
                    #     prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
            else:
                x = state
                if self.args.hetcomm:
                    x_action, x_perception = x[:,2,:], x[:,:2,:]
                    action_out_action, value_action = self.policy_action_net(x_action, info)
                    action_out_perception, value_perception = self.policy_perception_net(x_perception, info)
                else:
                    action_out, value = self.policy_net(x, info)

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            next_state, reward, done, info = self.env.step(actual)

            if self.args.display:
                self.env.display()

            ######################################## add entropies to entropy buffer ############################################################
            if self.args.hetgat:
                self.policy.batch_rewards[self.stats['num_episodes']].append(reward)
                self.policy.append_log_probs_properly(actual)
                # self.entropy_buffer_p.append(self.policy.entropies_p)
                # self.entropy_buffer_a.append(self.policy.entropies_a)

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]

            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()
            if self.args.hetcomm:
                trans = Transition(x_perception, [action[0][:2]], action_out_perception,  value_perception, episode_mask, episode_mini_mask, next_state, reward, misc)
                episode_perception.append(trans)
                trans = Transition(x_action, [action[0][2]], action_out_action, value_action, episode_mask, episode_mini_mask, next_state, reward, misc)
                episode_action.append(trans)
            else:
                trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
                episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']
        # print(stat['num_steps'])

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']
            if self.args.hetcomm:
                episode_action[-1] = episode_action[-1]._replace(reward = episode_action[-1].reward + reward)
                episode_perception[-1] = episode_perception[-1]._replace(reward = episode_perception[-1].reward + reward)
            else:
                episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        if self.args.hetcomm:
            return ([episode_perception,episode_action], stat)
        else:
            return (episode, stat)

    def compute_grad(self, batch):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions
        if self.args.hetcomm:
            batch_perception = batch[1]
            batch = batch[0]
            batch_size = len(batch.state)

        else:
            n = self.args.nagents
            batch_size = len(batch.state)

        if not self.args.hetgat:
            if not self.args.hetcomm:
                rewards = torch.Tensor(batch.reward)
                episode_masks = torch.Tensor(batch.episode_mask)
                episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
                actions = torch.Tensor(batch.action)
                actions = actions.transpose(1, 2).view(-1, n, dim_actions)

                # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
                # old_actions = old_actions.view(-1, n, dim_actions)
                # print(old_actions == actions)

                # can't do batch forward.
                values = torch.cat(batch.value, dim=0)
                action_out = list(zip(*batch.action_out))
                action_out = [torch.cat(a, dim=0) for a in action_out]

                alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)

                coop_returns = torch.Tensor(batch_size, n)
                ncoop_returns = torch.Tensor(batch_size, n)
                returns = torch.Tensor(batch_size, n)
                deltas = torch.Tensor(batch_size, n)
                advantages = torch.Tensor(batch_size, n)
                values = values.view(batch_size, n)

                prev_coop_return = 0
                prev_ncoop_return = 0
                prev_value = 0
                prev_advantage = 0

                for i in reversed(range(rewards.size(0))):
                    coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
                    ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

                    prev_coop_return = coop_returns[i].clone()
                    prev_ncoop_return = ncoop_returns[i].clone()

                    returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                                + ((1 - self.args.mean_ratio) * ncoop_returns[i])


                for i in reversed(range(rewards.size(0))):
                    advantages[i] = returns[i] - values.data[i]

                if self.args.normalize_rewards:
                    advantages = (advantages - advantages.mean()) / advantages.std()

                if self.args.continuous:
                    action_means, action_log_stds, action_stds = action_out
                    log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
                else:
                    log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
                    actions = actions.contiguous().view(-1, dim_actions)

                    if self.args.advantages_per_action:
                        log_prob = multinomials_log_densities(actions, log_p_a)
                    else:
                        log_prob = multinomials_log_density(actions, log_p_a)

                if self.args.advantages_per_action:
                    action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
                    action_loss *= alive_masks.unsqueeze(-1)
                else:
                    action_loss = -advantages.view(-1) * log_prob.squeeze()
                    action_loss *= alive_masks

                action_loss = action_loss.sum()
                stat['action_loss'] = action_loss.item()

                # value loss term
                targets = returns
                value_loss = (values - targets).pow(2).view(-1)
                value_loss *= alive_masks
                value_loss = value_loss.sum()

                stat['value_loss'] = value_loss.item()
                loss = action_loss + self.args.value_coeff * value_loss

                if not self.args.continuous:
                    # entropy regularization term
                    entropy = 0
                    for i in range(len(log_p_a)):
                        entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
                    stat['entropy'] = entropy.item()
                    if self.args.entr > 0:
                        loss -= self.args.entr * entropy
            else:
                for e, b in enumerate([batch_perception, batch]):
                    if e == 0:
                        n = 2
                    else:
                        n=1
                    rewards = torch.Tensor(b.reward)
                    episode_masks = torch.Tensor(b.episode_mask)
                    episode_mini_masks = torch.Tensor(b.episode_mini_mask)
                    actions = torch.Tensor(b.action)
                    if n==2:
                        actions = actions.transpose(1, 2).view(-1, n, dim_actions)
                    else:
                        actions = actions.reshape(-1,1,1)
                    # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
                    # old_actions = old_actions.view(-1, n, dim_actions)
                    # print(old_actions == actions)

                    # can't do batch forward.
                    values = torch.cat(b.value, dim=0)
                    action_out = list(zip(*b.action_out))
                    action_out = [torch.cat(a, dim=0) for a in action_out]

                    alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in b.misc])).view(-1)

                    coop_returns = torch.Tensor(batch_size, 3)
                    ncoop_returns = torch.Tensor(batch_size, 3)
                    returns = torch.Tensor(batch_size, 3)
                    deltas = torch.Tensor(batch_size, 3)
                    advantages = torch.Tensor(batch_size, n)
                    values = values.view(batch_size, n)

                    prev_coop_return = 0
                    prev_ncoop_return = 0
                    prev_value = 0
                    prev_advantage = 0

                    for i in reversed(range(rewards.size(0))):
                        coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
                        ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * \
                                           episode_mini_masks[i]

                        prev_coop_return = coop_returns[i].clone()
                        prev_ncoop_return = ncoop_returns[i].clone()

                        returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                                     + ((1 - self.args.mean_ratio) * ncoop_returns[i])

                    for i in reversed(range(rewards.size(0))):
                        if n==2:
                            advantages[i] = returns[i][:2] - values.data[i]
                        else:
                            advantages[i] = returns[i][2] - values.data[i]

                    if self.args.normalize_rewards:
                        advantages = (advantages - advantages.mean()) / advantages.std()

                    if self.args.continuous:
                        action_means, action_log_stds, action_stds = action_out
                        log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
                    else:
                        log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
                        actions = actions.contiguous().view(-1, dim_actions)

                        if self.args.advantages_per_action:
                            log_prob = multinomials_log_densities(actions, log_p_a)
                        else:
                            log_prob = multinomials_log_density(actions, log_p_a)

                    if self.args.advantages_per_action:
                        action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
                        action_loss *= alive_masks.unsqueeze(-1)
                    else:
                        action_loss = -advantages.view(-1) * log_prob.squeeze()
                        # action_loss *= alive_masks

                    action_loss = action_loss.sum()
                    stat['action_loss'] = action_loss.item()

                    # value loss term
                    targets = returns
                    if n==2:
                        value_loss = (values - targets[:,:n]).pow(2).view(-1)
                    else:
                        value_loss = (values - targets[:,2]).pow(2).view(-1)

                    # value_loss *= alive_masks
                    value_loss = value_loss.sum()

                    stat['value_loss'] = value_loss.item()
                    loss = action_loss + self.args.value_coeff * value_loss

                    if not self.args.continuous:
                        # entropy regularization term
                        entropy = 0
                        for i in range(len(log_p_a)):
                            entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
                        stat['entropy'] = entropy.item()
                        if self.args.entr > 0:
                            loss -= self.args.entr * entropy
                    loss.backward()
                    if n==1:
                        self.optimizer_action.step()
                    else:
                        self.optimizer_perception.step()

                return stat

        if self.args.hetgat:
            if self.policy.per_class_critic:
                loss = self.policy.batch_finish_per_class(
                    self.stats['num_episodes'], num_P=self.args.nfriendly_P,
                    num_A=self.args.nfriendly_A, sim_time=self.args.max_steps)
            elif self.policy.per_agent_critic:
                loss = self.policy.batch_finish_per_agent(
                    self.stats['num_episodes'], num_P=self.args.nfriendly_P,
                    num_A=self.args.nfriendly_A, sim_time=self.args.max_steps)
            else:
                loss = self.policy.batch_finish_episode(
                    self.stats['num_episodes'], num_P=self.args.nfriendly_P,
                    num_A=self.args.nfriendly_A, sim_time=self.args.max_steps)
            # self.episode_counter = self.stats['num_episodes']
            if self.args.hetgat_a2c:
                stat['action_loss'] = loss['policy'].item()
                stat['value_loss'] = loss['critic'].item()
            else:
                stat['action_loss'] = loss.item()
        else:
            loss.backward()

        return stat

    def run_batch(self, epoch):
        if self.args.hetcomm:
            batch = []
            batch_perception = []
        else:
            batch = []

        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            if self.args.hetcomm:
                batch+=episode[1]
                batch_perception+=episode[0]
            else:
                batch += episode

        # TODO: a lot of this can be cut for hetgat
        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))

        if self.args.hetcomm:
            batch_perception = Transition(*zip(*batch_perception))

        if self.args.hetcomm:
            return [batch,batch_perception], self.stats
        else:
            return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        self.reset_memory_peak()

        batch, stat = self.run_batch(epoch)
        if self.args.hetcomm:
            self.optimizer_perception.zero_grad()
            self.optimizer_action.zero_grad()
        else:
            self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        # is not self.args.hetgat and not self.args.hetcomm:
        if not self.args.hetcomm:
            for p in self.params:
                if p._grad is not None:
                    p._grad.data /= stat['num_steps']

            self.optimizer.step()

        return stat, np.array([self.cpu_memory_peak]), np.array([self.gpu_memory_peak])

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def reset_memory_peak(self):
        tracemalloc.stop()
        tracemalloc.start()
        torch.cuda.reset_peak_memory_stats()

        self.cpu_memory_peak = 0
        self.gpu_memory_peak = 0

    def set_cpu_memory_peak(self, cpu_mem):
        if cpu_mem > self.cpu_memory_peak:
            self.cpu_memory_peak = cpu_mem

    def set_gpu_memory_peak(self, gpu_mem):
        if gpu_mem > self.gpu_memory_peak:
            self.gpu_memory_peak = gpu_mem

    def get_memory_peak(self):
        self.set_cpu_memory_peak(tracemalloc.get_traced_memory()[1])
        self.set_gpu_memory_peak(torch.cuda.max_memory_allocated(device=torch.device('cuda')))

        return self.cpu_memory_peak, self.gpu_memory_peak
