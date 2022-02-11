import argparse
import signal
import sys
import time
import signal
import argparse

# addrss1 = 'C:\\Users\\ESI\\Documents\\PhD Georgia Tech\\PhD Research - Robotics AI\\Reports\\10- HetGAT for Learning Efficient Communication\\GitHubCodes\\HetGAT_MARL_Communication'
# addrss2 = 'C:\\Users\\ESI\\Documents\\PhD Georgia Tech\\PhD Research - Robotics AI\\Reports\\10- HetGAT for Learning Efficient Communication\\GitHubCodes\\HetGAT_MARL_Communication\\Predator_Prey\\'
# addrss3 = 'C:\\Users\\ESI\\Documents\\PhD Georgia Tech\\PhD Research - Robotics AI\\Reports\\10- HetGAT for Learning Efficient Communication\\GitHubCodes\\HetGAT_MARL_Communication\\test\\IC3Net'
# sys.path.insert(0, addrss1)
# sys.path.insert(0, addrss2)
# sys.path.insert(0, addrss3)
# sys.path.insert(0,'/home/rohanpaleja/PycharmProjects/HetGAT_MARL_Communication/')
# sys.path.insert(0, '//home/rohanpaleja/PycharmProjects/HetGAT_MARL_Communication/Predator_Prey/')
# sys.path.insert(0,'/home/rohanpaleja/PycharmProjects/HetGAT_MARL_Communication/test/IC3Net/')
# sys.path.insert(0,'/nethome/msklar3/HetGAT_MARL_Communication/')
# sys.path.insert(0,'/nethome/msklar3/HetGAT_MARL_Communication/Predator_Prey/')
# sys.path.insert(0,'/nethome/msklar3/HetGAT_MARL_Communication/test/IC3Net/')

import tracemalloc

import numpy as np
import torch
import visdom

import data
from action_utils import parse_action_args
from comm import CommNetMLP
from hetgat.policy import A2CPolicy, PGPolicy
from models import *
from multi_processing import MultiProcessTrainer
from trainer import Trainer
from eval_trainer import EvalTrainer
from utils import *

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=16,
                    help='How many processes to run')
# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')

parser.add_argument('--use_binary', default=False, action='store_true',
                    help='Wheather to use binarization in hetgat')
parser.add_argument('--msg_dim', default=16, type=int,
                    help='Message size of binarization')

# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed')  # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
parser.add_argument('--comm_range_P', default=-1, type=int,
                    help='range perception agents can communicate in (-1 for infinite)')
parser.add_argument('--comm_range_A', default=-1, type=int,
                    help='range action agents can communicate in (-1 for infinite)')
parser.add_argument('--lossy_comm', action='store_true', default=False,
                    help='communication is lost as range approaches maximum')
parser.add_argument('--min_comm_loss', default=0.0, type=float,
                    help='percentage of communication lost at no range')
parser.add_argument('--max_comm_loss', default=0.3, type=float,
                    help='percentage of communication lost at max range')

# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--save', default='tst', type=str,
                    help='save the model after training')
parser.add_argument('--save_every', default=10, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')
parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")
parser.add_argument('--use_cuda', action='store_true', default=False,
                    help='use cuda instead of cpu')

# hetgat specific args
parser.add_argument('--hetgat', action='store_true', default=False,
                    help="enable hetgat model")
parser.add_argument('--lr_gamma', type=float, default=0.1,
                    help="lr gamma parameter")
parser.add_argument('--hetgat_a2c', action='store_true', default=False,
                    help="enable hetgat a2c model")
parser.add_argument('--eval', action='store_true', default=False,
                    help='evaluate a model')
parser.add_argument('--eval_string', default='', type=str,
                    help='string that will be used to save result')

# CommNet specific args
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--hetcomm', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                         + ' Default 10000 (very high)')
parser.add_argument('--total_state_action_in_batch', default=500, type=int,
                    help='number of s,a in a batch.'
                         + ' Default 500 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log prob for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')

init_args_for_env(parser)
args = parser.parse_args()

if args.comm_range_P == -1 or args.comm_range_A == -1:
    args.lossy_comm = False

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games
    if args.env_name == "traffic_junction":
        args.comm_action_one = True

if not hasattr(args, 'nfriendly_P') and not hasattr(args, 'nfriendly_A'):
    args.nfriendly_A = 1
    args.nfriendly_P = args.nagents - args.nfriendly_A

args.nfriendly = args.nfriendly_P + args.nfriendly_A

# Enemy comm
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

env = data.init(args.env_name, args, False)

num_inputs = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)):  # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'
if args.hetcomm:
    args.recurrent = True
    args.rnn_type = 'LSTM'

parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0, 10000)

print(args)

if args.commnet:
    policy_net = CommNetMLP(args, num_inputs, 4)
    print(policy_net)
elif args.hetcomm:
    policy_action_net = HetCommNetMLP(args, num_inputs, 1)
    policy_action_net.action = True
    policy_perception_net = HetCommNetMLP(args, num_inputs, 2)
    # print(policy_net)
elif args.hetgat:
    pos_len = args.dim ** 2
    SSN_state_len = 4

    in_dim_raw = {'vision': args.vision,
                  'P': pos_len + SSN_state_len,
                  'A': pos_len,
                  'state': SSN_state_len
                  }
    in_dim = {'P': pos_len + SSN_state_len,
              'A': pos_len,
              'state': SSN_state_len}
    hid_dim = {'P': 16,
               'A': 16,
               'state': 16}
    out_dim = {'P': 5,
               'A': 6,
               'state': 8}
    with_two_state = True
    # if with_two_state:
    #     in_dim['state'] = SSN_state_len
    num_heads = 4

    device_name = 'cuda' if args.use_cuda else 'cpu'
    device = torch.device(device_name)

    obs = None if not hasattr(args, 'vision') else (2 * args.vision + 1) ** 2
    tensor_obs = None if not hasattr(args, 'tensor_obs') else args.tensor_obs

    milestones = [200, 400]
    if args.hetgat_a2c:
        policy = A2CPolicy(in_dim_raw, in_dim, hid_dim, out_dim, args.nfriendly_P,
                           args.nfriendly_A, num_heads=num_heads, msg_dim=args.msg_dim,
                           device=device, gamma=args.gamma, lr=args.lrate, weight_decay=0,
                           milestones=milestones, lr_gamma=0.1, use_real=(not args.use_binary),
                           use_CNN=False, use_tanh=False, per_class_critic=True,
                           per_agent_critic=False, with_two_state=with_two_state, obs=obs,
                           comm_range_P=args.comm_range_P, comm_range_A=args.comm_range_A,
                           lossy_comm=args.lossy_comm, min_comm_loss=args.min_comm_loss,
                           max_comm_loss=args.max_comm_loss, tensor_obs=tensor_obs, action_vision=args.A_vision)
    else:
        policy = PGPolicy(in_dim_raw, in_dim, hid_dim, out_dim, num_heads=num_heads,
                          device=device, gamma=args.gamma, lr=args.lrate,
                          weight_decay=0, milestones=milestones, lr_gamma=0.1,
                          use_real=True, use_CNN=False)
    policy_net = policy.model
elif args.random:
    policy_net = Random(args, num_inputs)
elif args.recurrent:
    policy_net = RNN(args, num_inputs)
else:
    policy_net = MLP(args, num_inputs)

if args.hetcomm:
    pass
else:
    if not args.display:
        display_models([policy_net])

# share parameters among threads, but not gradients
if args.hetcomm:
    for p in policy_action_net.parameters():
        p.data.share_memory_()

    for p in policy_perception_net.parameters():
        p.data.share_memory_()

else:
    for p in policy_net.parameters():
        p.data.share_memory_()

if args.nprocesses > 1:
    if args.hetgat:
        if __name__ == '__main__':
            trainer = MultiProcessTrainer(args,
                                          lambda: Trainer(args, policy_net, data.init(args.env_name, args), policy))
    else:
        if __name__ == '__main__':
            trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
else:
    if args.hetgat:
        if args.eval:
            trainer = EvalTrainer(args, policy_net, data.init(args.env_name, args), policy)
        else:
            trainer = Trainer(args, policy_net, data.init(args.env_name, args), policy)
    elif args.hetcomm:
        trainer = Trainer(args, [policy_perception_net, policy_action_net], data.init(args.env_name, args))
    else:
        if args.eval:
            trainer = EvalTrainer(args, policy_net, data.init(args.env_name, args))
        else:
            trainer = Trainer(args, policy_net, data.init(args.env_name, args))
if args.hetcomm:
    disp_trainer = Trainer(args, [policy_perception_net, policy_action_net], data.init(args.env_name, args, False))
else:
    disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))

torch.manual_seed(args.seed)
np.random.seed(args.seed)

disp_trainer.display = True


def disp():
    x = disp_trainer.get_episode()


log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_count'] = LogField(list(), True, 'epoch', 'num_steps')

if args.plot:
    vis = visdom.Visdom(env=args.plot_env)


def run(num_epochs):
    num_episodes = 0

    global_cpu_mem_peak = np.zeros((args.nprocesses,))
    global_gpu_mem_peak = np.zeros((args.nprocesses,))

    for ep in range(num_epochs):
        tracemalloc.start()
        epoch_cpu_mem_peak = np.zeros((args.nprocesses,))
        epoch_gpu_mem_peak = np.zeros((args.nprocesses,))

        epoch_begin_time = time.time()
        stat = dict()
        for n in range(args.epoch_size):
            print("[Epoch] batch", n)
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True

            s, cpu_mem_peak, gpu_mem_peak = trainer.train_batch(ep)

            merge_stat(s, stat)
            trainer.display = False
            num_episodes += stat['num_episodes']

            epoch_cpu_mem_peak = np.maximum(epoch_cpu_mem_peak, cpu_mem_peak)
            epoch_gpu_mem_peak = np.maximum(epoch_gpu_mem_peak, gpu_mem_peak)

        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            elif k == 'enemy_count':
                v.data.append(stat.get(k, []))
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        epoch_cpu_mem_peak[0] = tracemalloc.get_traced_memory()[1]
        epoch_gpu_mem_peak[0] = torch.cuda.max_memory_allocated(device=torch.device('cuda'))

        global_cpu_mem_peak = np.maximum(epoch_cpu_mem_peak, global_cpu_mem_peak)
        global_gpu_mem_peak = np.maximum(epoch_gpu_mem_peak, global_gpu_mem_peak)

        np.set_printoptions(precision=2)
        print('Epoch {}\tReward {}\tTime {:.2f}s, Episodes {}, CPU Memory Peak {}MB, GPU Memory Peak {}MB'.format(
            epoch, stat['reward'], epoch_time, num_episodes, epoch_cpu_mem_peak / 10 ** 6, epoch_gpu_mem_peak / 10 ** 6
        ))

        if 'enemy_reward' in stat.keys():
            print('Enemy-Reward: {}'.format(stat['enemy_reward']))
        if 'add_rate' in stat.keys():
            print('Add-Rate: {:.2f}'.format(stat['add_rate']))
        if 'success' in stat.keys():
            print('Success: {:.2f}'.format(stat['success']))
        if 'steps_taken' in stat.keys():
            print('Steps-taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat.keys():
            print('Comm-Action: {}'.format(stat['comm_action']))
        if 'enemy_comm' in stat.keys():
            print('Enemy-Comm: {}'.format(stat['enemy_comm']))
        if 'enemy_count' in stat.keys():
            print('Average-Enemy-Count: {}'.format(np.average(stat['enemy_count'])))

        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                             win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

        if args.save_every and ep and args.save != '' and ep % args.save_every == 0:
            # fname, ext = args.save.split('.')
            # save(fname + '_' + str(ep) + '.' + ext)
            save(args.save + '_' + str(ep), args)

        # if args.save != '':
        #     save(args.save + '_' + str(ep))

        print('Global CPU Memory Peak {}MB\nGlobal GPU Memory Peak {}MB'.format(
            epoch_cpu_mem_peak / 10 ** 6, epoch_gpu_mem_peak / 10 ** 6
        ))


def save(path, args):
    print(path)
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    if args.env_name == 'fire_commander':
        d['reward_params'] = {'False_Water_Pen': env.env.FALSE_WATER_DROP_PENALTY,
                              'CAPTURE_REWARD': env.env.CAPTURE_REWARD,
                              'NONSOURCE_CAPTURE_REWARD': env.env.NONSOURCE_CAPTURE_REWARD,
                              'DISCOVER_SOURCE_REWARD': env.env.DISCOVER_SOURCE_REWARD,
                              'DISCOVER_NONSOURCE_REWARD': env.env.DISCOVER_NONSOURCE_REWARD,
                              'FIRE_PENALTY': env.env.FIRE_PENALTY,
                              'TIMESTEP_PENALTY': env.env.TIMESTEP_PENALTY}

    d['seed'] = args.seed
    torch.save(d, path + '.tar')
    import matplotlib.pyplot as plt
    plt.plot(d['log']['steps_taken'].data)
    plt.xlabel('Epochs')
    plt.ylabel('Steps Taken')
    plt.title('Reward: ' + 'Water_Pen' + str(env.env.FALSE_WATER_DROP_PENALTY) + ' ,CAPTURE:' + str(
        env.env.CAPTURE_REWARD) + ' ,NONSOURCE_CAPTURE:' + str(env.env.NONSOURCE_CAPTURE_REWARD) + '\n DISC_SOURCE:' + str(
        env.env.DISCOVER_SOURCE_REWARD) + ' ,DISC_NONSOURCE:' + str(env.env.DISCOVER_NONSOURCE_REWARD) + ' ,FIRE_PEN:' + str(
        env.env.FIRE_PENALTY) + ' ,TIME_PEN:' + str(env.env.TIMESTEP_PENALTY), fontsize=8)
    plt.savefig(path + '.png')
    plt.clf()


def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])


def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Exiting gracefully.')
    if args.display:
        env.end_display()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    if args.load != '':
        load(args.load)

    run(args.num_epochs)

    if args.display:
        env.end_display()

    if args.save != '':
        save(args.save)

    if sys.flags.interactive == 0 and args.nprocesses > 1:
        trainer.quit()
        import os

        os._exit(0)
