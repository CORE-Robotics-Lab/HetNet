
import sys
import os
import wandb
import socket
import setproctitle

from pathlib import Path


import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.predator_prey.PredatorPreyEnv import PredatorPrey
from onpolicy.envs.predator_prey.PredatorCaptureEnv import PredatorCapture
from onpolicy.envs.fire_commander.fire_commander_env import FireCommanderEnv

from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


"""Eval script for PredatorPrey."""
def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "PredatorPrey":
                env = PredatorPrey(all_args)
            elif all_args.env_name == "PredatorCapture":
                env = PredatorCapture(all_args)
            elif all_args.env_name == "FireCommander":
                env = FireCommanderEnv(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "PredatorPrey":
                env = PredatorPrey(all_args)
            elif all_args.env_name == "PredatorCapture":
                env = PredatorCapture(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    """add hetnet specific parameters"""
    parser.add_argument("--n_types", type=int, default=1,
                        help="number of different heterogeneous types")

    parser.add_argument("--num_P", type=int, default=3,
                        help="number of perception agents for PP/PCP")

    parser.add_argument("--num_A", type=int, default=0,
                        help="number of action agents for PP/PCP")

    parser.add_argument("--num_X", type=int, default=0,
                        help="number of X agents for 3-class problems")

    parser.add_argument("--dim", type=int, default=5,
                        help="size of grid dimension")

    parser.add_argument("--episode_limit", type=int, default=80,
                        help="max steps in a single episode before timing out")


    parser.add_argument('--with_state', default=False, action='store_true',
                        help="whether or not to add state in the graph composition")

    parser.add_argument("--with_two_state", type=int, default=True,
                        help="include two state nodes in the graph, ignored if --with_state is False")


    parser.add_argument("--tensor_obs", default=False, action='store_true', help="utilize tensorized input")


    parser.add_argument("--vision", type=int, default=0,
                        help="vision setting for the predator agents")

    parser.add_argument("--upsample_x2", default=False, action='store_true', help="utilize upsampling on the state")

    parser.add_argument("--load_critic", default=False, action='store_true',
                        help="whether to load the critic when transferring")

    # FireCommander parameters
    parser.add_argument('--nfires', type=int, default=1,
                        help="Original number of fires")

    parser.add_argument('--fire_spread_off', action="store_true", default=False,
                        help="Turn off fire spread")

    parser.add_argument('--no_stay', action="store_true", default=False,
                        help="Whether predators have an action to stay in place")

    parser.add_argument('--mode', default='mixed', type=str,
                        help='cooperative|competitive|mixed (default: mixed)')

    parser.add_argument('--max_wind_speed', type=float, default=None,
                        help="Maximum speed of wind (default: grid size / 5)")

    parser.add_argument('--reward_type', type=int, default=0,
                        help="Reward type to use (0 -> negative timestep, 1 -> positive capture, 2 -> water dump penalty)")

    all_args = parser.parse_known_args(args)[0]



    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    # all_args.use_wandb = False
    print(all_args.use_wandb)
    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "hetgat_mappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(all_args.cuda_device)
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.env_name,
                         dir=str(run_dir),
                         job_type="training",
                         #  resume=True,
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_A + all_args.num_P

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.predator_runner import PredatorRunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":


    debug = False
    if debug:
        args = ['--env_name',
                # 'PredatorPrey',
                'PredatorCapture',

                '--num_P', '3',
                '--num_A', '2',
                '--dim', '10',
                '--n_types', '2',
                '--vision', '1',
                '--algorithm_name',
                'hetgat_mappo',
                # 'rmappo',
                '--experiment_name', 'hetgat_ppo_3p',
                '--seed', '0',
                '--use_wandb',
                # '--tensor_obs', False, # comment this to use tensor_obs; leave uncommented if using vectorized
                '--with_state', False,  # use SSN or not...
                '--n_training_threads', '128',  # parallel actors (set 128)
                '--n_rollout_threads', '1',  # parallel environments (set 1)
                '--num_mini_batch', '1',
                '--episode_length', '500',  # num state-action pairs within an update episode
                '--num_env_steps', '5000000',  # total num state-action pairs for training
                '--ppo_epoch', '5',
                '--use_value_active_masks', # TODO: what is default
                # '--use_eval',
                '--use_recurrent_policy',
                '--use_LSTM',
                '--entropy_coef', '0.01',
                '--hidden_size', '128'

                # '--cuda', False,
                # '--wandb_name', 'johnzhang', '--user_name', 'johnzhang'
                ]


        main(args=args)
    else:
        main(sys.argv[1:])
        # print(sys.argv[1:])
        # main(sys.argv[1:])























