
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
    """ Add HetNet specific parameters. """

    parser.add_argument("--n_types", type=int, default=1,
                        help="Number of different heterogeneous types.")

    parser.add_argument("--num_P", type=int, default=3,
                        help="Number of perception agents for PP/PCP/FC.")

    parser.add_argument("--num_A", type=int, default=0,
                        help="Number of action agents for PP/PCP/FC.")

    parser.add_argument("--num_X", type=int, default=0,
                        help="Number of X agents for 3-class problems.")

    parser.add_argument("--dim", type=int, default=5,
                        help="Size of grid dimension.")

    parser.add_argument("--vision", type=int, default=0,
                        help="vision setting for the predator agents")

    parser.add_argument("--tensor_obs", default=False, action='store_true',
                        help="Utilize tensorized observation space inputs.")

    parser.add_argument("--episode_limit", type=int, default=80,
                        help="Max steps in a single episode before timing out")

    parser.add_argument('--with_state', default=False, action='store_true',
                        help="whether or not to add state in the graph composition")

    parser.add_argument("--with_two_state", type=int, default=True,
                        help="Include two state nodes in the graph, ignored if --with_state is False")

    parser.add_argument("--load_critic", default=False, action='store_true',
                        help="whether to load the critic when transferring")

    parser.add_argument("--upsample_x2", default=False, action='store_true',
                        help="utilize upsampling on the state")

    parser.add_argument('--eval_string', default='', type=str,
                        help='string that will be used to save result')
    parser.add_argument('--eval_config', default='', type=str,
                        help='holds all of the evaluation starting conditions')

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

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
    runner.eval_init_conditions()

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
    main(sys.argv[1:])
