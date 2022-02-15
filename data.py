import sys
import gym
import ic3net_envs
from env_wrappers import GymWrapper

def init(env_name, args, final_init=True):
    if env_name == 'levers':
        env = gym.make('Levers-v0')
        env.multi_agent_init(args.total_agents, args.nagents)
        env = GymWrapper(env)
    elif env_name == 'number_pairs':
        env = gym.make('NumberPairs-v0')
        m = args.max_message
        env.multi_agent_init(args.nagents, m)
        env = GymWrapper(env)
    elif env_name == 'predator_prey':
        env = gym.make('PredatorPrey-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'traffic_junction':
        env = gym.make('TrafficJunction-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'starcraft':
        env = gym.make('StarCraftWrapper-v0')
        env.multi_agent_init(args, final_init)
        env = GymWrapper(env.env)
    elif env_name == 'predator_capture':
        env = gym.make('PredatorCapture-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'fire_commander':
        env = gym.make('FireCommander-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    else:
        raise RuntimeError("wrong env name")

    return env
