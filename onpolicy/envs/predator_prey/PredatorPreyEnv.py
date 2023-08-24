
import numpy as np
from gym.spaces import Discrete

from .multiagentenv import MultiAgentEnv

class PredatorPrey(MultiAgentEnv):
    def __init__(self, args, n_enemies=1, vision=0, moving_prey=False, no_stay=False, mode="mixed", enemy_comm=False,
                 tensor_obs=False, second_reward_scheme=False,):
        self.__version__ = "0.0.1"

        # default params
        self.OUTSIDE_CLASS = 1
        self.PREY_CLASS = 2
        self.PREDATOR_CLASS = 3
        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0
        self.POS_PREY_REWARD = 0.05
        self.episode_over = False

        # param arguments
        self.args = args
        self.nfriendly_P = args.num_P
        self.vision = vision
        self.nprey = n_enemies
        self.npredator = args.num_P
        dim = args.dim
        self.dims = (dim,dim)
        self.dim = dim
        self.mode = mode
        self.enemy_comm = enemy_comm
        self.stay = not no_stay
        self.tensor_obs = tensor_obs
        self.second_reward_scheme = second_reward_scheme
        self.episode_limit = args.episode_limit
        self.n_agents = self.nfriendly_P
        self.nagents = self.nfriendly_P
        self._episode_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.total_steps = 0
        
        if moving_prey:
            raise NotImplementedError

        # Define what an agent can do -
        if self.stay:
            self.naction = 5
        else:
            self.naction = 4

        self.BASE = (self.dims[0] * self.dims[1])
        self.OUTSIDE_CLASS += self.BASE
        self.PREY_CLASS += self.BASE
        self.PREDATOR_CLASS += self.BASE

        # 4 is 4 types of classes!
        self.num_classes = 4
        self.vocab_size = self.BASE + self.num_classes
        
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.n_agents):
            self.action_space.append(Discrete(self.naction))
            self.observation_space.append(self.get_obs_size())
            self.share_observation_space.append(self.get_state_size())

    def step(self, actions):
        """
        The agents take a step in the environment.

        Parameters
        ----------
        action : list/ndarray of length m, containing the indexes of what lever each 'm' chosen agents pulled.

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :
            state ():
            reward (float) : Ratio of Number of discrete levers pulled to total number of levers.
            episode_over (bool) : Will be true as episode length is 1
            info (dict) : diagnostic information useful for debugging.
        """

        if self.episode_over:
            raise RuntimeError("Episode is done")

        # action = np.array(actions.cpu()).squeeze()
        action = actions
        action = np.atleast_1d(action)
        infos = [{} for i in range(self.n_agents)]
        dones = np.zeros((self.n_agents), dtype=bool)
        bad_transition = False
        for i, a in enumerate(action):
            self._take_action(i, a)

        self._episode_steps += 1
        self.total_steps += 1

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."

        self.episode_over = False
        self.obs = self.get_obs()
        self.state = self.get_state()
       
        if self._episode_steps >= self.episode_limit:
            self.episode_over = True
            self.battles_game += 1
            self.timeouts += 1
            bad_transition = True

        reward = self._get_reward()
        
        for i in range(self.n_agents):
            infos[i] = {"battles_won": self.battles_won,
                        "battles_game": self.battles_game,
                        "battles_draw": self.timeouts,
                        "bad_transition": bad_transition,
                        "won": self.battle_won,
                        "episode_steps": self._episode_steps
                        }
            if self.episode_over:
                dones[i] = True
                
        return self.obs, self.state, reward, dones, infos, self.get_avail_actions()

    def get_obs(self):
        """ Returns all agent observations in a list """
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        for i, p in enumerate(self.predator_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS] += 1

        for i, p in enumerate(self.prey_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREY_CLASS] += 1

        obs = []
        for p in self.predator_loc:
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x].reshape((-1)))

        if self.enemy_comm:
            for p in self.prey_loc:
                slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
                slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
                obs.append(self.bool_base_grid[slice_y, slice_x])
        # obs = np.stack(obs)
        return obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]
    
    def get_obs_size(self):
        """ Returns the shape of the observation """
        return [self.vocab_size]

    def get_state(self):
        """ Global state """
        state = np.vstack((self.predator_loc, self.prey_loc))
        state = np.reshape(state, (-1)) / (self.dim-1) # flatten and normalize
        state = np.append(state, self._episode_steps/self.episode_limit)
        self.state = []
        for i in range(self.n_agents):
            self.state.append(state)
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state
        """
        state_size = (self.nagents+self.nprey)*2 + 1
        return [state_size]

    def get_avail_actions(self):
        """ All actions are available
        """
        avail_actions = []
        for i in range(self.npredator):
            avail_actions.append([1]*self.naction)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id
        """
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.naction

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.reached_prey = np.zeros(self.npredator)
        self._episode_steps = 0
        self.battle_won = False

        # Locations
        locs = self._get_cordinates()
        self.predator_loc, self.prey_loc = locs[:self.npredator], locs[self.npredator:]

        self._set_grid()

        # stat - like success ratio
        self.stat = dict()

        # Observation will be npredator * vision * vision ndarray
        self.obs = self.get_obs()
        self.state = self.get_state()

        return self.obs, self.state, self.get_avail_actions()

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "steps_taken": self.total_steps / self.battles_game, #average steps taken
        }
        return stats
    
    def render(self):
        """ Not implemented """
        pass
        raise NotImplementedError

    def close(self):
        """ Not applicable to PP or PCP """
        pass
        raise NotImplementedError

    def seed(self, seed):
        self._seed = seed
        return

    def save_replay(self):
        """Not implemented"""
        pass
        raise NotImplementedError
        
    def _get_cordinates(self):
        idx = np.random.choice(np.prod(self.dims),(self.npredator + self.nprey), replace=False)
        return np.vstack(np.unravel_index(idx, self.dims)).T

    def _set_grid(self):
        self.grid = np.arange(self.BASE).reshape(self.dims)
        # Padding for vision
        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)
        self.empty_bool_base_grid = self._onehot_initialization(self.grid)

    def _take_action(self, idx, act):
        # prey action
        if idx >= self.npredator:
            # fixed prey
            if not self.moving_prey:
                return
            else:
                raise NotImplementedError

        if self.reached_prey[idx] == 1:
            return

        # STAY action
        if act==5:
            return

        # UP
        if act==0 and self.grid[max(0,
                                self.predator_loc[idx][0] + self.vision - 1),
                                self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = max(0, self.predator_loc[idx][0]-1)

        # RIGHT
        elif act==1 and self.grid[self.predator_loc[idx][0] + self.vision,
                                min(self.dims[1] -1,
                                    self.predator_loc[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = min(self.dims[1]-1,
                                            self.predator_loc[idx][1]+1)

        # DOWN
        elif act==2 and self.grid[min(self.dims[0]-1,
                                    self.predator_loc[idx][0] + self.vision + 1),
                                    self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = min(self.dims[0]-1,
                                            self.predator_loc[idx][0]+1)

        # LEFT
        elif act==3 and self.grid[self.predator_loc[idx][0] + self.vision,
                                    max(0,
                                    self.predator_loc[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = max(0, self.predator_loc[idx][1]-1)

    def _get_reward(self):
        n = self.npredator if not self.enemy_comm else self.npredator + self.nprey
        reward = np.full(n, self.TIMESTEP_PENALTY)

        on_prey_val = np.zeros((self.predator_loc.shape[0]), dtype=bool)

        for prey in self.prey_loc:
            on_prey_i = np.all(self.predator_loc==prey, axis=1)
            on_prey_val = np.any([on_prey_val, on_prey_i], axis=0)

        on_prey = np.where(on_prey_val)[0]
        nb_predator_on_prey = on_prey.size

        if self.mode == 'cooperative':
            reward[on_prey] = self.POS_PREY_REWARD * nb_predator_on_prey
        elif self.mode == 'competitive':
            if nb_predator_on_prey:
                reward[on_prey] = self.POS_PREY_REWARD / nb_predator_on_prey
        elif self.mode == 'mixed':
            reward[on_prey] = self.PREY_REWARD
        else:
            raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")

        self.reached_prey[on_prey] = 1

        if np.all(self.reached_prey == 1) and self.mode == 'mixed':
            self.battles_won += 1
            self.battles_game += 1
            self.episode_over = True
            self.battle_won = True
        # Prey reward
        if nb_predator_on_prey == 0:
            reward[self.npredator:] = -1 * self.TIMESTEP_PENALTY
        else:
            # TODO: discuss & finalise
            reward[self.npredator:] = 0

        # Success ratio
        if self.mode != 'competitive':
            if nb_predator_on_prey == self.npredator:
                self.stat['success'] = 1
            else:
                self.stat['success'] = 0
        
        '''hacky way to deal with reward'''
        # TODO: check if reward has to be an array
        # reward = np.sum(reward)/self.n_agents
        reward = np.sum(reward)
        # rewards = [[reward]]*self.n_agents
        # print(reward)
        # return rewards
        return reward

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())

    def _onehot_initialization(self, a):
        ncols = self.vocab_size
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)
    
    def close(self):
        return 