
import copy
import numpy as np

from gym import spaces
from gym.spaces import Discrete

from .multiagentenv import MultiAgentEnv


class PredatorCapture(MultiAgentEnv):
    def __init__(self, args, n_enemies=1, vision=0, A_vision=-1, moving_prey=False, no_stay=False, mode="mixed",
                 enemy_comm=False, tensor_obs=False, second_reward_scheme=False,):
        self.__version__ = "0.0.1"

        # default params
        self.OUTSIDE_CLASS = 1
        self.PREY_CLASS = 2
        self.PREDATOR_CLASS = 3
        self.PREDATOR_CAPTURE_CLASS = 0
        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0
        self.POS_PREY_REWARD = 0.05
        self.ON_PREY_BUT_NOT_CAPTURE_REWARD = -0.025
        self.episode_over = False
        self.action_blind = True
        self.episode_eval_counter = 0
        self.eval_init_states = None

        # param arguments
        self.args = args
        self.nfriendly_P = args.num_P
        self.nfriendly_A = args.num_A

        self.vision = self.args.vision

        self.A_vision = A_vision
        self.nprey = n_enemies
        self.npredator = self.nfriendly_P
        self.npredator_capture = self.nfriendly_A
        self.predator_capture_index = self.npredator
        self.captured_prey_index = self.npredator + self.npredator_capture
        dim = args.dim
        self.dims = (dim,dim)
        self.dim = dim
        self.mode = mode
        self.enemy_comm = enemy_comm
        self.stay = not no_stay
        self.tensor_obs = self.args.tensor_obs
        self.second_reward_scheme = second_reward_scheme
        if self.A_vision != -1:
            # if args.A_vision != 0:
            #     raise NotImplementedError
            self.A_agents_have_vision = True
            self.A_vision = self.A_vision
            self.action_blind = False
        else:
            self.A_agents_have_vision = False
        self.episode_limit = args.episode_limit
        self.n_agents = self.nfriendly_P+self.nfriendly_A
        self.nagents = self.nfriendly_P+self.nfriendly_A
        self._episode_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.total_steps = 0
        if moving_prey:
            raise NotImplementedError
        # Define what an agent can do -
        if self.stay:
            self.naction = 6
        else:
            self.naction = 5
        self.BASE = (self.dims[0] * self.dims[1])
        self.OUTSIDE_CLASS += self.BASE
        self.PREY_CLASS += self.BASE
        self.PREDATOR_CLASS += self.BASE
        self.PREDATOR_CAPTURE_CLASS += self.BASE
        self.state_len = (self.nfriendly_P + self.nfriendly_A + 1) # Change this if you change state_len in main_copy.py

        # 4 is 4 types of classes!
        self.num_classes = 4
        self.vocab_size = self.BASE + self.num_classes

        if self.tensor_obs:
            self.observation_space_tens = spaces.Box(-np.inf, np.inf, shape=[self.dim, self.dim, 4])
            if self.vision == 0:
                self.feature_map = [np.zeros((1, 1, 3)), np.zeros((self.dim, self.dim, 1))]
            elif self.vision == 1:
                self.feature_map = [np.zeros((3, 3, 3)), np.zeros((self.dim, self.dim, 1))]
            else:
                self.feature_map = [np.zeros((3, 5, 5)), np.zeros((self.dim, self.dim, 1))]

            self.true_feature_map = np.zeros((self.dim, self.dim, 3 + self.nfriendly_P + self.nfriendly_A))

        else:
            self.observation_space = spaces.Box(low=0, high=1,
                                                shape=(self.vocab_size, (2 * self.vision) + 1, (2 * self.vision) + 1),
                                                dtype=int)
            # Actual observation will be of the shape 1 * npredator * (2v+1) * (2v+1) * vocab_size

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.n_agents):
            self.action_space.append(Discrete(self.naction))
            self.share_observation_space.append(self.get_state_size())

            if self.tensor_obs:
                # this is the shape of the obs array from the environment in tensor_obs; in the policy this (3,2) of arrays
                # needs to converted in states and obs in tensor format inside the policy
                self.observation_space.append([2])
            else:
                self.observation_space.append(self.get_obs_size())


    def step(self, actions):
        """ Returns reward, terminated, info """
        """
        The agents take a step in the environment.

        Parameters
        ----------
        action : list/ndarray of length m, containing the indexes of what lever each 'm' chosen agents pulled.

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :

            reward (float) : Ratio of Number of discrete levers pulled to total number of levers.
            episode_over (bool) : Will be true as episode length is 1
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")

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

        if self.tensor_obs:
            self.obs = self._get_tensor_obs()
        else:
            self.obs = self.get_obs()

        self.state = self.get_state()

        # debug = {'predator_locs':self.predator_loc.reshape(-1),'prey_locs':self.prey_loc.reshape(-1)}
        debug = {"battle_won": False}
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
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        for i, p in enumerate(self.predator_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS] += 1

        for i, p in enumerate(self.predator_capture_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CAPTURE_CLASS] += 1

        for i, p in enumerate(self.prey_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREY_CLASS] += 1

        obs = []
        for p in self.predator_loc:
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x].reshape((-1)))

        for p in self.predator_capture_loc:
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x].reshape((-1)))

            if self.action_blind:
                # [-1] is because we just appended it above
                obs[-1][self.BASE:] = np.full(shape=obs[-1][self.BASE:].shape, fill_value=-1)
            elif self.A_agents_have_vision:
                slice_A = slice(self.vision - self.A_vision, self.vision + self.A_vision + 1)

                A_vision_obs = obs[-1][slice_A, slice_A].copy()
                obs[-1] = np.full(shape=obs[-1].shape, fill_value=-1)
                # translates all other observations and states into -1s except the one that is "active"
                obs[-1][slice_A, slice_A] = A_vision_obs

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
        obs, _, _ = self.reset()
        obs_size = [obs[0].shape[0]]

        return obs_size

    def get_state(self):
        # this is the coordinates in the world of each predator, capture and prey agents;
        state = np.vstack((self.predator_loc, self.predator_capture_loc, self.prey_loc))
        state = np.reshape(state, (-1)) # flatten
        state = np.append(state, self._episode_steps/self.episode_limit)
        self.state = []
        for i in range(self.n_agents):
            self.state.append(state)
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        state_size = (self.nagents + self.nprey)*2 + 1
        return [state_size]

    def get_avail_actions(self):
        """ all actions are available """
        avail_actions = []
        for i in range(self.npredator+self.npredator_capture):
            avail_actions.append([1]*self.naction)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.naction

    def reset(self, eval_init_states=None):
        """ Returns initial observations and states"""
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """

        if eval_init_states is not None:
            self.eval_init_states = eval_init_states

        self.episode_over = False
        self.reached_prey = np.zeros(self.npredator+self.npredator_capture)
        self.captured_prey = np.zeros(self.npredator_capture)
        self.npredator_capture = self.nfriendly_A
        self._episode_steps = 0
        self.battle_won = False

        if self.eval_init_states is not None:
            locs = self.eval_init_states[self.episode_eval_counter]
            if self.episode_eval_counter < len(self.eval_init_states) - 1:
                self.episode_eval_counter += 1
        else:
            locs = self._get_cordinates()

        self.predator_loc = locs[:self.npredator]
        self.predator_capture_loc = locs[self.predator_capture_index:self.captured_prey_index]
        self.prey_loc = locs[self.captured_prey_index:]

        self._set_grid()

        # stat - like success ratio
        self.stat = dict()

        if self.tensor_obs:
            self.obs = self._get_tensor_obs()
        else:
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
            # "restarts": self.force_restarts,
        }
        return stats
    
    def render(self):
        raise NotImplementedError

    def close(self):
        """ Not implemented """
        pass

    def seed(self, seed):
        self._seed = seed
        return

    def save_replay(self):
        pass
        raise NotImplementedError
        
    def _get_cordinates(self):
        idx = np.random.choice(np.prod(self.dims), (self.npredator + self.npredator_capture + self.nprey),
                               replace=False)
        return np.vstack(np.unravel_index(idx, self.dims)).T

    def _set_grid(self):
        self.grid = np.arange(self.BASE).reshape(self.dims)
        # Mark agents in grid
        # self.grid[self.predator_loc[:,0], self.predator_loc[:,1]] = self.predator_ids
        # self.grid[self.prey_loc[:,0], self.prey_loc[:,1]] = self.prey_ids

        # Padding for vision
        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values=self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.grid)

    def _get_tensor_obs(self):
        """
        Notes: axis is initialized from the top corner. For example, (1,0) (one down, 0 right) refers to
        array([[0., 0., 0., 0., 0.],
               [1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])
        """
        true_feature_map = self.true_feature_map.copy()

        for i in ['predator', 'predator_capture', 'prey', 'self']:
            if i == 'predator':
                for p in self.predator_loc:
                    true_feature_map[p[0], p[1], 0] += 1
            elif i== 'predator_capture':
                for p in self.predator_capture_loc:
                    true_feature_map[p[0], p[1], 1] += 1
            elif i == 'prey':
                for p in self.prey_loc:
                    true_feature_map[p[0], p[1], 2] += 1
            elif i == 'self':
                all_predator_locs = np.vstack([self.predator_loc, self.predator_capture_loc])
                for k, p in enumerate(all_predator_locs):
                    true_feature_map[p[0], p[1], 3 + k] = 1

        obs = []
        agent_counter = 0
        for p in self.predator_loc:
            sub_obs = copy.deepcopy(self.feature_map)

            if self.vision == 0:
                y_start = p[0] - self.vision
                y_end = p[0] + self.vision
                x_start = p[1] - self.vision
                x_end = p[1] + self.vision

                sub_obs[0][:,:,0] = true_feature_map[y_start, x_start, 0]
                sub_obs[0][:,:,1] = true_feature_map[y_start, x_start, 1]
                sub_obs[0][:,:,2] = true_feature_map[y_start, x_start, 2]
                sub_obs[1] = true_feature_map[:, :, 3 + agent_counter]
            elif self.vision == 1:
                y_start = p[0] + 1 - self.vision
                y_end = p[0] + self.vision + 1 + 1
                x_start = p[1] + 1 - self.vision
                x_end = p[1] + 1 + self.vision + 1

                padded_feature_map_0 = np.pad(true_feature_map[:,:,0], 1, constant_values=-1)
                padded_feature_map_1 = np.pad(true_feature_map[:,:,1], 1, constant_values=-1)
                padded_feature_map_2 = np.pad(true_feature_map[:,:,2], 1, constant_values=-1)

                sub_obs[0][0] = padded_feature_map_0[y_start:y_end, x_start:x_end]
                sub_obs[0][1] = padded_feature_map_1[y_start:y_end, x_start:x_end]
                sub_obs[0][2] = padded_feature_map_2[y_start:y_end, x_start:x_end]
                sub_obs[1] = true_feature_map[:,:, 3 + agent_counter]
            else:
                # raise ValueError
                # Adds vision=2 support
                y_start = p[0] + 2 - self.vision
                y_end = p[0] + self.vision + 1 + 1 + 1
                x_start = p[1] + 2 - self.vision
                x_end = p[1] + 1 + self.vision + 1 + 1

                padded_feature_map_0 = np.pad(true_feature_map[:, :, 0], 2, constant_values=-1)
                padded_feature_map_1 = np.pad(true_feature_map[:, :, 1], 2, constant_values=-1)
                padded_feature_map_2 = np.pad(true_feature_map[:, :, 2], 2, constant_values=-1)

                sub_obs[0][0] = padded_feature_map_0[y_start:y_end, x_start:x_end]
                sub_obs[0][1] = padded_feature_map_1[y_start:y_end, x_start:x_end]
                sub_obs[0][2] = padded_feature_map_2[y_start:y_end, x_start:x_end]
                sub_obs[1] = true_feature_map[:, :, 3 + agent_counter]

            agent_counter += 1
            obs.append(sub_obs)

        for p in self.predator_capture_loc:
            sub_obs = self.feature_map.copy()
            sub_obs[1] = true_feature_map[:, :, 3 + agent_counter]
            agent_counter += 1
            obs.append(sub_obs)

        obs = np.stack(obs)

        return obs

    def _take_action(self, idx, act):
        # if idx == 2:
        #     print(act, self.predator_capture_loc, self.prey_loc)
        # prey action
        if idx >= self.npredator + self.npredator_capture:
            # fixed prey
            if not self.moving_prey:
                return
            else:
                raise NotImplementedError

        if not self.stay:
            raise NotImplementedError

        # STAY action
        if act==4:
            return

        # Predator Capture action
        if idx >= self.predator_capture_index:
            # TODO: fix sink
            if self.reached_prey[idx] == 1 and self.captured_prey[idx - self.npredator]:
                return
            if self.reached_prey[idx] == 1 and act in [0,1,2,3,4]:
                return

            # UP
            if act == 0 and self.grid[max(0,
                                          self.predator_capture_loc[idx - self.npredator][0] + self.vision - 1),
                                      self.predator_capture_loc[idx - self.npredator][1] + self.vision] != self.OUTSIDE_CLASS:
                self.predator_capture_loc[idx - self.npredator][0] = max(0, self.predator_capture_loc[idx - self.npredator][0] - 1)

            # RIGHT
            elif act == 1 and self.grid[self.predator_capture_loc[idx - self.npredator][0] + self.vision,
                                        min(self.dims[1] - 1,
                                            self.predator_capture_loc[idx - self.npredator][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
                self.predator_capture_loc[idx - self.npredator][1] = min(self.dims[1] - 1,
                                                self.predator_capture_loc[idx - self.npredator][1] + 1)

            # DOWN
            elif act == 2 and self.grid[min(self.dims[0] - 1,
                                            self.predator_capture_loc[idx - self.npredator][0] + self.vision + 1),
                                        self.predator_capture_loc[idx - self.npredator][1] + self.vision] != self.OUTSIDE_CLASS:
                
                self.predator_capture_loc[idx - self.npredator][0] = min(self.dims[0] - 1,
                                                self.predator_capture_loc[idx - self.npredator][0] + 1)

            # LEFT
            elif act == 3 and self.grid[self.predator_capture_loc[idx - self.npredator][0] + self.vision,
                                        max(0,
                                            self.predator_capture_loc[idx - self.npredator][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
                self.predator_capture_loc[idx - self.npredator][1] = max(0, self.predator_capture_loc[idx - self.npredator][1] - 1)

            elif act == 5:
                if (self.predator_capture_loc[idx - self.npredator] == self.prey_loc).all():
                    self.captured_prey[idx - self.npredator] = True

        else:
            if self.reached_prey[idx] == 1:
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

            # Predator agents cannot take capture.
            elif act==5:
                return

    def _get_reward(self):
        n = self.npredator + self.npredator_capture if not self.enemy_comm else self.npredator + self.npredator_capture + self.nprey
        reward = np.full(n, self.TIMESTEP_PENALTY)

        # find if predator is on prey
        # TODO: May be able to np.where pred_loc and pred_cap_loc seperately
        all_predator_locs = np.vstack([self.predator_loc, self.predator_capture_loc])
        on_prey_val = np.zeros((all_predator_locs.shape[0]), dtype=bool)

        for prey in self.prey_loc:
            on_prey_i = np.all(all_predator_locs==prey, axis=1)
            on_prey_val = np.any([on_prey_val, on_prey_i], axis=0)

        on_prey = np.where(on_prey_val)[0]      # indices of predators on prey
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

        # Take care of capture reward seperately
        if not self.second_reward_scheme:
            which_action_agents_have_not_captured_prey = np.where(self.captured_prey == 0)
            proper_action_agent_indexes = np.array(which_action_agents_have_not_captured_prey) + self.nfriendly_P
            reward[proper_action_agent_indexes] = self.TIMESTEP_PENALTY # self.TIMESTEP_PENALTY
        else:
            which_action_agents_have_not_captured_prey_but_are_on_prey = np.intersect1d(np.where(self.captured_prey == 0) , np.where(self.reached_prey[self.nfriendly_P:] == 1))

            proper_action_agent_indexes = np.array(which_action_agents_have_not_captured_prey_but_are_on_prey) + self.nfriendly_P
            reward[proper_action_agent_indexes] = self.ON_PREY_BUT_NOT_CAPTURE_REWARD  # self.TIMESTEP_PENALTY
            # if len(np.intersect1d(np.where(self.captured_prey == 0) , np.where(self.reached_prey[self.nfriendly_P:] == 1))) > 0:
            #     print(reward)


        if np.all(self.reached_prey == 1) and np.all(self.captured_prey == 1) and self.mode == 'mixed':
            self.battles_won += 1
            self.battles_game += 1
            self.episode_over = True
            self.battle_won = True

        # Success ratio
        if self.mode != 'competitive':
            if nb_predator_on_prey == self.npredator+self.npredator_capture and self.episode_over:
                self.stat['success'] = 1
            else:
                self.stat['success'] = 0

        reward = np.sum(reward)
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
