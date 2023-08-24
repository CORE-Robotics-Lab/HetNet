
import curses
import copy
import random
import math

import numpy as np
import gym
from gym import spaces

from .WildFire_Simulate_Original import WildFire


class FireCommanderEnv(gym.Env):
    """
    Simulate a Fire Commander environment.
    Each agent can just observe itself (its own identity) i.e. s_j = j and vision squares around it.

    Built on top of predator capture environment with updated fire movement and no prey sink.
    """
    def __init__(self, args):
        self.__version__ = "0.0.1"

        self.OUTSIDE_CLASS = 1
        self.FIRE_CLASS = 2
        self.PREDATOR_CLASS = 3
        self.PREDATOR_CAPTURE_CLASS = 4
        self.TIMESTEP_PENALTY = -0.1
        self.FALSE_WATER_DROP_PENALTY = -0.1
        self.CAPTURE_REWARD = 10
        self.FIRE_PENALTY = -0.1
        self.episode_over = False

        self.args = args

        self.dim = args.dim
        self.npredator = args.num_P
        self.npredator_capture = args.num_A

        self.n_agents = self.npredator + self.npredator_capture

        self.nfire_start = args.nfires
        self.nfire = args.nfires
        self.predator_capture_index = self.npredator
        self.captured_fire_index = self.npredator + self.npredator_capture

        self.dims = dims = (self.dim, self.dim)
        self.fire_spread_off = args.fire_spread_off
        self.mode = args.mode
        self.stay = not args.no_stay

        self.duration = args.episode_limit
        self.reward_type = args.reward_type
        self.max_wind_speed = args.max_wind_speed
        self.vision = args.vision
        self.tensor_obs = args.tensor_obs

        self.episode_limit = args.episode_limit

        self.eval_init_states = None
        self.episode_eval_counter = 0

        if self.reward_type == 0:
            self.TEMP_REWARD_TYPE = 'NEG_PER_FIRE'
        elif self.reward_type == 1:
            self.TEMP_REWARD_TYPE = 'NEG_TIMESTEP_BIG_POS_CAPTURE'
        elif self.reward_type == 2:
            self.TEMP_REWARD_TYPE = 'NEG_FALSE_DUMP_PENALTY'

        if self.max_wind_speed is None:
            self.max_wind_speed = self.dim / 5

        # Define what an agent can do -
        # (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
        if self.stay:
            self.naction = 6
        else:
            self.naction = 5

        self.BASE = (dims[0] * dims[1])
        self.OUTSIDE_CLASS += self.BASE
        self.FIRE_CLASS += self.BASE
        self.PREDATOR_CLASS += self.BASE
        self.PREDATOR_CAPTURE_CLASS = self.BASE

        # Setting max vocab size for 1-hot encoding
        self.vocab_size = 1 + 1 + self.BASE + 1 + 1
        #          predator + fire + grid + outside

        if self.tensor_obs:
            self.observation_space_tens = spaces.Box(-np.inf, np.inf, shape=[self.dim, self.dim, 4])
            if self.vision == 0:
                self.feature_map = [np.zeros((1, 1, 3)), np.zeros((self.dim, self.dim, 1))]
            elif self.vision == 1:
                self.feature_map = [np.zeros((3, 3, 3)), np.zeros((self.dim, self.dim, 1))]
            else:
                self.feature_map = [np.zeros((3, 5, 5)), np.zeros((self.dim, self.dim, 1))]

            self.true_feature_map = np.zeros((self.dim, self.dim, 3 + self.npredator + self.npredator_capture))

        else:
            # Actual observation will be of the shape 1 * npredator * (2v+1) * (2v+1) * vocab_size
            self.observation_space = spaces.Box(low=0, high=1,
                                                shape=(self.vocab_size, (2 * self.vision) + 1, (2 * self.vision) + 1),
                                                dtype=int)

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.n_agents):
            self.action_space.append(spaces.Discrete(self.naction))
            self.share_observation_space.append(self.get_state_size())

            if self.tensor_obs:
                # this is the shape of the obs array from the environment in tensor_obs; in the policy this (3,2) of arrays
                # needs to converted in states and obs in tensor format inside the policy
                self.observation_space.append([2])
            else:
                self.observation_space.append(self.get_obs_size())


    def init_curses(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)


    def step(self, action):
        """
        The agents take a step in the environment.

        Parameters
        ----------
        action : list/ndarray of length m, containing the indexes of what lever each 'm' chosen agents pulled.

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) : agent observations
            state (object): agent states
            reward (float) : Ratio of Number of discrete levers pulled to total number of levers.
            episode_over (bool) : Will be true as episode length is 1
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")

        self.false_water_drop = np.zeros(self.npredator_capture)
        self.fire_extinguished = np.zeros(self.npredator_capture)   # 0 -> no fire, 1 -> normal fire, 2 -> fire source

        action = np.array(action).squeeze()
        action = np.atleast_1d(action)

        infos = [{} for i in range(self.n_agents)]
        dones = np.zeros((self.n_agents), dtype=bool)
        bad_transition = False

        # Take agent actions
        for i, a in enumerate(action):
            self._take_action(i, a)

        self._episode_steps += 1
        self.total_steps += 1

        # Propagate fire
        if not self.fire_spread_off:
            self._fire_propagation()

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."
        self.episode_over = False

        if self.tensor_obs:
            self.obs = self._get_tensor_obs()
        else:
            self.obs = self.get_obs()

        self.state = self.get_state()

        debug = {
            'predator_locs': self.predator_loc,
            'predator_capture_locs': self.predator_capture_loc,
            'fire_locs': self.fire_loc,
        }

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

    def reset(self, eval_init_states=None):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        if eval_init_states is not None:
            self.eval_init_states = eval_init_states

        self._episode_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.total_steps = 0
        self.battle_won = False

        self.episode_over = False
        self.fire_out = False
        self.nfire = self.nfire_start   # reset fire count

        # Locations
        if self.eval_init_states is not None:
            locs = self.eval_init_states[self.episode_eval_counter]
            if self.episode_eval_counter < len(self.eval_init_states) - 1:
                self.episode_eval_counter += 1
        else:
            locs = self._get_coordinates()

        self.predator_loc = locs[:self.npredator]
        self.predator_capture_loc = locs[self.predator_capture_index:self.captured_fire_index]

        self.fire_loc = locs[self.captured_fire_index:]
        self.init_fire_loc = self.fire_loc

        self._fire_init()
        self._set_grid()

        # stat - like success ratio
        self.stat = dict()
        self.stat['enemy_count'] = []

        # Observation will be npredator * vision * vision ndarray
        if self.tensor_obs:
            self.obs = self._get_tensor_obs()
        else:
            self.obs = self.get_obs()
        self.state = self.get_state()

        return self.obs, self.state, self.get_avail_actions()

    def seed(self, seed):
        return

    def _fire_init(self):
        # Fire region (Color: Red (255, 0, 0))
        # The wildfire generation and propagation utilizes the FARSITE wildfire mathematical model
        # To clarify the fire state data, the state of the fire spot at each moment is stored in the dictionary list separately
        # Besides, the current fire map will also be stored as a matrix with the same size of the simulation model, which
        # reflects the fire intensity of each position on the world

        # Wildfire init parameters
        terrain_sizes = [*self.dims]
        hotspot_areas = []

        for fire in self.fire_loc:
            hotspot_areas.append([fire[0], fire[0], fire[1], fire[1]])

        avg_wind_speed = np.random.rand() * self.max_wind_speed
        avg_wind_direction = np.random.rand() * 2 * np.pi

        # Uniform fire model with default parameters
        self.fire_mdl = WildFire(terrain_sizes=terrain_sizes, hotspot_areas=hotspot_areas, num_ign_points=1,
                                 duration=self.duration, flame_angle=np.pi/3)
        self.ign_points_all = self.fire_mdl.hotspot_init()  # initialize hotspots (fire front fires)
        self.previous_terrain_map = self.ign_points_all.copy()  # initializing the starting terrain map
        self.geo_phys_info = self.fire_mdl.geo_phys_info_init(
            avg_wind_speed=avg_wind_speed, avg_wind_direction=avg_wind_direction)  # initialize geo-physical info
        self.pruned_list = []   # the pruned_List, store the pruned fire spots

    def _fire_propagation(self):
        """Propagate fire one step forward according to the fire model.
        Details:
            Moving w/o spreading
        """
        self.new_fire_front, current_geo_phys_info = self.fire_mdl.fire_propagation(
            self.dim, ign_points_all=self.ign_points_all, geo_phys_info=self.geo_phys_info,
            previous_terrain_map=self.previous_terrain_map, pruned_List=self.pruned_list)

        updated_terrain_map = self.previous_terrain_map

        # Add fire front to fire loc
        for fire in self.new_fire_front:
            fire_x, fire_y = int(fire[0]), int(fire[1])

            if fire_x <= self.dim - 1 and fire_y <= self.dim - 1 and fire_x >= 0 and fire_y >= 0:
                self.fire_loc = np.concatenate((self.fire_loc, [[fire_x, fire_y]]))

        self.fire_loc = np.unique(self.fire_loc, axis=0)
        self.nfire = self.fire_loc.shape[0]

        # Update propagation info
        if self.new_fire_front.shape[0] > 0:
            self.previous_terrain_map = np.concatenate((updated_terrain_map, self.new_fire_front))  # fire map with fire decay
            self.ign_points_all = self.new_fire_front

        # This version ties fire_loc to terrain_map
        # self.fire_loc = np.unique(self.previous_terrain_map[:,:2].astype(np.int), axis=0)
        # self.nfire = self.fire_loc.shape[0]

    def _get_coordinates(self):
        idx = np.random.choice(np.prod(self.dims),(self.npredator + self.npredator_capture + self.nfire), replace=False)
        return np.vstack(np.unravel_index(idx, self.dims)).T

    def _set_grid(self):
        self.grid = np.arange(self.BASE).reshape(self.dims)
        # Padding for vision
        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values=self.OUTSIDE_CLASS)
        self.empty_bool_base_grid = self._onehot_initialization(self.grid)

    def _get_obs(self):
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        for i, p in enumerate(self.predator_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS] += 1

        for i, p in enumerate(self.predator_capture_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CAPTURE_CLASS] += 1

        for i, p in enumerate(self.fire_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.FIRE_CLASS] += 1

        obs = []
        for p in self.predator_loc:
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x])

        # 29 are concatenation of one-hot vectors including the information for the fire and predator, and also the position information of the grid
        # 10:14
        # There is also one indicating if this grid is a result of padding. There can be paddings at the edge of the grids when vision >= 1
        for p in self.predator_capture_loc:
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x])
            obs[-1][0][0][self.BASE:] = [0] * 4   # TODO: why this line here is added onto code from PCP

        obs = np.stack(obs)
        return obs


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

        for i in ['predator', 'predator_capture', 'fire', 'self']:
            if i == 'predator':
                for p in self.predator_loc:
                    true_feature_map[p[0], p[1], 0] += 1
            elif i== 'predator_capture':
                for p in self.predator_capture_loc:
                    true_feature_map[p[0], p[1], 1] += 1
            elif i == 'fire':
                for p in self.fire_loc:
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

        # 29 are concatenation of one-hot vectors including the information for the prey and predator, and also the position information of the grid
        # 10:14
        # There is also one indicating if this grid is a result of padding. There can be paddings at the edge of the grids when vision >= 1
        for p in self.predator_capture_loc:
            sub_obs = self.feature_map.copy()
            sub_obs[1] = true_feature_map[:, :, 3 + agent_counter]
            agent_counter += 1
            obs.append(sub_obs)

        # if self.enemy_comm:
        #     for p in self.prey_loc:
        #         slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
        #         slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
        #         obs.append(self.bool_base_grid[slice_y, slice_x])
        obs = np.stack(obs)

        return obs

    def fire_locs(self):
        env_grid = np.zeros((self.dim,)*2)
        env_grid[tuple(self.fire_loc.T)] = 1
        return env_grid.flatten()


    def get_state(self):
        """
        Returns the global state of the environment composed of agent coordinates, fire locations and episode step
        number.
        """
        # this is the coordinates in the world of each predator, capture and prey agents;
        state = np.vstack((self.predator_loc, self.predator_capture_loc))
        state = np.reshape(state, (-1)) # flatten

        # add environment states (fire locations and episode step number)
        state = np.append(state, self.fire_locs())
        state = np.append(state, self._episode_steps/self.episode_limit)

        # repeat global state each agent
        self.state = []
        for i in range(self.n_agents):
            self.state.append(state)

        return self.state

    def get_state_size(self):
        """
        Returns the shape of the state
        """
        state_size = (self.n_agents)*2 + self.dim**2 + 1

        return [state_size]

    def get_avail_actions(self):
        """ All actions are available"""
        avail_actions = []
        for i in range(self.n_agents):
            avail_actions.append([1]*self.naction)
        return avail_actions

    def _take_action(self, idx, act):
        if act==4: # STAY action
            return

        # Predator Capture action
        if idx >= self.predator_capture_index:
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
                # put out fire
                pred_cap_loc = self.predator_capture_loc[idx - self.npredator]
                fire_loc_idx = np.argwhere(np.all(self.fire_loc == pred_cap_loc, axis=1))
                ign_point_idx = np.argwhere(np.all(self.ign_points_all[:,:2].astype(np.int) == pred_cap_loc, axis=1))

                if len(fire_loc_idx) + len(ign_point_idx) == 0:
                    self.false_water_drop[idx - self.npredator] = 1
                else:
                    if len(fire_loc_idx) > 0:
                        self.pruned_list.append([int(pred_cap_loc[0]), int(pred_cap_loc[1])])
                        self.fire_extinguished[idx - self.npredator] = 1

                    if len(ign_point_idx) > 0:
                        self.fire_extinguished[idx - self.npredator] = 2

                self.fire_loc = np.delete(self.fire_loc, fire_loc_idx, axis=0)
                self.ign_points_all = np.delete(self.ign_points_all, ign_point_idx, axis=0)
                self.nfire = self.fire_loc.shape[0]

                if self.nfire == 0:
                    self.fire_out = True

        else:
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

            elif act==5:
                return

    def _get_reward(self):
        """ Calculate reward for each agent in environment based on mode.
        """
        if self.mode == 'cooperative':
            raise NotImplementedError('>>> Reward not implemented for cooperative Fire Commander')
        elif self.mode == 'competitive':
            raise NotImplementedError('>>> Reward not implemented for competitive Fire Commander')
        elif self.mode == 'mixed':
            if self.TEMP_REWARD_TYPE == 'NEG_PER_FIRE':
                reward_val = self.nfire * self.FIRE_PENALTY
                reward = np.full(self.captured_fire_index, reward_val)
            elif self.TEMP_REWARD_TYPE == 'NEG_TIMESTEP_BIG_POS_CAPTURE':
                best_extinguished_fire = np.max(self.fire_extinguished)

                if best_extinguished_fire == 1:
                    reward = np.full(self.captured_fire_index, 0)
                elif best_extinguished_fire == 2:
                    reward = np.full(self.captured_fire_index, self.CAPTURE_REWARD)
                else:
                    reward = np.full(self.captured_fire_index, self.TIMESTEP_PENALTY)
            elif self.TEMP_REWARD_TYPE == 'NEG_FALSE_DUMP_PENALTY':
                reward_val = self.nfire * self.FIRE_PENALTY
                reward = np.full(self.captured_fire_index, reward_val)

                false_drop_penalty = self.false_water_drop * self.FALSE_WATER_DROP_PENALTY
                reward[self.npredator:self.npredator + self.npredator_capture] += false_drop_penalty
        else:
            raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")

        if self.mode == 'mixed' and self.fire_out:
            self.episode_over = True
            self.battles_won += 1
            self.battles_game += 1
            self.battle_won = True

        # Success ratio
        if self.mode != 'competitive':
            self.stat['success'] = self.fire_out

        self.stat['enemy_count'].append(self.nfire)

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

    def render(self, mode='human', close=False):
        grid = np.zeros(self.BASE, dtype=object).reshape(self.dims)
        self.stdscr.clear()

        for p in self.fire_loc:
            render_char = 'F' if np.all(self.ign_points_all[:,:2].astype(np.int) == p, axis=1) else 'f'

            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + render_char
            else:
                grid[p[0]][p[1]] = render_char
            # grid[p[0]][p[1]] = render_char

        for p in self.predator_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'P'
            else:
                grid[p[0]][p[1]] = 'P'
            # grid[p[0]][p[1]] = 'P'

        for p in self.predator_capture_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'A'
            else:
                grid[p[0]][p[1]] = 'A'
            # grid[p[0]][p[1]] = 'A'

        screen = ''

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                screen += str(item)
                if item != 0:
                    if 'X' in item and 'P' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(3))
                    elif 'X' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2))
                else:
                    self.stdscr.addstr(row_num, idx * 4, '0'.center(3), curses.color_pair(4))
            # screen += '\n'

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()
        # print(screen)

    def exit_render(self):
        curses.endwin()
