#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a Fire Commander environment.
Each agent can just observe itself (it's own identity) i.e. s_j = j and vision squares around it.

Built on top of predator capture environment with updated fire movement and no prey sink.

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
         -1 out of bound,
         indexing for predator agent (from 2?)
         ??? for fire agent (1 for fixed case, for now)
    - Action Space & Observation Space are according to an agent
    - Rewards -0.05 at each time step till the time
    - Episode never ends
    - Obs. State: Vocab of 1-hot < predator, fires & units >
"""
import math
# core modules
import random

# 3rd party modules
import gym
import numpy as np
from gym import spaces
from WildFire_Simulate_Original import WildFire

# import curses


class FireCommanderEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        self.OUTSIDE_CLASS = 1
        self.FIRE_CLASS = 2
        self.PREDATOR_CLASS = 3
        self.PREDATOR_CAPTURE_CLASS = 4
        self.TIMESTEP_PENALTY = -0.1
        self.FALSE_WATER_DROP_PENALTY = -0.1
        self.CAPTURE_REWARD = 10
        self.NONSOURCE_CAPTURE_REWARD = 0
        self.FIRE_PENALTY = -0.1
        self.DISCOVER_SOURCE_REWARD = 0
        self.DISCOVER_NONSOURCE_REWARD = 0
        self.episode_over = False
        self.action_blind = True

    def init_curses(self):
        pass
    #     self.stdscr = curses.initscr()
    #     curses.start_color()
    #     curses.use_default_colors()
    #     curses.init_pair(1, curses.COLOR_RED, -1)
    #     curses.init_pair(2, curses.COLOR_YELLOW, -1)
    #     curses.init_pair(3, curses.COLOR_CYAN, -1)
    #     curses.init_pair(4, curses.COLOR_GREEN, -1)

    def init_args(self, parser):
        env = parser.add_argument_group('Fire Commander task')
        env.add_argument('--nfires', type=int, default=1,
                         help="Original number of fires")
        env.add_argument('--dim', type=int, default=5,
                         help="Dimension of box")
        env.add_argument('--vision', type=int, default=2,
                         help="Vision of predator")
        env.add_argument('--fire_spread_off', action="store_true", default=False,
                         help="Turn off fire spread")
        env.add_argument('--no_stay', action="store_true", default=False,
                         help="Whether predators have an action to stay in place")
        parser.add_argument('--mode', default='mixed', type=str,
                        help='cooperative|competitive|mixed (default: mixed)')
        env.add_argument('--nfriendly_P', type=int, default=2,
                            help="Total number of friendly perception agents in play")
        env.add_argument('--nfriendly_A', type=int, default=1,
                            help="Total number of friendly action agents in play")
        env.add_argument('--max_wind_speed', type=float, default=None,
                            help="Maximum speed of wind (default: grid size / 5)")
        env.add_argument('--tensor_obs', action="store_true", default=False,
                         help="Do you want a tensor observation (not implemente for FC)")
        env.add_argument('--reward_type', type=int, default=3,
                         help="Reward type to use (0 -> negative timestep, 1 -> positive capture, 2 -> water dump penalty, 3 -> combined)")
        env.add_argument('--A_vision', type=int, default=-1,
                            help="Vision of A agents. If -1, defaults to blind")

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        params = ['dim', 'vision', 'fire_spread_off', 'mode', 'nfriendly_P', 'nfriendly_A', 'max_wind_speed', 'tensor_obs', 'reward_type', 'A_vision']

        for key in params:
            setattr(self, key, getattr(args, key))
        self.init_curses()
        self.npredator = args.nfriendly_P
        self.npredator_capture = args.nfriendly_A
        self.nfire_start = args.nfires
        self.nfire = args.nfires
        self.predator_capture_index = self.npredator
        self.captured_fire_index = self.npredator + self.npredator_capture
        self.dims = dims = (self.dim, self.dim)
        self.stay = not args.no_stay
        self.duration = args.max_steps

        if self.reward_type == 0:
            self.TEMP_REWARD_TYPE = 'NEG_PER_FIRE'
        elif self.reward_type == 1:
            self.TEMP_REWARD_TYPE = 'NEG_TIMESTEP_BIG_POS_CAPTURE'
        elif self.reward_type == 2:
            self.TEMP_REWARD_TYPE = 'NEG_FALSE_DUMP_PENALTY'
        elif self.reward_type == 3:
            self.TEMP_REWARD_TYPE = 'COMBINED'

        if self.max_wind_speed is None:
            self.max_wind_speed = self.dim / 5

        args.nagents = self.npredator + self.npredator_capture

        # (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
        # Define what an agent can do -
        self.naction = 6 if self.stay else 5

        self.action_space = spaces.MultiDiscrete([self.naction])

        self.BASE = (dims[0] * dims[1])
        self.OUTSIDE_CLASS += self.BASE
        self.FIRE_CLASS += self.BASE
        self.PREDATOR_CLASS += self.BASE
        self.PREDATOR_CAPTURE_CLASS = self.BASE

        # Setting max vocab size for 1-hot encoding
        self.vocab_size = 1 + 1 + self.BASE + 1 + 1
        #          predator + fire + grid + outside

        # Observation for each agent will be vision * vision ndarray
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.vocab_size, (2 * self.vision) + 1, (2 * self.vision) + 1), dtype=int)
        # Actual observation will be of the shape 1 * npredator * (2v+1) * (2v+1) * vocab_size

        return

    def step(self, action):
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

        self.false_water_drop = np.zeros(self.npredator_capture)    # if the A agents dropped water not on a fire
        self.fire_extinguished = np.zeros(self.npredator_capture)   # if the A agent dropped water on a fire (0 -> no fire, 1 -> normal fire, 2 -> fire source)
        self.extinguishing = np.zeros(self.npredator_capture)       # if the A agent took the extinguish action
        self.just_discovered_source = np.zeros(self.npredator)      # if the P agent discovered a source fire
        self.just_discovered_nonsource = np.zeros(self.npredator)   # if the P agent discovered a nonsource fire

        action = np.array(action).squeeze()
        action = np.atleast_1d(action)

        # Take agent actions
        for i, a in enumerate(action):
            self._take_action(i, a)

        # Propagate fire
        if not self.fire_spread_off:
            self._fire_propagation()

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."

        self.episode_over = False
        self.obs = self._get_obs()

        debug = {
            'predator_locs' : self.predator_loc,
            'predator_capture_locs' : self.predator_capture_loc,
            'fire_locs' : self.fire_loc
        }

        return self.obs, self._get_reward(), self.episode_over, debug

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.fire_out = False
        self.nfire = self.nfire_start   # reset fire count
        self.extinguishing = np.zeros(self.npredator_capture)
        self.discovered_fire = []
        self.just_discovered_source = np.zeros(self.npredator)
        self.just_discovered_nonsource = np.zeros(self.npredator)

        # Locations
        locs = self._get_coordinates()
        self.predator_loc = locs[:self.npredator]
        self.predator_capture_loc = locs[self.predator_capture_index:self.captured_fire_index]
        self.fire_loc = locs[self.captured_fire_index:]

        self._fire_init()
        self._set_grid()

        # stat - like success ratio
        self.stat = dict()
        self.stat['enemy_count'] = []

        # Observation will be npredator * vision * vision ndarray
        self.obs = self._get_obs()

        return self.obs

    def seed(self):
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
        '''Propagate fire one step forward according to the fire model.

        Details:
            Moving w/o spreading
        '''
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

        for i, p in enumerate(self.predator_loc):
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x])

            for fire in self.fire_loc:
                if fire in np.array(self.discovered_fire):
                    continue

                if fire[0] >= p[0] - self.vision and fire[0] <= p[0] + self.vision and fire[1] >= p[1] - self.vision and fire[1] <= p[1] + self.vision:
                    self.discovered_fire.append(fire)

                    if fire in self.ign_points_all[:,:2].astype(np.int):
                        self.just_discovered_source[i] = 1
                    else:
                        self.just_discovered_nonsource[i] = 1


        # There is also one indicating if this grid is a result of padding. There can be paddings at the edge of the grids when vision >= 1
        for p in self.predator_capture_loc:
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x])

            if self.action_blind:
                import copy
                c = copy.deepcopy(obs)
                obs[-1][:, :, self.BASE:] = np.zeros(shape=obs[-1][:, :, self.BASE:].shape)

        print((c[0] == obs[0]).all())
        if (c[0] == obs[0]).all() == False:
            print('bad')
        print((c[1] == obs[1]).all())
        if (c[1] == obs[1]).all() == False:
            print('bad')
        print((c[2] == obs[2]).all())

        obs = np.stack(obs)

        return obs

    def _take_action(self, idx, act):
        # STAY action
        if act==4:
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
                self.extinguishing[idx - self.npredator] = 1

                if len(fire_loc_idx) + len(ign_point_idx) == 0:
                    self.false_water_drop[idx - self.npredator] = 1
                else:
                    if len(self.discovered_fire) != 0:
                        dis_fire_idx = np.argwhere(np.all(self.discovered_fire == pred_cap_loc, axis=1))

                        if len(dis_fire_idx) != 0:
                            del self.discovered_fire[dis_fire_idx[0][0]]

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
                pass


    def _get_reward(self):
        '''Calculate reward for each agent in environment based on mode.

        Modes:
            Cooperative:
                Not Implemented
            Competitive:
                Not Implemented
            Mixed:
                TODO: Explain working reward
        '''
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
            elif self.TEMP_REWARD_TYPE == 'COMBINED':
                reward = np.full(self.captured_fire_index, self.FIRE_PENALTY * self.nfire)
                best_extinguished_fire = np.max(self.fire_extinguished)

                # Set positive P reward
                reward[:self.npredator] += self.just_discovered_source * self.DISCOVER_SOURCE_REWARD
                reward[:self.npredator] += self.just_discovered_nonsource * self.DISCOVER_NONSOURCE_REWARD

                # Set position A reward
                agents_extinguish_non_source_fire = np.array(np.where(self.fire_extinguished == 1)) + self.npredator
                agents_extinguish_source_fire = np.array(np.where(self.fire_extinguished == 2)) + self.npredator

                reward[agents_extinguish_non_source_fire] = self.NONSOURCE_CAPTURE_REWARD
                reward[agents_extinguish_source_fire] = self.CAPTURE_REWARD

                reward[self.npredator:self.npredator + self.npredator_capture] += self.false_water_drop * self.FALSE_WATER_DROP_PENALTY
        else:
            raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")

        if self.mode == 'mixed' and self.fire_out:
            self.episode_over = True

        # Success ratio
        if self.mode != 'competitive':
            self.stat['success'] = self.fire_out

        self.stat['enemy_count'].append(self.nfire)

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

        for p in self.predator_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'P'
            else:
                grid[p[0]][p[1]] = 'P'

        for i, p in enumerate(self.predator_capture_loc):
            char_rep = 'C' if self.extinguishing[i] else 'A'

            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + char_rep
            else:
                grid[p[0]][p[1]] = char_rep

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

