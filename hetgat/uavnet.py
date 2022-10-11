# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:00:23 2020

@author: pheno

Heterogeneous Graph Neural Network for UAV coordination
"""

import torch
import torch.nn as nn
import math
import time
from hetgat.graph.fastreal import MultiHeteroGATLayerReal, MultiHeteroGATLayerLossyReal
from hetgat.graph.fastbinary import MultiHeteroGATLayerBinary, MultiHeteroGATLayerLossyBinary
# from hetgat.utils import build_hetgraph
import torch.nn as nn


# GNN with state node - non-batched version
# Used in FireCommander_Easy

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


class UAVNetA2CEasy(nn.Module):

    # in_dim_raw is the dict of raw inputs
    # in_dim_raw['Status']: UAV status vector, same for P and A
    # in_dim_raw['Sensor']: (H, W) height and width of sensor image
    def __init__(self, in_dim_raw, in_dim, hid_dim, out_dim, num_P, num_A,
                 num_heads, msg_dim=16, use_CNN=True, use_real=True, use_tanh=False,
                 per_class_critic=False, per_agent_critic=False, device=None,
                 with_two_state=False, obs=1, comm_range_P=-1, comm_range_A=-1,
                 lossy_comm=False, min_comm_loss=0, max_comm_loss=0.3, tensor_obs=False, total_state_action_in_batch=500,
                 action_vision=-1):
        super(UAVNetA2CEasy, self).__init__()

        self.device = device


        self.num_P = num_P
        self.num_A = num_A

        self.tensor_obs = tensor_obs
        self.vision = in_dim_raw['vision']
        self.obs_squares_dict = {0: 1,1:9, 2: 25, 3: 49}

        self.in_dim = in_dim
        self.P_s = in_dim['P'] - in_dim['state']
        self.world_dim = int(math.sqrt(self.P_s))

        hid_dim = hid_dim
        out_dim = out_dim
        self.hid_size = 32

        hid_dim_input = {}
        for key in hid_dim:
            hid_dim_input[key] = hid_dim[key] * num_heads

        # preprocessing of UAV status
        self.obs_squares = 1 if obs is None else obs

        ''' TODO: May need to multiply output size by obs_squares

            if so we'll increase the outputs below then increase hidden
            arrays in init_hidden and increase the in_dim and hid_dim of HeteroGATLayerReal for layer1
        '''

        self.prepro_obs = nn.Linear(in_dim['state'] * self.obs_squares, in_dim['state'] * self.obs_squares)
        self.prepro_stat = nn.Linear(self.P_s * self.obs_squares, self.P_s * self.obs_squares)

        self.total_state_action_in_batch = total_state_action_in_batch
        self.size_of_batch = 1
        # TODO: maybe delete next line
        # self.init_hidden(self.size_of_batch)
        # TODO: why is next line in_dim['state']
        self.f_module_obs = nn.LSTMCell(in_dim_raw['state'] * self.obs_squares, in_dim_raw['state'])

        self.f_module_stat = nn.LSTMCell(self.P_s * self.obs_squares, self.P_s)
        self.use_CNN = False
        self.use_real = use_real
        in_dim['obs_squares'] = self.obs_squares

        # TODO: add this back in
        # self.features = nn.Linear(in_dim_raw['Sensor'][0] * in_dim_raw['Sensor'][1] * in_dim_raw['Sensor'][2],
        #                           in_dim['P'] - in_dim['A'])

        # gnn layers
        # N layers = N round of communication during one time stamp
        if use_real:

            self.layer1 = MultiHeteroGATLayerReal(in_dim, hid_dim,
                                                  num_heads)
            self.layer2 = MultiHeteroGATLayerReal(hid_dim_input, out_dim,
                                                  num_heads, merge='avg')
        else:

            self.layer1 = MultiHeteroGATLayerBinary(in_dim, hid_dim,
                                                    num_heads, msg_dim)
            self.layer2 = MultiHeteroGATLayerBinary(hid_dim_input, out_dim,
                                                    num_heads, msg_dim, merge='avg')

        self.relu = nn.ReLU()
        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.per_class_critic = per_class_critic
        self.per_agent_critic = per_agent_critic
        self.with_two_state = with_two_state
        if self.per_class_critic and self.per_agent_critic:
            print('Error！ per_class_critic and per_agent_critic cannot be both True')
        if self.per_class_critic:
            self.P_critic_head = nn.Linear(out_dim['state'], 1)
            self.A_critic_head = nn.Linear(out_dim['state'], 1)
        elif self.per_agent_critic:
            self.P_critic_head = nn.Linear(out_dim['P'] + out_dim['state'], 1)
            self.A_critic_head = nn.Linear(out_dim['A'] + out_dim['state'], 1)
        else:
            self.critic_head = nn.Linear(out_dim['state'], 1)

    '''
    input
        g: DGL heterograph
        raw_f_d: dictionary of raw input features
            raw_f_d['P_s']: sensor images Np x [1 x 1 x Height x Width]
            raw_f_d['P']: status Np x H
            raw_f_d['A']: status Na x H
            raw_f_d['state']: 1x4 system info
    '''

    ########## from MAGIC ##############
    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None
        x, extras = x
        # x = self.encoder(x)

        # hidden_state, cell_state = extras

        return x, extras

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        h = {}

        h['P_s'] = tuple((torch.zeros(self.num_P, self.P_s, requires_grad=True).to(self.device),
                          torch.zeros(self.num_P, self.P_s, requires_grad=True).to(self.device)))
        h['P_o'] = tuple((torch.zeros(self.num_P, self.in_dim['state'], requires_grad=True).to(self.device),
                          torch.zeros(self.num_P, self.in_dim['state'], requires_grad=True).to(self.device)))

        h['A_s'] = tuple((torch.zeros(self.num_A, self.in_dim['A'], requires_grad=True).to(self.device),
                          torch.zeros(self.num_A, self.in_dim['A'], requires_grad=True).to(self.device)))

        return h



    def remove_excess_action_features_from_all(self, x):
        P = torch.zeros(1, self.P_s * self.obs_squares)
        A = torch.zeros(1, self.in_dim['A'] * self.obs_squares)

        for i in range(self.num_P + self.num_A):
            x_pos, f_pos = 0, 0

            if i < self.num_P:
                dx, df = self.in_dim['P'], self.P_s
            else:
                dx, df = self.in_dim['P'], self.in_dim['A']

            f_i = torch.zeros(1, df * self.obs_squares)

            for _ in range(self.obs_squares):
                f_i[0, f_pos:f_pos + df] = x[0][i][x_pos:x_pos + df]

                x_pos += dx
                f_pos += df

            if i < self.num_P:
                if i == 0:
                    P = f_i
                else:
                    P = torch.cat([P, f_i], axis=0)
            else:
                if i == self.num_P:
                    A = f_i
                else:
                    A = torch.cat([A, f_i], axis=0)

        return P, A

    def get_obs_features(self, x):
        P = torch.zeros(1, self.in_dim['state'] * self.obs_squares)

        for i in range(self.num_P):
            ob_pos, Pi_pos = 0, 0
            P_i = torch.zeros(1, self.in_dim['state'] * self.obs_squares)

            for _ in range(self.obs_squares):
                ob_feat = ob_pos + self.P_s
                P_i[0, Pi_pos:Pi_pos + self.in_dim['state']] = x[0][i][ob_feat:ob_feat + self.in_dim['state']]

                ob_pos += self.P_s
                Pi_pos += self.in_dim['state']

            if i == 0:
                P = P_i
            else:
                P = torch.cat([P, P_i], axis=0)

        return P

    def get_obs_features_uneven_obs(self, x):
        # P = torch.zeros(1, self.in_dim['state'] * self.obs_squares_dict[self.vision])
        # A = torch.zeros(1, self.in_dim['state'] * self.obs_squares_dict[self.action_vision])

        P = torch.zeros(1, self.num_P, self.in_dim['state'] * self.obs_squares)
        A = torch.zeros(1, self.num_A, self.in_dim['state'] * self.obs_squares_dict[self.action_vision])

        for i in range(self.num_P):
            # Ex. for 10x10 obs index is 100, pos index is 0
            observation_index = int((x.shape[2] - self.in_dim['state'] * self.obs_squares) / self.obs_squares)
            position_index = 0
            P_i = torch.zeros(1, self.in_dim['state'] * self.obs_squares)

            position_additive = (x.shape[2] - self.in_dim['state'] * self.obs_squares) / self.obs_squares
            observation_additive = self.in_dim['state']

            next_position_index = int(position_index + position_additive)
            next_observation_index = int(observation_index + observation_additive)
            P_i_index_counter = 0
            P_i_next_index_counter = observation_additive

            for j in range(self.obs_squares):
                P_i[0, P_i_index_counter:P_i_next_index_counter] = x[0][i][observation_index:next_observation_index]

                # position starts at last observation
                position_index = int(next_observation_index)
                # observation comes after position
                observation_index = int(position_index + position_additive)

                next_position_index = int(position_index + position_additive)
                next_observation_index = int(observation_index + observation_additive)
                P_i_index_counter += observation_additive
                P_i_next_index_counter += observation_additive

            P[:, i] = P_i

        # action agents
        grid = np.arange(self.obs_squares).reshape((int(np.sqrt(self.obs_squares)), int(np.sqrt(self.obs_squares))))
        middle_point = int(np.median(np.arange(int(np.sqrt(self.obs_squares)))))

        indexes_of_action_observation = grid[middle_point-self.action_vision:middle_point+self.action_vision+1,
                                        middle_point-self.action_vision:middle_point+self.action_vision+1].flatten()

        x_shaped_to_grid = x.reshape(1, x.shape[1], -1, self.in_dim['P'])

        for i in range(self.num_P, self.num_P + self.num_A):
            A_i = x_shaped_to_grid[0, i, indexes_of_action_observation, self.in_dim['A']:].reshape(1, -1)
            A[:, i-self.num_P] = A_i

        return P, A

    def get_states_obs_from_tensor(self, x):
        x_per_stat = []
        x_act_stat = []
        x_per_obs = []
        for i in range(self.num_P):
            x_per_stat.append(x[i][1])
        for i in range(self.num_P, self.num_P + self.num_A):
            x_act_stat.append(x[i][1])
        for i in range(self.num_P):
            x_per_obs.append(x[i][0])
        return torch.tensor(x_per_stat), torch.tensor(x_act_stat), torch.tensor(x_per_obs)

    def forward(self, x, g):
        x, extras = self.forward_state_encoder(x)

        # hidden_state = hidden_state.to(self.device)
        # cell_state = cell_state.to(self.device)
        hidden_state_per_stat = extras['P_s'][0].to(self.device)
        hidden_state_act_stat = extras['A_s'][0].to(self.device)
        hidden_state_per_obs = extras['P_o'][0].to(self.device)
        cell_state_per_stat = extras['P_s'][1].to(self.device)
        cell_state_act_stat = extras['A_s'][1].to(self.device)
        cell_state_per_obs = extras['P_o'][1].to(self.device)

        x_per_stat, x_act_stat = self.remove_excess_action_features_from_all(x)
        x_per_obs = self.get_obs_features(x).to(self.device)

        feat_dict = {}

        '''
        Data preprocessing - P
        '''
        # state_per_stat = torch.tensor(x_per_stat, dtype=torch.float64).to(self.device)

        state_per_stat = x_per_stat.clone().detach()

        state_per_stat = self.relu(self.prepro_stat(state_per_stat)) # output is 2,225


        hidden_state_per_stat, cell_state_per_stat = self.f_module_stat(state_per_stat.squeeze(),
                                                                        (hidden_state_per_stat.double(),
                                                                         cell_state_per_stat.double()))

        x_per_obs = x_per_obs.to(self.device)

        x_per_obs = self.relu(self.prepro_obs(x_per_obs))


        hidden_state_per_obs, cell_state_per_obs = self.f_module_obs(x_per_obs.squeeze(),
                                                                     (hidden_state_per_obs.double(),
                                                                      cell_state_per_obs.double()))
        feat_dict['P'] = torch.cat([hidden_state_per_stat, hidden_state_per_obs], dim=1)
        if self.num_A != 0:
            state_act = torch.Tensor(x_act_stat).to(self.device)

            state_act = self.relu(self.prepro_stat(state_act))


            hidden_state_act_stat, cell_state_act_stat = self.f_module_stat(state_act.squeeze().reshape((state_act.shape[0], -1)),
                                                                        (hidden_state_act_stat, cell_state_act_stat))
            feat_dict['A'] = hidden_state_act_stat

        # complete_state = torch.cat([state_per, state_act])
        # hidden_state, cell_state = self.f_module(complete_state.squeeze(), (hidden_state, cell_state))
        # feat_dict['P'] = hidden_state[0:2]
        # feat_dict['A'] = hidden_state[2].reshape(1, 32)
        '''
        # Np x 1 x H x W
        x = self.features(raw_f_d['P_s'])
        # Np x dim x H/4 x W/4
        x = self.avgpool(x)
        # Np x dim x 1 x 1
        p_s = torch.flatten(x, 1) # this works well when Np == 1
        # Np x dim
        '''
        '''
        Data preprocessing - A
        # '''
        # status_A = self.prepro(raw_f_d['A'])

        # feat_dict['A'] = status_A

        # add state node
        if self.with_two_state:
            feat_dict['state'] = torch.tensor([
                [self.num_P, self.num_A, self.world_dim, self.total_state_action_in_batch],
                [self.num_P, self.num_A, self.world_dim, self.total_state_action_in_batch]
            ]).to(self.device)
        else:
            feat_dict['state'] = torch.tensor([self.num_P, self.num_A, self.world_dim, self.total_state_action_in_batch]).to(self.device)

        h1 = self.layer1(g, feat_dict)
        h2 = self.layer2(g, h1)

        # get critic prediction, 1x1
        if self.per_class_critic:
            if self.with_two_state:
                # h2['state'] is 2 x dim, first 1 x dim is p state, second is a state
                P_critic_value = self.P_critic_head(self.relu(h2['state'][:1, :]))
                A_critic_value = self.A_critic_head(self.relu(h2['state'][1:, :]))
                if self.use_tanh:
                    P_critic_value = self.tanh(P_critic_value)
                    A_critic_value = self.tanh(A_critic_value)
            else:
                P_critic_value = self.P_critic_head(self.relu(h2['state']))
                A_critic_value = self.A_critic_head(self.relu(h2['state']))
                if self.use_tanh:
                    P_critic_value = self.tanh(P_critic_value)
                    A_critic_value = self.tanh(A_critic_value)

            h = {}
            h['P_s'] = hidden_state_per_stat, cell_state_per_stat
            h['P_o'] = hidden_state_per_obs, cell_state_per_obs
            h['A_s'] = hidden_state_act_stat, cell_state_act_stat

            return h2, P_critic_value, A_critic_value, h
        elif self.per_agent_critic:
            # P agents
            # num_P = len(h2['P'])
            num_P = self.num_P
            tmp_list = []
            for i in range(num_P):
                tmp_list.append(h2['state'])

            hp_emb = h2['P']  # num_P x out['P']
            hs_emb = torch.cat(tmp_list)  # num_P x out['state']
            P_critic_input = torch.cat((hp_emb, hs_emb), dim=1)  # num_P x out['P'+'state']
            P_critic_value = self.P_critic_head(self.relu(P_critic_input))  # num_P x 1
            # A agents
            num_A = len(h2['A'])
            tmp_list = []
            for i in range(num_A):
                tmp_list.append(h2['state'])

            ha_emb = h2['A']  # num_A x out['P']
            hs_emb = torch.cat(tmp_list)  # num_A x out['state']
            A_critic_input = torch.cat((ha_emb, hs_emb), dim=1)  # num_A x out['A'+'state']
            A_critic_value = self.A_critic_head(self.relu(A_critic_input))  # num_A x 1

            if self.use_tanh:
                P_critic_value = self.tanh(P_critic_value)
                A_critic_value = self.tanh(A_critic_value)
            h = {}
            h['P_s'] = hidden_state_per_stat, cell_state_per_stat
            h['P_o'] = hidden_state_per_obs, cell_state_per_obs
            h['A_s'] = hidden_state_act_stat, cell_state_act_stat
            return h2, P_critic_value, A_critic_value, h
        else:
            critic_value = self.critic_head(self.relu(h2['state']))
            if self.use_tanh:
                critic_value = self.relu(critic_value)

            h = {}
            h['P_s'] = hidden_state_per_stat, cell_state_per_stat
            h['P_o'] = hidden_state_per_obs, cell_state_per_obs

            h['A_s'] = hidden_state_act_stat, cell_state_act_stat

            return h2, critic_value, h


# GNN - non-batched version
class UAVNet(nn.Module):

    # in_dim_raw is the dict of raw inputs
    # in_dim_raw['Status']: UAV status vector, same for P and A
    # in_dim_raw['Sensor']: (H, W) height and width of sensor image
    # use_real: select between real-valued message & binary message
    def __init__(self, in_dim_raw, in_dim, hid_dim, out_dim, num_heads,
                 msg_dim=[], use_CNN=True, use_real=True, use_tanh=False,
                 per_class_critic=False, device=None):
        super(UAVNet, self).__init__()

        self.device = device

        hid_dim_input = {}
        for key in hid_dim:
            hid_dim_input[key] = hid_dim[key] * num_heads
        self.size_of_batch = 1
        self.hid_size = 32
        self.prepro_obs = nn.Linear(4, 4)
        self.prepro_stat = nn.Linear(25, 25)
        self.size_of_batch = 1
        self.init_hidden(self.size_of_batch)
        self.f_module_obs = nn.LSTMCell(4, 4)
        self.f_module_stat = nn.LSTMCell(25, 25)

        self.use_CNN = False

        # preprocessing of Sensor image

        # if self.use_CNN:
        #     # [CONV - RELU - POOL] + Flatten
        #     self.features = nn.Sequential(
        #         nn.Conv2d(in_dim_raw['Sensor'][0], in_dim['P'] - in_dim['A'], kernel_size=3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d(kernel_size=2, stride=1)
        #     )
        #
        #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # else:
        #     self.features = nn.Linear(in_dim_raw['Sensor'][0] * in_dim_raw['Sensor'][1] * in_dim_raw['Sensor'][2],
        #                               in_dim['P'] - in_dim['A'])

        # gnn layers
        self.use_real = use_real

        # real-valued message
        # N layers = N round of communication during one time stamp
        self.layer1 = MultiHeteroGATReal(in_dim, hid_dim, num_heads)
        # self.layer2 = MultiHeteroGATLayer(hid_dim_input, hid_dim, msg_dim, num_heads)
        self.layer2 = MultiHeteroGATReal(hid_dim_input, out_dim,
                                         num_heads, merge='avg')
        # NOTE USING TANH
        self.relu = nn.Tanh()

    '''
    input
        g: DGL heterograph
        raw_f_d: dictionary of raw input features
            raw_f_d['P_s']: sensor images Np x Depth x Height x Width
            raw_f_d['P']: status Np x H
            raw_f_d['A']: status Na x H
    '''

    ########## from MAGIC ##############
    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        x, extras = x
        # x = self.encoder(x)

        # hidden_state, cell_state = extras

        return x, extras

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        h = {}
        h['P_s'] = tuple((torch.zeros(batch_size * 2, 25, requires_grad=True).to(self.device).to(self.device),
                          torch.zeros(batch_size * 2, 25, requires_grad=True).to(self.device).to(self.device)))
        h['P_o'] = tuple((torch.zeros(batch_size * 2, 4, requires_grad=True).to(self.device).to(self.device),
                          torch.zeros(batch_size * 2, 4, requires_grad=True).to(self.device).to(self.device)))

        h['A_s'] = tuple((torch.zeros(batch_size * 1, 25, requires_grad=True).to(self.device).to(self.device),
                          torch.zeros(batch_size * 1, 25, requires_grad=True).to(self.device).to(self.device)))

        return h

    ########## from MAGIC ##############

    def remove_excess_action_features_from_all(self, x):
        b = x[0][:2][0][:25].reshape(1, 25)
        c = x[0][:2][1][:25].reshape(1, 25)
        d = torch.cat([b, c], axis=0)
        return d, x[0][2][:25].reshape(1, 25)

    def get_obs_features(self, x):
        a = x[0][:2][0][25:].reshape(1, 4)
        b = x[0][:2][1][25:].reshape(1, 4)
        c = torch.cat([a, b], axis=0)
        return c

    def forward(self, x, g):
        x, extras = self.forward_state_encoder(x)
        # hidden_state = hidden_state.to(self.device)
        # cell_state = cell_state.to(self.device)
        hidden_state_per_stat = extras['P_s'][0]
        hidden_state_act_stat = extras['A_s'][0]
        hidden_state_per_obs = extras['P_o'][0]
        cell_state_per_stat = extras['P_s'][1]
        cell_state_act_stat = extras['A_s'][1]
        cell_state_per_obs = extras['P_o'][1]
        x_per_stat, x_act_stat = self.remove_excess_action_features_from_all(x)
        x_per_obs = self.get_obs_features(x).to(self.device)
        feat_dict = {}
        '''
        Data preprocessing - P
        '''
        state_per_stat = torch.Tensor(x_per_stat).to(self.device)
        state_per_stat = self.relu(self.prepro_stat(state_per_stat))
        hidden_state_per_stat, cell_state_per_stat = self.f_module_stat(state_per_stat.squeeze(),
                                                                        (hidden_state_per_stat, cell_state_per_stat))
        hidden_state_per_obs, cell_state_per_obs = self.f_module_obs(x_per_obs.squeeze(),
                                                                     (hidden_state_per_obs, cell_state_per_obs))

        feat_dict['P'] = torch.cat([hidden_state_per_stat, hidden_state_per_obs], dim=1)

        state_act = torch.Tensor(x_act_stat).to(self.device)
        state_act = self.relu(self.prepro_stat(state_act))
        hidden_state_act_stat, cell_state_act_stat = self.f_module_stat(state_act,
                                                                        (hidden_state_act_stat, cell_state_act_stat))
        feat_dict['A'] = hidden_state_act_stat
        # h1 = self.layer1(g, feat_dict)
        # h2 = self.layer2(g, h1)
        # feat_dict['P'] = torch.cat([status_P, p_s], dim=1)

        '''
        Data preprocessing - A
        '''
        # status_A = self.relu(self.prepro(raw_f_d['A']))

        # feat_dict['A'] = status_A

        h1 = self.layer1(g, feat_dict)
        h2 = self.layer2(g, h1)
        h = {}
        h['P_s'] = hidden_state_per_stat, cell_state_per_stat
        h['P_o'] = hidden_state_per_obs, cell_state_per_obs

        h['A_s'] = hidden_state_act_stat, cell_state_act_stat

        return h2, h
