# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 00:22:47 2020

@author: pheno

HetGNN-based policy for UAV coordination
"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from hetgat.uavnet import UAVNetA2CEasy, UAVNet
from hetgat.utils import build_hetgraph
from torch.distributions import Categorical
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import MultiStepLR
import time

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

class PGPolicy(object):

    def __init__(self, in_dim_raw, in_dim, hid_dim, out_dim,
                 msg_dim=[], num_heads=2, device=torch.device('cuda'),
                 gamma=0.95, lr=1e-4, weight_decay=1e-3,
                 milestones=[30, 80], lr_gamma=0.1,
                 use_real=True, use_CNN=True, max_grad_norm=0.75):
        self.device = device
        self.use_real = use_real

        self.model = UAVNet(in_dim_raw, in_dim, hid_dim, out_dim, num_heads,
                            msg_dim=msg_dim, use_CNN=use_CNN, use_real=use_real,
                            device=device).to(self.device)

        self.gamma = gamma
        self.lmbda = 0

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)

        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=lr_gamma)

        self.sum_rs = []
        self.sum_std = []
        self.mean_p_c = []
        self.std_p_c = []
        self.mean_a_c = []
        self.std_a_c = []
        self.mean_steps = []
        self.std_steps = []
        self.loss_history = []
        self.initialize_batch(100)

    '''
    Sample actions using UAVNet
    Parameters:
        num_P, num_A, PnP, PnA, AnA: info for graph construction
            num_P: number of P
            num_A: number of A
            PnP: (P, P) pairs within communication range
            PnA: (P, A) pairs within communication range
            AnA: (A, A) pairs within communication range
        r_f_d: dictionary of raw input features (numpy array)
            r_f_d['P_s']: sensor images tensor Np x Depth x Height x Width
            r_f_d['P']: status Np x H
            r_f_d['A']: status Na x H
    Returns:
        actions: dictionary of sampled actions (numpy array int64)
            actions['P']: actions of P
            actions['A']: actions of A
    '''

    def get_actions(self, num_P, num_A, PnP, PnA, AnA, r_f_d):
        # construct heterograph
        g = build_hetgraph(num_P, num_A, PnP, PnA, AnA, with_self_loop = False)
        g = g.to(self.device)

        # get torch tensor
        r_f_d_tensor = {}
        r_f_d_tensor['P'] = torch.Tensor(r_f_d['P']).to(self.device)
        r_f_d_tensor['A'] = torch.Tensor(r_f_d['A']).to(self.device)
        # N x 1 x H x W
        r_f_d_tensor['P_s'] = torch.Tensor(r_f_d['P_s']).to(self.device)

        # sample actions
        actions = {}
        with torch.no_grad():
            # get logits
            results = self.model(g, r_f_d_tensor)
            # sample for P
            mp = Categorical(logits=results['P'])
            p_idx = mp.sample()
            actions['P'] = p_idx.cpu().numpy()
            # sample for A
            ma = Categorical(logits=results['A'])
            a_idx = ma.sample()
            actions['A'] = a_idx.cpu().numpy()

        return actions

    '''
    Initialize batch buffer
    '''

    def initialize_batch(self, batch_size):
        self.batch_P_log_probs = [[] for i in range(batch_size)]
        self.batch_A_log_probs = [[] for i in range(batch_size)]
        self.batch_rewards = [[] for i in range(batch_size)]

    '''
    Batch version
    '''
    def modify_per_to_be_proper_size(self, per_prob):
        l = torch.zeros(2,6)
        l[0][0] = per_prob[0][0]
        l[0][1] = per_prob[0][1]
        l[0][2] = per_prob[0][2]
        l[0][3] = per_prob[0][3]
        l[0][4] = per_prob[0][4]
        l[1][0] = per_prob[1][0]
        l[1][1] = per_prob[1][1]
        l[1][2] = per_prob[1][2]
        l[1][3] = per_prob[1][3]
        l[1][4] = per_prob[1][4]
        return l.to(self.device)

    def batch_select_action_universal(self, x,  i_b):
        # construct heterograph
        # g = build_hetgraph(num_P, num_A, PnP, PnA, AnA, with_self_loop=self.use_real)
        g = build_hetgraph(with_state=False, with_self_loop =False)
        g = g.to(self.device)

        actions = {}

        results, prev_hid = self.model(x, g)
        self.results = results
        self.i_b = i_b
        results['P'] = self.modify_per_to_be_proper_size(results['P'])
        # return_results = torch.cat((results['P'],results['A']),dim=0)
        return_results = self.results
        # print(results['P'])
        # sample for P
        # mp = Categorical(logits=results['P'])
        # p_idx = mp.sample()  # size N
        # actions['P'] = p_idx.cpu().numpy()
        # # save log_prob of P agents
        # self.batch_P_log_probs[i_b].append(mp.log_prob(p_idx))  # size N
        #
        # # sample for A
        # ma = Categorical(logits=results['A'])
        # a_idx = ma.sample()
        # actions['A'] = a_idx.cpu().numpy()
        # # save log_prob of A agents
        # self.batch_A_log_probs[i_b].append(ma.log_prob(a_idx))
        return return_results, prev_hid

    def append_log_probs_properly(self, actual):
        mp = Categorical(logits=self.results['P'])
        p_idx = torch.Tensor([actual[0][0], actual[0][1]]).to(self.device)
        # p_idx = mp.sample()  # size N
        # actions['P'] = p_idx.cpu().numpy()
        # save log_prob of P agents
        self.batch_P_log_probs[self.i_b].append(mp.log_prob(p_idx))  # size N

        # sample for A
        ma = Categorical(logits=self.results['A'])
        a_idx = torch.Tensor([actual[0][2]]).to(self.device)
        # actions['A'] = a_idx.cpu().numpy()
        # save log_prob of A agents
        self.batch_A_log_probs[self.i_b].append(ma.log_prob(a_idx))

    def batch_select_action(self, x,  i_b):
        # construct heterograph
        # g = build_hetgraph(num_P, num_A, PnP, PnA, AnA, with_self_loop=self.use_real)
        g = build_hetgraph(with_state=False, with_self_loop =False)
        g = g.to(self.device)

        actions = {}

        results, prev_hid = self.model(x, g)
        # print(results['P'])
        # sample for P
        mp = Categorical(logits=results['P'])
        p_idx = mp.sample()  # size N
        actions['P'] = p_idx.cpu().numpy()
        # save log_prob of P agents
        self.batch_P_log_probs[i_b].append(mp.log_prob(p_idx))  # size N

        # sample for A
        ma = Categorical(logits=results['A'])
        a_idx = ma.sample()
        actions['A'] = a_idx.cpu().numpy()
        # save log_prob of A agents
        self.batch_A_log_probs[i_b].append(ma.log_prob(a_idx))

        return actions, prev_hid

    '''
    Batch version
    '''


    def batch_finish_episode(self, batch_size, sim_time, num_P=2, num_A=1):
        batch_policy_loss = [[] for i in range(batch_size)]
        batch_total_loss = []

        # pad episodes with early termination, N x total_number x Time
        # using max reward which is 0 (note PREY_REWARD = 0 from env code)
        num_agents = num_P + num_A
        batch_returns = torch.zeros(batch_size, num_agents, sim_time).to(self.device)

        # 1. compute total reward of each episode, per agent
        # first num_P are P agents, next num_A are A agents
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            for j in range(num_agents):
                batch_returns[i_b][j][:r_size] = self.batch_r(i_b,j)

        # 2. compute time-based baseline values
        # batch_baselines = torch.mean(batch_returns, dim=0)
        # per agent
        P_returns = batch_returns[:,:num_P,:] # N x num_P x Time
        A_returns = batch_returns[:,num_P:,:] # N x num_A x Time

        # P_baselines = torch.mean(P_returns, dim=(0,1)) # Size Time
        # A_baselines = torch.mean(A_returns, dim=(0,1)) # Size Time
        P_baselines = torch.mean(P_returns, dim=0) # num_P x Time
        A_baselines = torch.mean(A_returns, dim=0) # num_A x Time

        # 3. calculate advantages for each transition
        # batch_advs = batch_returns - batch_baselines
        P_advs = P_returns - P_baselines
        A_advs = A_returns - A_baselines

        # 4. normalize advantages within a batch
        eps = np.finfo(np.float32).eps.item()
        # adv_mean = batch_advs.mean()
        # adv_std = batch_advs.std()
        # batch_advs_norm = (batch_advs - adv_mean) / (adv_std + eps)
        P_adv_mean = P_advs.mean()
        P_adv_std = P_advs.std()
        P_advs_norm = (P_advs - P_adv_mean) / (P_adv_std + eps)

        A_adv_mean = A_advs.mean()
        A_adv_std = A_advs.std()
        A_advs_norm = (A_advs - A_adv_mean) / (A_adv_std + eps)

        # 5. calculate loss for each episode in the batch
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            # num_P * r_size
            P_log_prob_list = torch.stack(self.batch_P_log_probs[i_b], dim=1)
            P_adv_n = P_advs_norm[i_b, :, :r_size]  # num_P x r_size

            batch_total_loss.append(torch.sum(-P_log_prob_list * P_adv_n))

            A_log_prob_list = torch.stack(self.batch_A_log_probs[i_b], dim=1)
            A_adv_n = A_advs_norm[i_b, :, :r_size]

            batch_total_loss.append(torch.sum(-A_log_prob_list * A_adv_n))
            # for st in range(sim_time):
            #     # check transtions before early termination
            #     if st < len(self.batch_P_log_probs[i_b]):
            #         # adv_n = batch_advs_norm[i_b][st]
            #         # # get log prob of all P agents
            #         # P_log_prob_list = self.batch_P_log_probs[i_b][st]
            #         # if len(P_log_prob_list) > 0:
            #         #     batch_policy_loss[i_b].append(-P_log_prob_list.sum() * adv_n)
            #
            #         # # get log prob of all A agents
            #         # A_log_prob_list = self.batch_A_log_probs[i_b][st]
            #         # if len(A_log_prob_list) > 0:
            #         #     batch_policy_loss[i_b].append(-A_log_prob_list.sum() * adv_n)
            #         # P agents
            #         P_log_prob_list = self.batch_P_log_probs[i_b][st]
            #         if len(P_log_prob_list) > 0:
            #             # for j in range(num_P):
            #             #     adv_n = P_advs_norm[i_b][j][st]
            #             #     batch_policy_loss[i_b].append(-P_log_prob_list[j] * adv_n)
            #             adv_n_list = P_advs_norm[i_b,:,st] # size num_P
            #             batch_policy_loss[i_b].append(torch.sum(-P_log_prob_list * adv_n_list))
            #
            #         # A agents
            #         A_log_prob_list = self.batch_A_log_probs[i_b][st]
            #         if len(A_log_prob_list) > 0:
            #             # for j in range(num_A):
            #             #     adv_n = A_advs_norm[i_b][j][st]
            #             #     batch_policy_loss[i_b].append(-A_log_prob_list[j] * adv_n)
            #             adv_n_list = A_advs_norm[i_b,:,st] # size num_A
            #             batch_policy_loss[i_b].append(torch.sum(-A_log_prob_list * adv_n_list))

            # if len(batch_policy_loss[i_b]) > 0:
            #     batch_total_loss.append(
            #         torch.stack(batch_policy_loss[i_b]).sum())

        # reset gradients
        self.optimizer.zero_grad()

        # sum up over all batches
        total_loss = torch.stack(batch_total_loss).mean()
        loss_np = total_loss.data.cpu().numpy()

        # perform backprop
        total_loss.backward()

        # perform Gradient Clipping-by-norm
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # self.optimizer.step()

        # reset rewards and action buffer
        for i_b in range(batch_size):
            del self.batch_rewards[i_b][:]
            del self.batch_P_log_probs[i_b][:]
            del self.batch_A_log_probs[i_b][:]

        return loss_np

    '''
    Compute total reward of each episode
    '''

    def batch_r(self, i_b, agent_idx):
        R = 0.0
        returns = []  # list to save the true values

        for rw in self.batch_rewards[i_b][::-1]:
            # calculate the discounted value
            R = rw[agent_idx] + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)
        return returns

    '''
    Adjust learning rate using lr_scheduler
    '''

    def adjust_lr(self, metrics=0.0):
        self.lr_scheduler.step()


'''
Actor Critic - batch version
    use GAE
    add use_easy for supporting FireCommander_Easy
'''
class A2CPolicy(object):

    def __init__(self, in_dim_raw, in_dim, hid_dim, out_dim, num_P, num_A, msg_dim=16,
                 num_heads=2, device=torch.device('cuda'),
                 gamma=0.95, lr=1e-4, weight_decay=1e-3, lmbda=0.95,
                 milestones=[30, 80], lr_gamma=0.1, use_easy=False,
                 use_real=False, use_CNN=True, use_tanh=False,
                 max_grad_norm=0.75, per_class_critic=False,
                 per_agent_critic=False, with_two_state=True, obs=None,
                 comm_range_P=-1, comm_range_A=-1, lossy_comm=False, tensor_obs=False,
                 min_comm_loss=0, max_comm_loss=0.3, total_state_action_in_batch=500,
                 action_vision=-1):

        self.device = device
        self.use_real = use_real
        self.per_class_critic = per_class_critic
        self.per_agent_critic = per_agent_critic
        self.with_two_state = with_two_state

        self.num_P = num_P
        self.num_A = num_A
        self.comm_range_P = comm_range_P
        self.comm_range_A = comm_range_A
        self.in_dim = in_dim
        self.tensor_obs = tensor_obs

        self.model = UAVNetA2CEasy(in_dim_raw, in_dim, hid_dim, out_dim,
            num_P, num_A, num_heads, msg_dim=msg_dim, use_CNN=use_CNN,
            use_real=use_real, use_tanh=use_tanh, per_class_critic=per_class_critic,
            per_agent_critic=per_agent_critic, device=device,
            with_two_state=with_two_state, obs=obs, comm_range_P=comm_range_P,
            comm_range_A=comm_range_A, lossy_comm=lossy_comm,
            min_comm_loss=min_comm_loss, max_comm_loss=max_comm_loss,
            tensor_obs=tensor_obs, action_vision=action_vision).to(self.device)

        self.gamma = gamma
        self.lmbda = lmbda
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)
        # self.optimizer = torch.optim.Adam(self.model.layer1.heads[0].attn_fc_p2a.parameters(), lr=.1,
        #                                   weight_decay=self.weight_decay)

        self.lr_scheduler = MultiStepLR(self.optimizer, milestones = milestones, gamma = lr_gamma)

        self.sum_rs = []
        self.sum_std = []
        self.mean_p_c = []
        self.std_p_c = []
        self.mean_a_c = []
        self.std_a_c = []
        self.mean_steps = []
        self.std_steps = []
        self.loss_history = []

        self.critic_loss_tracker = []
        self.policy_loss_tracker = []
        self.initialize_batch(total_state_action_in_batch)

    '''
    Sample actions using UAVNetA2C
    Parameters:
        num_P, num_A, PnP, PnA, AnA: info for graph construction
            num_P: number of P
            num_A: number of A
            PnP: (P, P) pairs within communication range
            PnA: (P, A) pairs within communication range
            AnA: (A, A) pairs within communication range
        r_f_d: dictionary of raw input features (numpy array)
            r_f_d['P_s']: sensor images tensor Np x Depth x Height x Width
            r_f_d['P']: status Np x H
            r_f_d['A']: status Na x H
            r_f_d['state']: 1x5 system info
    Returns:
        actions: dictionary of sampled actions (numpy array int64)
            actions['P']: actions of P
            actions['A']: actions of A
    '''
    def get_actions(self, num_P, num_A, PnP, PnA, AnA, r_f_d):
        # construct heterograph
        g = build_hetgraph(num_P, num_A, PnP, PnA, AnA, with_state=True,
            with_self_loop=False, with_two_state=self.with_two_state,
            comm_range_P=self.comm_range_P, comm_range_A=self.comm_range_A)
        g = g.to(self.device)

        # get torch tensor
        r_f_d_tensor = {}
        r_f_d_tensor['P'] = torch.Tensor(r_f_d['P']).to(self.device)
        r_f_d_tensor['A'] = torch.Tensor(r_f_d['A']).to(self.device)
        # N x 1 x H x W
        r_f_d_tensor['P_s'] = torch.Tensor(r_f_d['P_s']).to(self.device)

        r_f_d_tensor['state'] = torch.Tensor(r_f_d['state']).to(self.device)

        # sample actions
        actions = {}
        with torch.no_grad():
            # get logits
            if self.per_class_critic or self.per_agent_critic:
                results, c_v, a_v = self.model(g, r_f_d_tensor)
            else:
                results, c_v = self.model(g, r_f_d_tensor)
            # sample for P
            mp = Categorical(logits = results['P'])
            p_idx = mp.sample()
            actions['P'] = p_idx.cpu().numpy()
            # sample for A
            ma = Categorical(logits = results['A'])
            a_idx = ma.sample()
            actions['A'] = a_idx.cpu().numpy()

        return actions

    '''
    Initialize batch buffer
    '''
    def initialize_batch(self, batch_size):
        self.batch_P_log_probs = [[] for i in range(batch_size)]
        self.batch_A_log_probs = [[] for i in range(batch_size)]
        self.batch_rewards = [[] for i in range(batch_size)]
        self.batch_saved_critics = [[] for i in range(batch_size)]
        self.batch_entrpies_p = [[] for i in range(batch_size)]
        self.batch_entrpies_a = [[] for i in range(batch_size)]
        if self.per_class_critic:
            self.batch_P_critics = [[] for i in range(batch_size)]
            self.batch_A_critics = [[] for i in range(batch_size)]
        elif self.per_agent_critic:
            self.batch_P_critics = [[] for i in range(batch_size)]
            self.batch_A_critics = [[] for i in range(batch_size)]


    def modify_per_to_be_proper_size(self, per_prob):
        l = torch.zeros(2,6)
        l[0][0] = per_prob[0][0]
        l[0][1] = per_prob[0][1]
        l[0][2] = per_prob[0][2]
        l[0][3] = per_prob[0][3]
        l[0][4] = per_prob[0][4]
        l[1][0] = per_prob[1][0]
        l[1][1] = per_prob[1][1]
        l[1][2] = per_prob[1][2]
        l[1][3] = per_prob[1][3]
        l[1][4] = per_prob[1][4]
        return l.to(self.device)

    '''
    A2C version
    '''
    def batch_select_action_universal(self, x, i_b):
        """
        A2C version
        """
        # construct heterograph
        pos = [p[:self.in_dim['A']] for p in x[0][0]]

        g = build_hetgraph(pos, num_P=self.num_P, num_A=self.num_A, with_state=True,
            with_self_loop=False, with_two_state=self.with_two_state,
            comm_range_P=self.comm_range_P, comm_range_A=self.comm_range_A)
        g = g.to(self.device)

        # sample actions
        actions = {}

        # get logits
        if self.per_class_critic or self.per_agent_critic:
            results, c_v, a_v, prev_hid = self.model(x, g)
        else:
            results, c_v, prev_hid = self.model(x, g)
        self.results = results

        self.i_b = i_b
        # results['P'] = self.modify_per_to_be_proper_size(results['P'])

        return_results = results #
        # return_results = torch.cat((results['P'],results['A']),dim=0)

        # # sample for P
        # mp = Categorical(logits = results['P'])
        # p_idx = mp.sample() # size N
        # actions['P'] = p_idx.cpu().numpy()
        # # save log_prob of P agents
        # self.batch_P_log_probs[i_b].append(mp.log_prob(p_idx)) # size N
        #
        # # sample for A
        # ma = Categorical(logits = results['A'])
        # a_idx = ma.sample()
        # actions['A'] = a_idx.cpu().numpy()
        # # save log_prob of A agents
        # self.batch_A_log_probs[i_b].append(ma.log_prob(a_idx))
        #
        # # save critic predictions
        if self.per_class_critic or self.per_agent_critic:
            self.batch_P_critics[i_b].append(c_v)
            self.batch_A_critics[i_b].append(a_v)
        else:
            self.batch_saved_critics[i_b].append(c_v)  # NEED THIS FOR GRAD

        return return_results, c_v, prev_hid

    def append_log_probs_properly(self, actual):
        mp = Categorical(logits=self.results['P'])
        p_idx = torch.Tensor(actual[0][:self.num_P]).to(self.device)

        # p_idx = mp.sample()  # size N
        # actions['P'] = p_idx.cpu().numpy()
        # save log_prob of P agents

        self.batch_P_log_probs[self.i_b].append(mp.log_prob(p_idx))  # size N

        # sample for A
        ma = Categorical(logits=self.results['A'])
        a_idx = torch.Tensor(actual[0][self.num_P:]).to(self.device)

        ################### calculate entropies from action distributions #############################################################
        self.entropies_p = mp.entropy().mean().to(self.device)
        self.entropies_a = ma.entropy().mean().to(self.device)

        # a_idx = mp.sample()
        # actions['A'] = a_idx.cpu().numpy()
        # save log_prob of A agents

        ma.log_prob(a_idx)
        self.batch_A_log_probs[self.i_b].append(ma.log_prob(a_idx))

        self.batch_entrpies_p[self.i_b].append(self.entropies_p)
        self.batch_entrpies_a[self.i_b].append(self.entropies_a)

    '''
    Batch version
    '''
    def batch_finish_episode(self, batch_size, sim_time, num_P=2, num_A=1):
        """
        A2C version
        """
        # batch_policy_loss = [[] for i in range(batch_size)]
        # batch_critic_loss = [[] for i in range(batch_size)]
        # batch_total_loss = []
        batch_total_policy_loss = []
        batch_total_critic_loss = []

        # pad episodes with early termination, N x total_number x Time
        # using max reward which is 0 (note PREY_REWARD = 0 from env code)
        num_agents = num_P + num_A
        batch_returns = torch.zeros(batch_size, num_agents, sim_time).to(self.device)
        batch_advs = torch.zeros(batch_size, num_agents, sim_time).to(self.device)

        # 1. compute total reward and advantage of each episode, per agent
        # first num_P are P agents, next num_A are A agents
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            for j in range(num_agents):
                batch_returns[i_b][j][:r_size], batch_advs[i_b][j][:r_size] = self.batch_GAE(i_b,j)

        P_advs = batch_advs[:,:num_P,:] # N x num_P x Time
        A_advs = batch_advs[:,num_P:,:] # N x num_A x Time

        critic_target = torch.mean(batch_returns, dim=1) # N x Time

        # 2. normalize advantages within a batch
        eps = np.finfo(np.float32).eps.item()
        P_adv_mean = P_advs.mean()
        P_adv_std = P_advs.std()
        P_advs_norm = (P_advs - P_adv_mean) / (P_adv_std + eps)

        A_adv_mean = A_advs.mean()
        A_adv_std = A_advs.std()
        A_advs_norm = (A_advs - A_adv_mean) / (A_adv_std + eps)

        # 3. calculate loss for each episode in the batch
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            # num_P * r_size
            P_log_prob_list = torch.stack(self.batch_P_log_probs[i_b], dim=1)
            P_adv_n = P_advs_norm[i_b, :, :r_size]  # num_P x r_size

            batch_total_policy_loss.append(torch.sum(-P_log_prob_list * P_adv_n))

            A_log_prob_list = torch.stack(self.batch_A_log_probs[i_b], dim=1)
            A_adv_n = A_advs_norm[i_b, :, :r_size]

            batch_total_policy_loss.append(torch.sum(-A_log_prob_list * A_adv_n))

            # critic size [r_size]
            critic_list = torch.stack(self.batch_saved_critics[i_b]).squeeze()
            batch_total_critic_loss.append(
                F.mse_loss(critic_list, critic_target[i_b][:r_size]))

            # for st in range(sim_time):
            #     # check transtions before early termination
            #     if st < len(self.batch_P_log_probs[i_b]):
            #         # P agents
            #         P_log_prob_list = self.batch_P_log_probs[i_b][st]
            #         if len(P_log_prob_list) > 0:
            #             # for j in range(num_P):
            #             #     adv_n = P_advs_norm[i_b][j][st]
            #             #     batch_policy_loss[i_b].append(-P_log_prob_list[j] * adv_n)
            #             adv_n_list = P_advs_norm[i_b,:,st] # size num_P
            #             batch_policy_loss[i_b].append(torch.sum(-P_log_prob_list * adv_n_list))

                    # # A agents
                    # A_log_prob_list = self.batch_A_log_probs[i_b][st]
                    # if len(A_log_prob_list) > 0:
                    #     # for j in range(num_A):
                    #     #     adv_n = A_advs_norm[i_b][j][st]
                    #     #     batch_policy_loss[i_b].append(-A_log_prob_list[j] * adv_n)
                    #     adv_n_list = A_advs_norm[i_b,:,st] # size num_A
                    #     batch_policy_loss[i_b].append(torch.sum(-A_log_prob_list * adv_n_list))
                    #
                    # # calculate critic (value) loss
                    # batch_critic_loss[i_b].append(
                    #     F.mse_loss(self.batch_saved_critics[i_b][st],
                    #                critic_target[i_b][st].reshape(1,1)))
                    # # self.critic_loss_tracker.append(batch_critic_loss[i_b][-1])

            # if len(batch_policy_loss[i_b]) > 0:
            #     # batch_total_loss.append(
            #     #     torch.stack(batch_policy_loss[i_b]).sum() * 100 +
            #     #     torch.stack(batch_critic_loss[i_b]).sum())
            #     batch_total_policy_loss.append(torch.stack(batch_policy_loss[i_b]).sum())
            #     batch_total_critic_loss.append(torch.stack(batch_critic_loss[i_b]).sum())


        # reset gradients
        self.optimizer.zero_grad()

        # sum up over all batches
        total_policy_loss = torch.stack(batch_total_policy_loss).sum() / batch_size
        total_critic_loss = torch.stack(batch_total_critic_loss).sum() / batch_size

        total_loss = total_policy_loss * 50 + total_critic_loss

        loss_np = {'total': total_loss.data.cpu().numpy(),
                   'policy': total_policy_loss.data.cpu().numpy(),
                   'critic': total_critic_loss.data.cpu().numpy()}

        # perform backprop
        total_loss.backward()

        # perform Gradient Clipping-by-norm
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # self.optimizer.step()

        # check if models update their weight
        #　print('sum of graph weights', self.model.layer1.heads[0].attn_fc_p2a.weight.sum().item())

        # reset rewards and action buffer
        for i_b in range(batch_size):
            del self.batch_rewards[i_b][:]
            del self.batch_P_log_probs[i_b][:]
            del self.batch_A_log_probs[i_b][:]
            del self.batch_saved_critics[i_b][:]
            del self.batch_entrpies_p[i_b][:]
            del self.batch_entrpies_a[i_b][:]

        return loss_np

    '''
    Generalized Advantage Estimation
    '''
    def batch_GAE(self, i_b, agent_idx, Normalize = False):
        returns = []
        adv = []
        gae = 0.0

        for i in reversed(range(len(self.batch_rewards[i_b]))):
            if i == len(self.batch_rewards[i_b]) - 1:
                nextvalue = 0.0
                currentvalue = self.batch_saved_critics[i_b][i].item()
            else:
                nextvalue = self.batch_saved_critics[i_b][i+1].item()
                currentvalue = self.batch_saved_critics[i_b][i].item()

            delta = self.batch_rewards[i_b][i][agent_idx] + self.gamma * nextvalue - currentvalue
            gae = delta + self.gamma * self.lmbda * gae

            adv.insert(0, gae)
            returns.insert(0, gae + currentvalue)

        adv = torch.tensor(adv).to(self.device)
        returns = torch.tensor(returns).to(self.device)

        if Normalize:
            eps = np.finfo(np.float32).eps.item()
            adv = (adv - adv.mean()) / (adv.std() + eps)

        return returns, adv

    '''
    Vanilla A2C for advatage computation
        Advtange = sum_r - critic_prediction
    '''
    def batch_r_a(self, i_b, agent_idx):
        returns = [] # list to save the true values
        advs = []
        R = 0.0

        for i in reversed(range(len(self.batch_rewards[i_b]))):
            R = self.batch_rewards[i_b][i][agent_idx] + self.gamma * R
            returns.insert(0, R)

            adv_tmp = R - self.batch_saved_critics[i_b][i].item()
            advs.insert(0, adv_tmp)

        advs = torch.tensor(advs).to(self.device)
        returns = torch.tensor(returns).to(self.device)

        return returns, advs

    '''
    Adjust learning rate using lr_scheduler
    '''
    def adjust_lr(self, metrics=0.0):
        self.lr_scheduler.step()

    '''
    Batch version, use GAE by default, per-class critic version
    '''
    def batch_finish_per_class(self, batch_size, sim_time, num_P=2, num_A=1, use_GAE = True):
        if not self.per_class_critic:
            print('Error, wrong function called. This is for per-class critic')
            return -1.0

        batch_total_policy_loss = []
        batch_total_critic_loss = []

        # pad episodes with early termination, N x total_number x Time
        # using 0
        num_agents = num_P + num_A
        batch_returns = torch.zeros(batch_size, num_agents, sim_time).to(self.device)
        batch_advs = torch.zeros(batch_size, num_agents, sim_time).to(self.device)

        # 1. compute total reward and advantage of each episode, per agent
        # first num_P are P agents, next num_A are A agents
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            for j in range(num_agents):
                if use_GAE:
                    batch_returns[i_b][j][:r_size], batch_advs[i_b][j][:r_size] = self.batch_GAE_per_class(i_b,j,num_P,num_A)
                else:
                    batch_returns[i_b][j][:r_size], batch_advs[i_b][j][:r_size] = self.batch_r_a_per_class(i_b,j,num_P,num_A)

        P_advs = batch_advs[:,:num_P,:] # N x num_P x Time
        A_advs = batch_advs[:,num_P:,:] # N x num_A x Time

        P_critic_target = torch.mean(batch_returns[:,:num_P,:], dim=1) # N x Time
        A_critic_target = torch.mean(batch_returns[:,num_P:,:], dim=1) # N x Time

        # 2. normalize advantages within a batch
        eps = np.finfo(np.float32).eps.item()
        P_adv_mean = P_advs.mean()
        P_adv_std = P_advs.std()
        P_advs_norm = (P_advs - P_adv_mean) / (P_adv_std + eps)

        A_adv_mean = A_advs.mean()
        A_adv_std = A_advs.std()
        A_advs_norm = (A_advs - A_adv_mean) / (A_adv_std + eps)

        # 3. calculate loss for each episode in the batch
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            # num_P * r_size
            P_log_prob_list = torch.stack(self.batch_P_log_probs[i_b], dim=1)
            P_adv_n = P_advs_norm[i_b,:,:r_size]  # num_P x r_size

            ################## change here for the other way of adding entropy bonuses #####################################################
            # batch_total_policy_loss.append(torch.sum(-P_log_prob_list * P_adv_n) - 0.01*self.entropies_p)
            # self.batch_entrpies_p[i_b].append(self.entropies_p)
            batch_total_policy_loss.append(torch.sum(-P_log_prob_list * P_adv_n))

            A_log_prob_list = torch.stack(self.batch_A_log_probs[i_b], dim=1)
            A_adv_n = A_advs_norm[i_b,:,:r_size]

            ################## change here for the other way of adding entropy bonuses #####################################################
            # batch_total_policy_loss.append(torch.sum(-A_log_prob_list * A_adv_n) - 0.01*self.entropies_a)
            # self.batch_entrpies_a[i_b].append(self.entropies_a)
            batch_total_policy_loss.append(torch.sum(-A_log_prob_list * A_adv_n))

            # critic size [r_size]
            P_critic_list = torch.stack(self.batch_P_critics[i_b]).squeeze()
            A_critic_list = torch.stack(self.batch_A_critics[i_b]).squeeze()

            batch_total_critic_loss.append(
                F.mse_loss(P_critic_list, P_critic_target[i_b][:r_size]))

            batch_total_critic_loss.append(
                F.mse_loss(A_critic_list, A_critic_target[i_b][:r_size]))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up over all batches
        total_policy_loss = torch.stack(batch_total_policy_loss).sum() / batch_size
        total_critic_loss = torch.stack(batch_total_critic_loss).sum() / batch_size

        ################## change here for the other way of adding entropy bonuses ###############################################################
        # total_loss = total_policy_loss * 50 + total_critic_loss

        stacked_ent_p = []
        stacked_ent_a = []
        for jj in range(0, batch_size):
            stacked_ent_p.append(torch.stack(self.batch_entrpies_p[jj]).sum())
            stacked_ent_a.append(torch.stack(self.batch_entrpies_a[jj]).sum())

        total_loss = total_policy_loss * 50 + total_critic_loss - 0.05*(torch.stack(stacked_ent_p).mean() + torch.stack(stacked_ent_a).mean())

        loss_np = {'total': total_loss.data.cpu().numpy(),
                   'policy': total_policy_loss.data.cpu().numpy(),
                   'critic': total_critic_loss.data.cpu().numpy()}

        # perform backprop
        total_loss.backward()

        # for x in self.model.parameters():
        #     print('grad', x.grad)

        # perform Gradient Clipping-by-norm
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # self.optimizer.step()

        # reset rewards and action buffer
        for i_b in range(batch_size):
            del self.batch_rewards[i_b][:]
            del self.batch_P_log_probs[i_b][:]
            del self.batch_A_log_probs[i_b][:]
            del self.batch_P_critics[i_b][:]
            del self.batch_A_critics[i_b][:]
            del self.batch_entrpies_p[i_b][:]
            del self.batch_entrpies_a[i_b][:]

        return loss_np

    '''
    Generalized Advantage Estimation, per-class critic
    '''
    def batch_GAE_per_class(self, i_b, agent_idx, num_P, num_A, Normalize = False):
        returns = []
        adv = []
        gae = 0.0

        for i in reversed(range(len(self.batch_rewards[i_b]))):
            if i == len(self.batch_rewards[i_b]) - 1:
                nextvalue = 0.0
                if agent_idx < num_P:
                    currentvalue = self.batch_P_critics[i_b][i].item()
                else:
                    currentvalue = self.batch_A_critics[i_b][i].item()
            else:
                if agent_idx < num_P:
                    nextvalue = self.batch_P_critics[i_b][i+1].item()
                    currentvalue = self.batch_P_critics[i_b][i].item()
                else:
                    nextvalue = self.batch_A_critics[i_b][i+1].item()
                    currentvalue = self.batch_A_critics[i_b][i].item()

            delta = self.batch_rewards[i_b][i][agent_idx] + self.gamma * nextvalue - currentvalue
            gae = delta + self.gamma * self.lmbda * gae

            adv.insert(0, gae)
            returns.insert(0, gae + currentvalue)

        adv = torch.tensor(adv).to(self.device)
        returns = torch.tensor(returns).to(self.device)

        if Normalize:
            eps = np.finfo(np.float32).eps.item()
            adv = (adv - adv.mean()) / (adv.std() + eps)

        return returns, adv

    '''
    Vanilla A2C for advatage computation, per-class critic
        Advtange = sum_r - critic_prediction
    '''
    def batch_r_a_per_class(self, i_b, agent_idx, num_P, num_A):
        returns = [] # list to save the true values
        advs = []
        R = 0.0

        for i in reversed(range(len(self.batch_rewards[i_b]))):
            R = self.batch_rewards[i_b][i][agent_idx] + self.gamma * R
            returns.insert(0, R)

            if agent_idx < num_P:
                adv_tmp = R - self.batch_P_critics[i_b][i].item()
            else:
                adv_tmp = R - self.batch_A_critics[i_b][i].item()

            advs.insert(0, adv_tmp)

        advs = torch.tensor(advs).to(self.device)
        returns = torch.tensor(returns).to(self.device)

        return returns, advs

    '''
    Batch version, use GAE by default, per-agent critic version
    '''

    def batch_finish_per_agent(self, batch_size, sim_time, num_P=2, num_A=1, use_GAE=True):
        if not self.per_agent_critic:
            print('Error, wrong function called. This is for per-agent critic')
            return -1.0

        batch_total_policy_loss = []
        batch_total_critic_loss = []

        # pad episodes with early termination, N x total_number x Time
        # using 0
        num_agents = num_P + num_A
        batch_returns = torch.zeros(batch_size, num_agents, sim_time).to(self.device)
        batch_advs = torch.zeros(batch_size, num_agents, sim_time).to(self.device)

        # 1. compute total reward and advantage of each episode, per agent
        # first num_P are P agents, next num_A are A agents
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            for j in range(num_agents):
                if use_GAE:
                    batch_returns[i_b][j][:r_size], batch_advs[i_b][j][:r_size] = self.batch_GAE_per_agent(i_b, j,
                                                                                                           num_P, num_A)
                else:
                    batch_returns[i_b][j][:r_size], batch_advs[i_b][j][:r_size] = self.batch_r_a_per_agent(i_b, j,
                                                                                                           num_P, num_A)

        P_advs = batch_advs[:, :num_P, :]  # N x num_P x Time
        A_advs = batch_advs[:, num_P:, :]  # N x num_A x Time

        P_critic_target = batch_returns[:, :num_P, :]  # N x num_P x Time
        A_critic_target = batch_returns[:, num_P:, :]  # N x num_A x Time

        # 2. normalize advantages within a batch
        eps = np.finfo(np.float32).eps.item()
        P_adv_mean = P_advs.mean()
        P_adv_std = P_advs.std()
        P_advs_norm = (P_advs - P_adv_mean) / (P_adv_std + eps)

        A_adv_mean = A_advs.mean()
        A_adv_std = A_advs.std()
        A_advs_norm = (A_advs - A_adv_mean) / (A_adv_std + eps)

        # 3. calculate loss for each episode in the batch
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            # num_P * r_size
            P_log_prob_list = torch.stack(self.batch_P_log_probs[i_b], dim=1)
            P_adv_n = P_advs_norm[i_b, :, :r_size]  # num_P x r_size

            batch_total_policy_loss.append(torch.sum(-P_log_prob_list * P_adv_n))

            A_log_prob_list = torch.stack(self.batch_A_log_probs[i_b], dim=1)
            A_adv_n = A_advs_norm[i_b, :, :r_size]

            batch_total_policy_loss.append(torch.sum(-A_log_prob_list * A_adv_n))

            # critic shape: num_P x r_size
            P_critic_list = torch.cat(self.batch_P_critics[i_b], dim=1)
            A_critic_list = torch.cat(self.batch_A_critics[i_b], dim=1)

            batch_total_critic_loss.append(
                F.mse_loss(P_critic_list, P_critic_target[i_b, :, :r_size]))

            batch_total_critic_loss.append(
                F.mse_loss(A_critic_list, A_critic_target[i_b, :, :r_size]))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up over all batches
        total_policy_loss = torch.stack(batch_total_policy_loss).sum() / batch_size
        total_critic_loss = torch.stack(batch_total_critic_loss).sum() / batch_size

        total_loss = total_policy_loss * 50 + total_critic_loss

        loss_np = {'total': total_loss.data.cpu().numpy(),
                   'policy': total_policy_loss.data.cpu().numpy(),
                   'critic': total_critic_loss.data.cpu().numpy()}

        # perform backprop
        total_loss.backward()

        # perform Gradient Clipping-by-norm
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # self.optimizer.step()

        # reset rewards and action buffer
        for i_b in range(batch_size):
            del self.batch_rewards[i_b][:]
            del self.batch_P_log_probs[i_b][:]
            del self.batch_A_log_probs[i_b][:]
            del self.batch_P_critics[i_b][:]
            del self.batch_A_critics[i_b][:]

        return loss_np

    '''
    Generalized Advantage Estimation, per-agent critic
    '''

    def batch_GAE_per_agent(self, i_b, agent_idx, num_P, num_A, Normalize=False):
        returns = []
        adv = []
        gae = 0.0

        for i in reversed(range(len(self.batch_rewards[i_b]))):
            if i == len(self.batch_rewards[i_b]) - 1:
                nextvalue = 0.0
                if agent_idx < num_P:
                    currentvalue = self.batch_P_critics[i_b][i][agent_idx].item()
                else:
                    currentvalue = self.batch_A_critics[i_b][i][agent_idx - num_P].item()
            else:
                if agent_idx < num_P:
                    nextvalue = self.batch_P_critics[i_b][i + 1][agent_idx].item()
                    currentvalue = self.batch_P_critics[i_b][i][agent_idx].item()
                else:
                    nextvalue = self.batch_A_critics[i_b][i + 1][agent_idx - num_P].item()
                    currentvalue = self.batch_A_critics[i_b][i][agent_idx - num_P].item()

            delta = self.batch_rewards[i_b][i][agent_idx] + self.gamma * nextvalue - currentvalue
            gae = delta + self.gamma * self.lmbda * gae

            adv.insert(0, gae)
            returns.insert(0, gae + currentvalue)

        adv = torch.tensor(adv).to(self.device)
        returns = torch.tensor(returns).to(self.device)

        if Normalize:
            eps = np.finfo(np.float32).eps.item()
            adv = (adv - adv.mean()) / (adv.std() + eps)

        return returns, adv

    '''
    Vanilla A2C for advatage computation, per-agent critic
        Advtange = sum_r - critic_prediction
    '''

    def batch_r_a_per_agent(self, i_b, agent_idx, num_P, num_A):
        returns = []  # list to save the true values
        advs = []
        R = 0.0

        for i in reversed(range(len(self.batch_rewards[i_b]))):
            R = self.batch_rewards[i_b][i][agent_idx] + self.gamma * R
            returns.insert(0, R)

            if agent_idx < num_P:
                adv_tmp = R - self.batch_P_critics[i_b][i][agent_idx].item()
            else:
                adv_tmp = R - self.batch_A_critics[i_b][i][agent_idx - num_P].item()

            advs.insert(0, adv_tmp)

        advs = torch.tensor(advs).to(self.device)
        returns = torch.tensor(returns).to(self.device)

        return returns, advs
