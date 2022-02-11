# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 23:06:14 2020

@author: pheno

Hetero graph conv layer for UAV coordination
    0. real-valued message
    1. with state node as centralized Critic
        HeteroGATLayerReal, MultiHeteroGATLayerReal
    2. without state node version
        HeteroGATReal, MultiHeteroGATReal
    3. Optimize attention coefficient computation
        however, no or little speed up is observed after implementing this
    4. Ultilize DGL's multi-head trick (from homograph) to get rid of the for loop
        e.g. now num_heads = 4 with hid_dim = 8 has the same speed as num_heads = 1
        with hid_dim = 32
Version: 2021-4-30
"""

from pdb import set_trace
import torch
import torch.nn as nn
# import torch.nn.functional as F
import math

import dgl.function as fn
from dgl.ops import edge_softmax
import numpy as np
# from dgl.nn.functional import edge_softmax
import time
from utils import real_comm_loss, get_comm_loss_bitless_k

import sys
sys.path.insert(0, '/home/eseraj3/HetNet_MARL_Communication/test/IC3Net/utils.py')
from utils import real_comm_loss

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

# in_dim: dict of input feature dimension for each node
# out_dim: dict of output feature dimension for each node
class HeteroGATLayerReal(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, l_alpha=0.2, use_relu=True, action_have_vision=False):
        super(HeteroGATLayerReal, self).__init__()
        self._num_heads  = num_heads
        self._in_dim = in_dim
        self._out_dim = out_dim
        self.action_have_vision = action_have_vision

        if action_have_vision:
            self.fc = nn.ModuleDict({
            'P': nn.Linear(in_dim['P'], out_dim['P'] * num_heads),
            'A': nn.Linear(in_dim['A'] + in_dim['state'] * in_dim['obs_squares'], out_dim['A'] * num_heads),
            'p2p': nn.Linear(in_dim['P'], out_dim['P'] * num_heads),
            'p2a': nn.Linear(in_dim['P'], out_dim['A'] * num_heads),
            'a2p': nn.Linear(in_dim['A'] + in_dim['state'] * in_dim['obs_squares'], out_dim['P'] * num_heads),
            'a2a': nn.Linear(in_dim['A'] + in_dim['state'] * in_dim['obs_squares'], out_dim['A'] * num_heads),
            'p2s': nn.Linear(in_dim['P'], out_dim['state'] * num_heads),
            'a2s': nn.Linear(in_dim['A'] + in_dim['state'] * in_dim['obs_squares'], out_dim['state'] * num_heads),
            'in': nn.Linear(in_dim['state'], out_dim['state'] * num_heads)
            })
        else:
            self.fc = nn.ModuleDict({
            'P': nn.Linear(in_dim['P'], out_dim['P'] * num_heads),
            'A': nn.Linear(in_dim['A'], out_dim['A'] * num_heads),
            'p2p': nn.Linear(in_dim['P'], out_dim['P'] * num_heads),
            'p2a': nn.Linear(in_dim['P'], out_dim['A'] * num_heads),
            'a2p': nn.Linear(in_dim['A'], out_dim['P'] * num_heads),
            'a2a': nn.Linear(in_dim['A'], out_dim['A'] * num_heads),
            'p2s': nn.Linear(in_dim['P'], out_dim['state'] * num_heads),
            'a2s': nn.Linear(in_dim['A'], out_dim['state'] * num_heads),
            'in': nn.Linear(in_dim['state'], out_dim['state'] * num_heads)
            })

        self.leaky_relu = nn.LeakyReLU(negative_slope=l_alpha)
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()

        # attention coefficients
        self.p2p_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
        self.p2p_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
        self.p2a_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
        self.p2a_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
        self.a2p_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
        self.a2p_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
        self.a2a_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
        self.a2a_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))


        # state node
        # self.attn_fc_p2s = nn.Linear(2 * out_dim['state'], 1, bias=False)
        # self.attn_fc_a2s = nn.Linear(2 * out_dim['state'], 1, bias=False)
        self.p2s_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.p2s_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.a2s_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.a2s_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters
        References
            1 https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html
            2 https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
        """
        gain = nn.init.calculate_gain('relu')

        # # fc weights
        # nn.init.xavier_normal_(self.fc['P'].weight, gain=gain)
        # nn.init.xavier_normal_(self.fc['A'].weight, gain=gain)
        # nn.init.xavier_normal_(self.fc['p2p'].weight, gain=gain)
        # nn.init.xavier_normal_(self.fc['p2a'].weight, gain=gain)
        # nn.init.xavier_normal_(self.fc['a2p'].weight, gain=gain)
        # nn.init.xavier_normal_(self.fc['a2a'].weight, gain=gain)
        # nn.init.xavier_normal_(self.fc['p2s'].weight, gain=gain)
        # nn.init.xavier_normal_(self.fc['a2s'].weight, gain=gain)
        # nn.init.xavier_normal_(self.fc['in'].weight, gain=gain)

        # # fc biases
        # # https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/graphconv.html#GraphConv
        # nn.init.zeros_(self.fc['P'].bias)
        # nn.init.zeros_(self.fc['A'].bias)
        # nn.init.zeros_(self.fc['p2p'].bias)
        # nn.init.zeros_(self.fc['p2a'].bias)
        # nn.init.zeros_(self.fc['a2p'].bias)
        # nn.init.zeros_(self.fc['a2a'].bias)
        # nn.init.zeros_(self.fc['p2s'].bias)
        # nn.init.zeros_(self.fc['a2s'].bias)
        # nn.init.zeros_(self.fc['in'].bias)

        # attention
        nn.init.xavier_normal_(self.p2p_src, gain=gain)
        nn.init.xavier_normal_(self.p2p_dst, gain=gain)
        nn.init.xavier_normal_(self.p2a_src, gain=gain)
        nn.init.xavier_normal_(self.p2a_dst, gain=gain)
        nn.init.xavier_normal_(self.a2p_src, gain=gain)
        nn.init.xavier_normal_(self.a2p_dst, gain=gain)
        nn.init.xavier_normal_(self.a2a_src, gain=gain)
        nn.init.xavier_normal_(self.a2a_dst, gain=gain)

        nn.init.xavier_normal_(self.p2s_src, gain=gain)
        nn.init.xavier_normal_(self.p2s_dst, gain=gain)
        nn.init.xavier_normal_(self.a2s_src, gain=gain)
        nn.init.xavier_normal_(self.a2s_dst, gain=gain)

    def forward(self, g, feat_dict):
        '''
        From hi to Whi
        '''
        # feature of P
        Whp = self.fc['P'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['P'])
        g.nodes['P'].data['Wh_P'] = Whp

        # feature of A
        Wha = self.fc['A'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['A'])
        g.nodes['A'].data['Wh_A'] = Wha

        '''
        Feature transform for each edge type (communication channel)
        '''
        # p2p
        Whp2p = self.fc['p2p'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['P'])
        g.nodes['P'].data['Wh_p2p'] = Whp2p

        # p2a
        Whp2a = self.fc['p2a'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['A'])
        g.nodes['P'].data['Wh_p2a'] = Whp2a

        # a2p
        Wha2p = self.fc['a2p'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['P'])
        g.nodes['A'].data['Wh_a2p'] = Wha2p

        # a2a
        Wha2a = self.fc['a2a'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['A'])
        g.nodes['A'].data['Wh_a2a'] = Wha2a

        # for state-related edges
        Whp2s = self.fc['p2s'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['P'].data['Wh_p2s'] = Whp2s

        Wha2s = self.fc['a2s'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['A'].data['Wh_a2s'] = Wha2s

        Whin = self.fc['in'](feat_dict['state'].double()).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['state'].data['Wh_in'] = Whin

        '''
        Attention computation on subgraphs
        '''
        # p2p
        if g['p2p'].number_of_edges() > 0:
            Attn_src_p2p = (Whp2p * self.p2p_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_p2p = (Whp * self.p2p_dst).sum(dim=-1).unsqueeze(-1)
            g['p2p'].srcdata.update({'Attn_src_p2p': Attn_src_p2p})
            g['p2p'].dstdata.update({'Attn_dst_p2p': Attn_dst_p2p})

            g['p2p'].apply_edges(fn.u_add_v('Attn_src_p2p', 'Attn_dst_p2p', 'e_p2p'))
            e_p2p = self.leaky_relu(g['p2p'].edata.pop('e_p2p'))

            # compute softmax
            g['p2p'].edata['a_p2p'] = edge_softmax(g['p2p'], e_p2p)
            # message passing
            g['p2p'].update_all(fn.u_mul_e('Wh_p2p', 'a_p2p', 'm_p2p'),
                                fn.sum('m_p2p', 'ft_p2p'))

        # p2a
        if g['p2a'].number_of_edges() > 0:
            Attn_src_p2a = (Whp2a * self.p2a_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_p2a = (Wha * self.p2a_dst).sum(dim=-1).unsqueeze(-1)
            # both works
            #g.nodes['P'].data['Attn_src_p2a'] = Attn_src_p2a
            g['p2a'].srcdata.update({'Attn_src_p2a': Attn_src_p2a})
            # g.nodes['A'].data['Attn_dst_p2a'] = Attn_dst_p2a
            g['p2a'].dstdata.update({'Attn_dst_p2a': Attn_dst_p2a})
            '''
            Note:
                g.dstdata['Attn_dst_p2a']['A'] gives the data tensor
                but g.dstdata['A'] gives {}
                so the first key right after dstdata is the feature key
                    not the node type key
            '''
            g['p2a'].apply_edges(fn.u_add_v('Attn_src_p2a', 'Attn_dst_p2a', 'e_p2a'))
            # g['p2a'].edata['e_p2a'] gives the tensor
            e_p2a = self.leaky_relu(g['p2a'].edata.pop('e_p2a'))

            # compute softmax
            g['p2a'].edata['a_p2a'] = edge_softmax(g['p2a'], e_p2a)
            # message passing
            g['p2a'].update_all(fn.u_mul_e('Wh_p2a', 'a_p2a', 'm_p2a'),
                                fn.sum('m_p2a', 'ft_p2a'))
            # results =  g.nodes['A'].data['ft_p2a']

        # a2p
        if g['a2p'].number_of_edges() > 0:
            Attn_src_a2p = (Wha2p * self.a2p_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_a2p = (Whp * self.a2p_dst).sum(dim=-1).unsqueeze(-1)
            g['a2p'].srcdata.update({'Attn_src_a2p': Attn_src_a2p})
            g['a2p'].dstdata.update({'Attn_dst_a2p': Attn_dst_a2p})

            g['a2p'].apply_edges(fn.u_add_v('Attn_src_a2p', 'Attn_dst_a2p', 'e_a2p'))
            e_a2p = self.leaky_relu(g['a2p'].edata.pop('e_a2p'))

            # compute softmax
            g['a2p'].edata['a_a2p'] = edge_softmax(g['a2p'], e_a2p)
            # message passing
            g['a2p'].update_all(fn.u_mul_e('Wh_a2p', 'a_a2p', 'm_a2p'),
                                fn.sum('m_a2p', 'ft_a2p'))

        # a2a
        if g['a2a'].number_of_edges() > 0:
            Attn_src_a2a = (Wha2a * self.a2a_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_a2a = (Wha * self.a2a_dst).sum(dim=-1).unsqueeze(-1)
            g['a2a'].srcdata.update({'Attn_src_a2a': Attn_src_a2a})
            g['a2a'].dstdata.update({'Attn_dst_a2a': Attn_dst_a2a})

            g['a2a'].apply_edges(fn.u_add_v('Attn_src_a2a', 'Attn_dst_a2a', 'e_a2a'))
            e_a2a = self.leaky_relu(g['a2a'].edata.pop('e_a2a'))

            # compute softmax
            g['a2a'].edata['a_a2a'] = edge_softmax(g['a2a'], e_a2a)
            # message passing
            g['a2a'].update_all(fn.u_mul_e('Wh_a2a', 'a_a2a', 'm_a2a'),
                                fn.sum('m_a2a', 'ft_a2a'))

        # p2s
        # if g['p2s'].number_of_edges() > 0: # this is always true
        Attn_src_p2s = (Whp2s * self.p2s_src).sum(dim=-1).unsqueeze(-1)
        Attn_dst_p2s = (Whin * self.p2s_dst).sum(dim=-1).unsqueeze(-1)
        g['p2s'].srcdata.update({'Attn_src_p2s': Attn_src_p2s})
        g['p2s'].dstdata.update({'Attn_dst_p2s': Attn_dst_p2s})

        g['p2s'].apply_edges(fn.u_add_v('Attn_src_p2s', 'Attn_dst_p2s', 'e_p2s'))
        e_p2s = self.leaky_relu(g['p2s'].edata.pop('e_p2s'))

        # compute softmax
        g['p2s'].edata['a_p2s'] = edge_softmax(g['p2s'], e_p2s)
        # message passing
        g['p2s'].update_all(fn.u_mul_e('Wh_p2s', 'a_p2s', 'm_p2s'),
                            fn.sum('m_p2s', 'ft_p2s'))

        # a2s
        Attn_src_a2s = (Wha2s * self.a2s_src).sum(dim=-1).unsqueeze(-1)
        Attn_dst_a2s = (Whin * self.a2s_dst).sum(dim=-1).unsqueeze(-1)
        g['a2s'].srcdata.update({'Attn_src_a2s': Attn_src_a2s})
        g['a2s'].dstdata.update({'Attn_dst_a2s': Attn_dst_a2s})

        g['a2s'].apply_edges(fn.u_add_v('Attn_src_a2s', 'Attn_dst_a2s', 'e_a2s'))
        e_a2s = self.leaky_relu(g['a2s'].edata.pop('e_a2s'))

        # compute softmax
        g['a2s'].edata['a_a2s'] = edge_softmax(g['a2s'], e_a2s)
        # message passing
        g['a2s'].update_all(fn.u_mul_e('Wh_a2s', 'a_a2s', 'm_a2s'),
                            fn.sum('m_a2s', 'ft_a2s'))

        '''
        Combine features from subgraphs
            Sum up to hi'
            Need to check if subgraph is valid
        '''
        # new feature of P
        Whp_new = g.nodes['P'].data['Wh_P'].clone()

        if g['p2p'].number_of_edges() > 0:
            Whp_new += g.nodes['P'].data['ft_p2p']
        if g['a2p'].number_of_edges() > 0:
            Whp_new += g.nodes['P'].data['ft_a2p']

        g.nodes['P'].data['h'] = Whp_new

        # new feature of A
        Wha_new = g.nodes['A'].data['Wh_A'].clone()
        if g['p2a'].number_of_edges() > 0:
            Wha_new += g.nodes['A'].data['ft_p2a']
        if g['a2a'].number_of_edges() > 0:
            Wha_new += g.nodes['A'].data['ft_a2a']

        g.nodes['A'].data['h'] = Wha_new

        # new feature of state
        Whstate_new = g.nodes['state'].data['Wh_in'] + \
            g.nodes['state'].data['ft_p2s'] + \
                g.nodes['state'].data['ft_a2s']

        g.nodes['state'].data['h'] = Whstate_new

        # deal with relu activation
        if self.use_relu:
            return {ntype : self.relu(g.nodes[ntype].data['h']) for ntype in g.ntypes}
        else:
            return {ntype : g.nodes[ntype].data['h'] for ntype in g.ntypes}

# merge = 'cat' or 'avg'
class MultiHeteroGATLayerReal(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, action_have_vision=False, merge='cat'):
        super(MultiHeteroGATLayerReal, self).__init__()

        self._num_heads = num_heads
        self._merge = merge

        if self._merge == 'cat':
            self.gat_conv = HeteroGATLayerReal(in_dim, out_dim, num_heads, action_have_vision=action_have_vision)
        else:
            self.gat_conv = HeteroGATLayerReal(in_dim, out_dim, num_heads, use_relu=False, action_have_vision=action_have_vision)

    def forward(self, g, feat_dict):
        tmp = self.gat_conv(g, feat_dict)
        results = {}

        if self._merge == 'cat':
            # concat on the output feature dimension (dim=1)
            for ntype in g.ntypes:
                results[ntype] = torch.flatten(tmp[ntype], 1)
        else:
            # merge using average
            for ntype in g.ntypes:
                results[ntype] = torch.mean(tmp[ntype], 1)

        return results

# in_dim: dict of input feature dimension for each node
# out_dim: dict of output feature dimension for each node
class HeteroGATReal(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, l_alpha = 0.2, use_relu = True):
        super(HeteroGATReal, self).__init__()
        self._num_heads  = num_heads
        self._in_dim = in_dim
        self._out_dim = out_dim

        self.fc = nn.ModuleDict({
            'P': nn.Linear(in_dim['P'], out_dim['P'] * num_heads),
            'A': nn.Linear(in_dim['A'], out_dim['A'] * num_heads),
            'p2p': nn.Linear(in_dim['P'], out_dim['P'] * num_heads),
            'p2a': nn.Linear(in_dim['P'], out_dim['A'] * num_heads),
            'a2p': nn.Linear(in_dim['A'], out_dim['P'] * num_heads),
            'a2a': nn.Linear(in_dim['A'], out_dim['A'] * num_heads),
            })

        self.leaky_relu = nn.LeakyReLU(negative_slope = l_alpha)
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()

        # attention coefficients
        self.p2p_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
        self.p2p_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
        self.p2a_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
        self.p2a_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
        self.a2p_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
        self.a2p_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
        self.a2a_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
        self.a2a_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters
        References
            1 https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html
            2 https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
        """
        gain = nn.init.calculate_gain('relu')

        # # fc weights
        # nn.init.xavier_normal_(self.fc['P'].weight, gain=gain)
        # nn.init.xavier_normal_(self.fc['A'].weight, gain=gain)
        # nn.init.xavier_normal_(self.fc['p2p'].weight, gain=gain)
        # nn.init.xavier_normal_(self.fc['p2a'].weight, gain=gain)
        # nn.init.xavier_normal_(self.fc['a2p'].weight, gain=gain)
        # nn.init.xavier_normal_(self.fc['a2a'].weight, gain=gain)

        # # fc biases
        # # https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/graphconv.html#GraphConv
        # nn.init.zeros_(self.fc['P'].bias)
        # nn.init.zeros_(self.fc['A'].bias)
        # nn.init.zeros_(self.fc['p2p'].bias)
        # nn.init.zeros_(self.fc['p2a'].bias)
        # nn.init.zeros_(self.fc['a2p'].bias)
        # nn.init.zeros_(self.fc['a2a'].bias)

        # attention
        nn.init.xavier_normal_(self.p2p_src, gain=gain)
        nn.init.xavier_normal_(self.p2p_dst, gain=gain)
        nn.init.xavier_normal_(self.p2a_src, gain=gain)
        nn.init.xavier_normal_(self.p2a_dst, gain=gain)
        nn.init.xavier_normal_(self.a2p_src, gain=gain)
        nn.init.xavier_normal_(self.a2p_dst, gain=gain)
        nn.init.xavier_normal_(self.a2a_src, gain=gain)
        nn.init.xavier_normal_(self.a2a_dst, gain=gain)

    def forward(self, g, feat_dict):
        '''
        From hi to Whi
        '''
        # feature of P
        Whp = self.fc['P'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['P'])
        g.nodes['P'].data['Wh_P'] = Whp

        # feature of A
        Wha = self.fc['A'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['A'])
        g.nodes['A'].data['Wh_A'] = Wha

        '''
        Feature transform for each edge type (communication channel)
        '''
        # p2p
        Whp2p = self.fc['p2p'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['P'])
        g.nodes['P'].data['Wh_p2p'] = Whp2p

        # p2a
        Whp2a = self.fc['p2a'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['A'])
        g.nodes['P'].data['Wh_p2a'] = Whp2a

        # a2p
        Wha2p = self.fc['a2p'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['P'])
        g.nodes['A'].data['Wh_a2p'] = Wha2p

        # a2a
        Wha2a = self.fc['a2a'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['A'])
        g.nodes['A'].data['Wh_a2a'] = Wha2a

        '''
        Attention computation on subgraphs
        '''
        # p2p
        if g['p2p'].number_of_edges() > 0:
            Attn_src_p2p = (Whp2p * self.p2p_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_p2p = (Whp * self.p2p_dst).sum(dim=-1).unsqueeze(-1)
            g['p2p'].srcdata.update({'Attn_src_p2p': Attn_src_p2p})
            g['p2p'].dstdata.update({'Attn_dst_p2p': Attn_dst_p2p})

            g['p2p'].apply_edges(fn.u_add_v('Attn_src_p2p', 'Attn_dst_p2p', 'e_p2p'))
            e_p2p = self.leaky_relu(g['p2p'].edata.pop('e_p2p'))

            # compute softmax
            g['p2p'].edata['a_p2p'] = edge_softmax(g['p2p'], e_p2p)
            # message passing
            g['p2p'].update_all(fn.u_mul_e('Wh_p2p', 'a_p2p', 'm_p2p'),
                                fn.sum('m_p2p', 'ft_p2p'))

        # p2a
        if g['p2a'].number_of_edges() > 0:
            Attn_src_p2a = (Whp2a * self.p2a_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_p2a = (Wha * self.p2a_dst).sum(dim=-1).unsqueeze(-1)
            # both works
            #g.nodes['P'].data['Attn_src_p2a'] = Attn_src_p2a
            g['p2a'].srcdata.update({'Attn_src_p2a': Attn_src_p2a})
            # g.nodes['A'].data['Attn_dst_p2a'] = Attn_dst_p2a
            g['p2a'].dstdata.update({'Attn_dst_p2a': Attn_dst_p2a})
            '''
            Note:
                g.dstdata['Attn_dst_p2a']['A'] gives the data tensor
                but g.dstdata['A'] gives {}
                so the first key right after dstdata is the feature key
                    not the node type key
            '''
            g['p2a'].apply_edges(fn.u_add_v('Attn_src_p2a', 'Attn_dst_p2a', 'e_p2a'))
            # g['p2a'].edata['e_p2a'] gives the tensor
            e_p2a = self.leaky_relu(g['p2a'].edata.pop('e_p2a'))

            # compute softmax
            g['p2a'].edata['a_p2a'] = edge_softmax(g['p2a'], e_p2a)
            # message passing
            g['p2a'].update_all(fn.u_mul_e('Wh_p2a', 'a_p2a', 'm_p2a'),
                                fn.sum('m_p2a', 'ft_p2a'))
            # results =  g.nodes['A'].data['ft_p2a']

        # a2p
        if g['a2p'].number_of_edges() > 0:
            Attn_src_a2p = (Wha2p * self.a2p_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_a2p = (Whp * self.a2p_dst).sum(dim=-1).unsqueeze(-1)
            g['a2p'].srcdata.update({'Attn_src_a2p': Attn_src_a2p})
            g['a2p'].dstdata.update({'Attn_dst_a2p': Attn_dst_a2p})

            g['a2p'].apply_edges(fn.u_add_v('Attn_src_a2p', 'Attn_dst_a2p', 'e_a2p'))
            e_a2p = self.leaky_relu(g['a2p'].edata.pop('e_a2p'))

            # compute softmax
            g['a2p'].edata['a_a2p'] = edge_softmax(g['a2p'], e_a2p)
            # message passing
            g['a2p'].update_all(fn.u_mul_e('Wh_a2p', 'a_a2p', 'm_a2p'),
                                fn.sum('m_a2p', 'ft_a2p'))

        # a2a
        if g['a2a'].number_of_edges() > 0:
            Attn_src_a2a = (Wha2a * self.a2a_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_a2a = (Wha * self.a2a_dst).sum(dim=-1).unsqueeze(-1)
            g['a2a'].srcdata.update({'Attn_src_a2a': Attn_src_a2a})
            g['a2a'].dstdata.update({'Attn_dst_a2a': Attn_dst_a2a})

            g['a2a'].apply_edges(fn.u_add_v('Attn_src_a2a', 'Attn_dst_a2a', 'e_a2a'))
            e_a2a = self.leaky_relu(g['a2a'].edata.pop('e_a2a'))

            # compute softmax
            g['a2a'].edata['a_a2a'] = edge_softmax(g['a2a'], e_a2a)
            # message passing
            g['a2a'].update_all(fn.u_mul_e('Wh_a2a', 'a_a2a', 'm_a2a'),
                                fn.sum('m_a2a', 'ft_a2a'))

        '''
        Combine features from subgraphs
            Sum up to hi'
            Need to check if subgraph is valid
        '''
        # new feature of P
        Whp_new = g.nodes['P'].data['Wh_P'].clone()

        if g['p2p'].number_of_edges() > 0:
            Whp_new += g.nodes['P'].data['ft_p2p']
        if g['a2p'].number_of_edges() > 0:
            Whp_new += g.nodes['P'].data['ft_a2p']

        g.nodes['P'].data['h'] = Whp_new

        # new feature of A
        Wha_new = g.nodes['A'].data['Wh_A'].clone()
        if g['p2a'].number_of_edges() > 0:
            Wha_new += g.nodes['A'].data['ft_p2a']
        if g['a2a'].number_of_edges() > 0:
            Wha_new += g.nodes['A'].data['ft_a2a']

        g.nodes['A'].data['h'] = Wha_new

        # deal with relu activation
        if self.use_relu:
            return {ntype : self.relu(g.nodes[ntype].data['h']) for ntype in g.ntypes}
        else:
            return {ntype : g.nodes[ntype].data['h'] for ntype in g.ntypes}

# merge = 'cat' or 'avg'
class MultiHeteroGATReal(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeteroGATReal, self).__init__()

        self._num_heads = num_heads
        self._merge = merge

        if self._merge == 'cat':
            self.gat_conv = HeteroGATReal(in_dim, out_dim, num_heads)
        else:
            self.gat_conv = HeteroGATReal(in_dim, out_dim, num_heads, use_relu = False)

    def forward(self, g, feat_dict):
        tmp = self.gat_conv(g, feat_dict)
        results = {}

        if self._merge == 'cat':
            # concat on the output feature dimension (dim=1)
            for ntype in g.ntypes:
                results[ntype] = torch.flatten(tmp[ntype], 1)
        else:
            # merge using average
            for ntype in g.ntypes:
                results[ntype] = torch.mean(tmp[ntype], 1)

        return results

class HeteroGATLayerLossyReal(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, l_alpha=0.2, use_relu=True,
            comm_range_P=-1, comm_range_A=-1, min_comm_loss=0, max_comm_loss=0.3):
        super(HeteroGATLayerLossyReal, self).__init__()

        self._num_heads  = num_heads
        self._in_dim = in_dim
        self._out_dim = out_dim

        self.comm_range_P = comm_range_P
        self.comm_range_A = comm_range_A
        self.min_comm_loss = min_comm_loss
        self.max_comm_loss = max_comm_loss
        
        self.bitless_k_P = get_comm_loss_bitless_k(self.comm_range_P, self.max_comm_loss)
        self.bitless_k_A = get_comm_loss_bitless_k(self.comm_range_A, self.max_comm_loss)

        self.fc = nn.ModuleDict({
            'P': nn.Linear(in_dim['P'], out_dim['P'] * num_heads),
            'A': nn.Linear(in_dim['A'], out_dim['A'] * num_heads),
            'p2p': nn.Linear(in_dim['P'], out_dim['P'] * num_heads),
            'p2a': nn.Linear(in_dim['P'], out_dim['A'] * num_heads),
            'a2p': nn.Linear(in_dim['A'], out_dim['P'] * num_heads),
            'a2a': nn.Linear(in_dim['A'], out_dim['A'] * num_heads),
            'p2s': nn.Linear(in_dim['P'], out_dim['state'] * num_heads),
            'a2s': nn.Linear(in_dim['A'], out_dim['state'] * num_heads),
            'in': nn.Linear(in_dim['state'], out_dim['state'] * num_heads)
            })

        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=l_alpha)

        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()

        # attention coefficients
        self.p2p_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
        self.p2p_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
        self.p2a_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
        self.p2a_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
        self.a2p_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
        self.a2p_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
        self.a2a_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
        self.a2a_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))


        # state node
        # self.attn_fc_p2s = nn.Linear(2 * out_dim['state'], 1, bias=False)
        # self.attn_fc_a2s = nn.Linear(2 * out_dim['state'], 1, bias=False)
        self.p2s_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.p2s_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.a2s_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.a2s_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters
        References
            1 https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html
            2 https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
        """
        gain = nn.init.calculate_gain('relu')

        # attention
        nn.init.xavier_normal_(self.p2p_src, gain=gain)
        nn.init.xavier_normal_(self.p2p_dst, gain=gain)
        nn.init.xavier_normal_(self.p2a_src, gain=gain)
        nn.init.xavier_normal_(self.p2a_dst, gain=gain)
        nn.init.xavier_normal_(self.a2p_src, gain=gain)
        nn.init.xavier_normal_(self.a2p_dst, gain=gain)
        nn.init.xavier_normal_(self.a2a_src, gain=gain)
        nn.init.xavier_normal_(self.a2a_dst, gain=gain)

        nn.init.xavier_normal_(self.p2s_src, gain=gain)
        nn.init.xavier_normal_(self.p2s_dst, gain=gain)
        nn.init.xavier_normal_(self.a2s_src, gain=gain)
        nn.init.xavier_normal_(self.a2s_dst, gain=gain)

    def lossy_u_mul_e(self, msg_type):
        m_type = 'm_' + msg_type
        h_type = 'h_' + msg_type
        a_type = 'a_' + msg_type

        sender_type = msg_type[0].upper()
        receiver_type = msg_type[-1].upper()

        Wh_self = 'Wh_' + receiver_type

        if msg_type == 'p2p':
            src, dst = self.p2p_src, self.p2p_dst
        elif msg_type == 'p2a':
            src, dst = self.p2a_src, self.p2a_dst
        elif msg_type == 'a2p':
            src, dst = self.a2p_src, self.a2p_dst
        elif msg_type == 'a2a':
            src, dst = self.a2a_src, self.a2a_dst

        bitless_k = self.bitless_k_P if sender_type == 'P' else self.bitless_k_A

        def inner_lossy_u_mul_e(edges):
            dist = edges.data['dist']

            lossy_h = edges.src[h_type].clone()
            lossy_h = real_comm_loss(dist, lossy_h, bitless_k)

            lossy_Wh = self.fc[msg_type](lossy_h).view(-1, self._num_heads, self._out_dim[receiver_type])

            Attn_src = (lossy_Wh * src).sum(dim=-1).unsqueeze(-1)
            Attn_dst = (edges.dst[Wh_self] * dst).sum(dim=-1).unsqueeze(-1)

            lossy_attn = self.softmax(
                self.leaky_relu(Attn_src + Attn_dst))

            return { m_type : lossy_Wh * lossy_attn }

        return inner_lossy_u_mul_e

    def forward(self, g, feat_dict):
        '''
        From hi to Whi
        '''
        # feature of P
        Whp = self.fc['P'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['P'])
        g.nodes['P'].data['Wh_P'] = Whp

        # feature of A
        Wha = self.fc['A'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['A'])
        g.nodes['A'].data['Wh_A'] = Wha

        '''
        Feature transform for each edge type (communication channel)
        '''
        g.nodes['P'].data['h_p2p'] = feat_dict['P']
        g.nodes['P'].data['h_p2a'] = feat_dict['P']
        g.nodes['A'].data['h_a2p'] = feat_dict['A']
        g.nodes['A'].data['h_a2a'] = feat_dict['A']

        # for state-related edges
        Whp2s = self.fc['p2s'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['P'].data['Wh_p2s'] = Whp2s

        Wha2s = self.fc['a2s'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['A'].data['Wh_a2s'] = Wha2s

        Whin = self.fc['in'](feat_dict['state'].double()).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['state'].data['Wh_in'] = Whin

        '''
        Attention computation on subgraphs
        '''
        # p2p message passing
        if g['p2p'].number_of_edges() > 0:
            g['p2p'].update_all(self.lossy_u_mul_e('p2p'),
                                fn.sum('m_p2p', 'ft_p2p'))

        # p2a message passing
        if g['p2a'].number_of_edges() > 0:
            g['p2a'].update_all(self.lossy_u_mul_e('p2a'),
                                fn.sum('m_p2a', 'ft_p2a'))

        # a2p message passing
        if g['a2p'].number_of_edges() > 0:
            g['a2p'].update_all(self.lossy_u_mul_e('a2p'),
                                fn.sum('m_a2p', 'ft_a2p'))

        # a2a message passing
        if g['a2a'].number_of_edges() > 0:
            g['a2a'].update_all(self.lossy_u_mul_e('a2a'),
                                fn.sum('m_a2a', 'ft_a2a'))

        # p2s
        Attn_src_p2s = (Whp2s * self.p2s_src).sum(dim=-1).unsqueeze(-1)
        Attn_dst_p2s = (Whin * self.p2s_dst).sum(dim=-1).unsqueeze(-1)

        g['p2s'].srcdata.update({'Attn_src_p2s': Attn_src_p2s})
        g['p2s'].dstdata.update({'Attn_dst_p2s': Attn_dst_p2s})

        g['p2s'].apply_edges(fn.u_add_v('Attn_src_p2s', 'Attn_dst_p2s', 'e_p2s'))
        e_p2s = self.leaky_relu(g['p2s'].edata.pop('e_p2s'))

        # compute softmax
        g['p2s'].edata['a_p2s'] = edge_softmax(g['p2s'], e_p2s)

        # message passing
        g['p2s'].update_all(fn.u_mul_e('Wh_p2s', 'a_p2s', 'm_p2s'),
                            fn.sum('m_p2s', 'ft_p2s'))

        # a2s
        Attn_src_a2s = (Wha2s * self.a2s_src).sum(dim=-1).unsqueeze(-1)
        Attn_dst_a2s = (Whin * self.a2s_dst).sum(dim=-1).unsqueeze(-1)

        g['a2s'].srcdata.update({'Attn_src_a2s': Attn_src_a2s})
        g['a2s'].dstdata.update({'Attn_dst_a2s': Attn_dst_a2s})

        g['a2s'].apply_edges(fn.u_add_v('Attn_src_a2s', 'Attn_dst_a2s', 'e_a2s'))
        e_a2s = self.leaky_relu(g['a2s'].edata.pop('e_a2s'))

        # compute softmax
        g['a2s'].edata['a_a2s'] = edge_softmax(g['a2s'], e_a2s)

        # message passing
        g['a2s'].update_all(fn.u_mul_e('Wh_a2s', 'a_a2s', 'm_a2s'),
                            fn.sum('m_a2s', 'ft_a2s'))

        '''
        Combine features from subgraphs
            Sum up to hi'
            Need to check if subgraph is valid
        '''
        # new feature of P
        Whp_new = g.nodes['P'].data['Wh_P'].clone()

        if g['p2p'].number_of_edges() > 0:
            Whp_new += g.nodes['P'].data['ft_p2p']
        if g['a2p'].number_of_edges() > 0:
            Whp_new += g.nodes['P'].data['ft_a2p']

        g.nodes['P'].data['h'] = Whp_new

        # new feature of A
        Wha_new = g.nodes['A'].data['Wh_A'].clone()
        if g['p2a'].number_of_edges() > 0:
            Wha_new += g.nodes['A'].data['ft_p2a']
        if g['a2a'].number_of_edges() > 0:
            Wha_new += g.nodes['A'].data['ft_a2a']

        g.nodes['A'].data['h'] = Wha_new

        # new feature of state
        Whstate_new = g.nodes['state'].data['Wh_in'] + \
            g.nodes['state'].data['ft_p2s'] + \
                g.nodes['state'].data['ft_a2s']

        g.nodes['state'].data['h'] = Whstate_new

        # deal with relu activation
        if self.use_relu:
            return {ntype : self.relu(g.nodes[ntype].data['h']) for ntype in g.ntypes}
        else:
            return {ntype : g.nodes[ntype].data['h'] for ntype in g.ntypes}

# merge = 'cat' or 'avg'
class MultiHeteroGATLayerLossyReal(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat',
            comm_range_P=-1, comm_range_A=-1, min_comm_loss=0, max_comm_loss=0.3):
        super(MultiHeteroGATLayerLossyReal, self).__init__()

        self._num_heads = num_heads
        self._merge = merge

        if self._merge == 'cat':
            self.gat_conv = HeteroGATLayerLossyReal(in_dim, out_dim, num_heads,
                comm_range_P=comm_range_P, comm_range_A=comm_range_A,
                min_comm_loss=min_comm_loss, max_comm_loss=max_comm_loss)
        else:
            self.gat_conv = HeteroGATLayerLossyReal(in_dim, out_dim, num_heads,
                use_relu=False, comm_range_P=comm_range_P, comm_range_A=comm_range_A,
                min_comm_loss=min_comm_loss, max_comm_loss=max_comm_loss)

    def forward(self, g, feat_dict):
        tmp = self.gat_conv(g, feat_dict)
        results = {}

        if self._merge == 'cat':
            # concat on the output feature dimension (dim=1)
            for ntype in g.ntypes:
                results[ntype] = torch.flatten(tmp[ntype], 1)
        else:
            # merge using average
            for ntype in g.ntypes:
                results[ntype] = torch.mean(tmp[ntype], 1)

        return results
