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

from logging import raiseExceptions
from pdb import set_trace
import torch
import torch.nn as nn
# import torch.nn.functional as F
import math
import sys, os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
import dgl.function as fn
from dgl.ops import edge_softmax
sys.stderr = stderr
import numpy as np
# from dgl.nn.functional import edge_softmax
import time
from onpolicy.algorithms.hetgat_mappo.graphs.utils import real_comm_loss, get_comm_loss_bitless_k

# import sys
# sys.path.insert(0, '/home/eseraj3/HetNet_MARL_Communication/test/IC3Net/utils.py')
# from utils import real_comm_loss

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

    def __init__(self, in_dim, out_dim, num_heads, device='cpu', l_alpha=0.2, use_relu=True, action_have_vision=False):
        super(HeteroGATLayerReal, self).__init__()
        self._num_heads  = num_heads
        self._in_dim = in_dim
        self._out_dim = out_dim
        self.action_have_vision = action_have_vision

        self.device = torch.device(device)

        self.fc = nn.ModuleDict({
        'P': nn.Linear(in_dim['P'], out_dim['P'] * num_heads, device=self.device),
        'A': nn.Linear(in_dim['A'], out_dim['A'] * num_heads, device=self.device),
        'p2p': nn.Linear(in_dim['P'], out_dim['P'] * num_heads, device=self.device),
        'p2a': nn.Linear(in_dim['P'], out_dim['A'] * num_heads, device=self.device),
        'a2p': nn.Linear(in_dim['A'], out_dim['P'] * num_heads, device=self.device),
        'a2a': nn.Linear(in_dim['A'], out_dim['A'] * num_heads, device=self.device),
        'p2s': nn.Linear(in_dim['P'], out_dim['state'] * num_heads, device=self.device),
        'a2s': nn.Linear(in_dim['A'], out_dim['state'] * num_heads, device=self.device),
        'in': nn.Linear(in_dim['state'], out_dim['state'] * num_heads, device=self.device),
        # 'X': nn.Linear(in_dim['X'], out_dim['X'] * num_heads, device=self.device),
        # 'p2x': nn.Linear(in_dim['P'], out_dim['X'] * num_heads, device=self.device),
        # 'a2x': nn.Linear(in_dim['A'], out_dim['X'] * num_heads, device=self.device),
        # 'x2x': nn.Linear(in_dim['X'], out_dim['X'] * num_heads, device=self.device),
        # 'x2p': nn.Linear(in_dim['X'], out_dim['P'] * num_heads, device=self.device),
        # 'x2a': nn.Linear(in_dim['X'], out_dim['A'] * num_heads, device=self.device),
        # 'x2s': nn.Linear(in_dim['X'], out_dim['state'] * num_heads, device=self.device)
        })

        self.leaky_relu = nn.LeakyReLU(negative_slope=l_alpha)
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()

        # attention coefficients
        if self.device == torch.device("cpu"):
            self.p2p_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
            self.p2p_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
            self.p2a_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
            self.p2a_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
            self.a2p_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
            self.a2p_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
            self.a2a_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
            self.a2a_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
            # adding the third agent class
            # self.p2x_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['X'])))
            # self.p2x_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['X'])))
            # self.a2x_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['X'])))
            # self.a2x_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['X'])))
            # self.x2x_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['X'])))
            # self.x2x_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['X'])))
            # self.x2p_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P'])))
            # self.x2p_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['P']))) 
            # self.x2a_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
            # self.x2a_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['A'])))
            # state
            self.p2s_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
            self.p2s_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
            self.a2s_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
            self.a2s_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
            # self.x2s_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
            # self.x2s_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        
        # elif self.device == torch.device("cuda"):
        elif "cuda" in self.device.type:
            self.p2p_src = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['P'])).to(self.device))
            self.p2p_dst = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['P'])).to(self.device))
            self.p2a_src = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['A'])).to(self.device))
            self.p2a_dst = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['A'])).to(self.device))
            self.a2p_src = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['P'])).to(self.device))
            self.a2p_dst = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['P'])).to(self.device))
            self.a2a_src = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['A'])).to(self.device))
            self.a2a_dst = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['A'])).to(self.device))
            # adding the third agent class
            # self.p2x_src = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['X'])))
            # self.p2x_dst = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['X'])))
            # self.a2x_src = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['X'])))
            # self.a2x_dst = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['X'])))
            # self.x2x_src = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['X'])))
            # self.x2x_dst = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['X'])))
            # self.x2p_src = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['P'])))
            # self.x2p_dst = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['P']))) 
            # self.x2a_src = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['A'])))
            # self.x2a_dst = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['A'])))
            # state
            self.p2s_src = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['state'])).to(self.device))
            self.p2s_dst = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['state'])).to(self.device))
            self.a2s_src = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['state'])).to(self.device))
            self.a2s_dst = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['state'])).to(self.device))
            # self.x2s_src = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['state'])))
            # self.x2s_dst = nn.Parameter(torch.cuda.FloatTensor(size=(1, num_heads, out_dim['state'])))
        else:
            raise NotImplementedError
        # state node

        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters
        References
            1 https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html
            2 https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
        """
        gain = nn.init.calculate_gain('relu')
        # gain = 1
        
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

        # attention orthogonal initialization
        # nn.init.orthogonal_(self.p2p_src, gain=gain)
        # nn.init.orthogonal_(self.p2p_dst, gain=gain)
        # nn.init.orthogonal_(self.p2s_src, gain=gain)
        # nn.init.orthogonal_(self.p2s_dst, gain=gain)
        
        # attention
        nn.init.xavier_normal_(self.p2p_src, gain=gain)
        nn.init.xavier_normal_(self.p2p_dst, gain=gain)
        nn.init.xavier_normal_(self.p2a_src, gain=gain)
        nn.init.xavier_normal_(self.p2a_dst, gain=gain)
        nn.init.xavier_normal_(self.a2p_src, gain=gain)
        nn.init.xavier_normal_(self.a2p_dst, gain=gain)
        nn.init.xavier_normal_(self.a2a_src, gain=gain)
        nn.init.xavier_normal_(self.a2a_dst, gain=gain)

        # nn.init.xavier_normal_(self.p2x_src, gain=gain)
        # nn.init.xavier_normal_(self.p2x_dst, gain=gain)
        # nn.init.xavier_normal_(self.a2x_src, gain=gain)
        # nn.init.xavier_normal_(self.a2x_dst, gain=gain)
        # nn.init.xavier_normal_(self.x2x_src, gain=gain)
        # nn.init.xavier_normal_(self.x2x_dst, gain=gain)
        # nn.init.xavier_normal_(self.x2p_src, gain=gain)
        # nn.init.xavier_normal_(self.x2p_dst, gain=gain)
        # nn.init.xavier_normal_(self.x2a_src, gain=gain)
        # nn.init.xavier_normal_(self.x2a_dst, gain=gain)
        
        nn.init.xavier_normal_(self.p2s_src, gain=gain)
        nn.init.xavier_normal_(self.p2s_dst, gain=gain)
        nn.init.xavier_normal_(self.a2s_src, gain=gain)
        nn.init.xavier_normal_(self.a2s_dst, gain=gain)
        # nn.init.xavier_normal_(self.x2s_src, gain=gain)
        # nn.init.xavier_normal_(self.x2s_dst, gain=gain)

    def forward(self, g, feat_dict, last_layer=False):
        '''
        From hi to Whi
        '''
        # TODO: fix this
        # if feat_dict['state'][0][1]==0:
        #     feat_dict.pop('A')
        # feature of P
        if 'P' in feat_dict:
            Whp = self.fc['P'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['P'])
            g.nodes['P'].data['Wh_P'] = Whp

        # feature of A
        if 'A' in feat_dict:
            Wha = self.fc['A'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['A'])
            g.nodes['A'].data['Wh_A'] = Wha

        if 'X' in feat_dict:
            Whx = self.fc['X'](feat_dict['X']).view(-1, self._num_heads, self._out_dim['X'])
            g.nodes['X'].data['Wh_X'] = Whx
        '''
        Feature transform for each edge type (communication channel)
        '''
        if 'P' in feat_dict:
            # p2p
            Whp2p = self.fc['p2p'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['P'])
            g.nodes['P'].data['Wh_p2p'] = Whp2p
            if 'A' in feat_dict:
                # p2a
                Whp2a = self.fc['p2a'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['A'])
                g.nodes['P'].data['Wh_p2a'] = Whp2a

        if 'A' in feat_dict:
            # a2p
            Wha2p = self.fc['a2p'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['P'])
            g.nodes['A'].data['Wh_a2p'] = Wha2p

            # a2a
            Wha2a = self.fc['a2a'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['A'])
            g.nodes['A'].data['Wh_a2a'] = Wha2a

        if 'X' in feat_dict:
            # p2x
            Whp2x = self.fc['p2x'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['X'])
            g.nodes['P'].data['Wh_p2x'] = Whp2x
            
            # a2x
            Wha2x = self.fc['a2x'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['X'])
            g.nodes['A'].data['Wh_a2x'] = Wha2x
            
            # x2x
            Whx2x = self.fc['x2x'](feat_dict['X']).view(-1, self._num_heads, self._out_dim['X'])
            g.nodes['X'].data['Wh_x2x'] = Whx2x
            
            # x2p
            Whx2p = self.fc['x2p'](feat_dict['X']).view(-1, self._num_heads, self._out_dim['P'])
            g.nodes['X'].data['Wh_x2p'] = Whx2p
            
            # x2a
            Whx2a = self.fc['x2a'](feat_dict['X']).view(-1, self._num_heads, self._out_dim['A'])
            g.nodes['X'].data['Wh_x2a'] = Whx2a 
        
        
        # for state-related edges
        if 'state' in feat_dict:
            if 'P' in feat_dict:
                Whp2s = self.fc['p2s'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['state'])
                g.nodes['P'].data['Wh_p2s'] = Whp2s

            if 'A' in feat_dict:
                Wha2s = self.fc['a2s'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['state'])
                g.nodes['A'].data['Wh_a2s'] = Wha2s
                
            if 'X' in feat_dict:
                Whx2s = self.fc['x2s'](feat_dict['X']).view(-1, self._num_heads, self._out_dim['state'])
                g.nodes['X'].data['Wh_x2s'] = Whx2s
                
            # Whin = self.fc['in'](feat_dict['state'].double()).view(-1, self._num_heads, self._out_dim['state'])
            Whin = self.fc['in'](feat_dict['state']).view(-1, self._num_heads, self._out_dim['state'])
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
        if 'A' in feat_dict:
            # p2a
            if g['p2a'].number_of_edges() > 0:
                Attn_src_p2a = (Whp2a * self.p2a_src).sum(dim=-1).unsqueeze(-1)
                Attn_dst_p2a = (Wha * self.p2a_dst).sum(dim=-1).unsqueeze(-1)
                # both works
                # g.nodes['P'].data['Attn_src_p2a'] = Attn_src_p2a
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
        
        if 'X' in feat_dict:
            # p2x
            if g['p2x'].number_of_edges() > 0:
                Attn_src_p2x = (Whp2x * self.p2x_src).sum(dim=-1).unsqueeze(-1)
                Attn_dst_p2x = (Whx * self.p2x_dst).sum(dim=-1).unsqueeze(-1)
                g['p2x'].srcdata.update({'Attn_src_p2x': Attn_src_p2x})
                g['p2x'].dstdata.update({'Attn_dst_p2x': Attn_dst_p2x})

                g['p2x'].apply_edges(fn.u_add_v('Attn_src_p2x', 'Attn_dst_p2x', 'e_p2x'))
                e_p2x = self.leaky_relu(g['p2x'].edata.pop('e_p2x'))

                # compute softmax
                g['p2x'].edata['a_p2x'] = edge_softmax(g['p2x'], e_p2x)
                # message passing
                g['p2x'].update_all(fn.u_mul_e('Wh_p2x', 'a_p2x', 'm_p2x'),
                                    fn.sum('m_p2x', 'ft_p2x'))

            # a2x
            if g['a2x'].number_of_edges() > 0:
                Attn_src_a2x = (Wha2x * self.a2x_src).sum(dim=-1).unsqueeze(-1)
                Attn_dst_a2x = (Whx * self.a2x_dst).sum(dim=-1).unsqueeze(-1)
                g['a2x'].srcdata.update({'Attn_src_a2x': Attn_src_a2x})
                g['a2x'].dstdata.update({'Attn_dst_a2x': Attn_dst_a2x})

                g['a2x'].apply_edges(fn.u_add_v('Attn_src_a2x', 'Attn_dst_a2x', 'e_a2x'))
                e_a2x = self.leaky_relu(g['a2x'].edata.pop('e_a2x'))

                # compute softmax
                g['a2x'].edata['a_a2x'] = edge_softmax(g['a2x'], e_a2x)
                # message passing
                g['a2x'].update_all(fn.u_mul_e('Wh_a2x', 'a_a2x', 'm_a2x'),
                                    fn.sum('m_a2x', 'ft_a2x'))

            # x2x
            if g['x2x'].number_of_edges() > 0:
                Attn_src_x2x = (Whx2x * self.x2x_src).sum(dim=-1).unsqueeze(-1)
                Attn_dst_x2x = (Whx * self.x2x_dst).sum(dim=-1).unsqueeze(-1)
                g['x2x'].srcdata.update({'Attn_src_x2x': Attn_src_x2x})
                g['x2x'].dstdata.update({'Attn_dst_x2x': Attn_dst_x2x})

                g['x2x'].apply_edges(fn.u_add_v('Attn_src_x2x', 'Attn_dst_x2x', 'e_x2x'))
                e_x2x = self.leaky_relu(g['x2x'].edata.pop('e_x2x'))

                # compute softmax
                g['x2x'].edata['a_x2x'] = edge_softmax(g['x2x'], e_x2x)
                # message passing
                g['x2x'].update_all(fn.u_mul_e('Wh_x2x', 'a_x2x', 'm_x2x'),
                                    fn.sum('m_x2x', 'ft_x2x'))
            
            # x2p
            if g['x2p'].number_of_edges() > 0:
                Attn_src_x2p = (Whx2p * self.x2p_src).sum(dim=-1).unsqueeze(-1)
                Attn_dst_x2p = (Whp * self.x2p_dst).sum(dim=-1).unsqueeze(-1)
                g['x2p'].srcdata.update({'Attn_src_x2p': Attn_src_x2p})
                g['x2p'].dstdata.update({'Attn_dst_x2p': Attn_dst_x2p})

                g['x2p'].apply_edges(fn.u_add_v('Attn_src_x2p', 'Attn_dst_x2p', 'e_x2p'))
                e_x2p = self.leaky_relu(g['x2p'].edata.pop('e_x2p'))

                # compute softmax
                g['x2p'].edata['a_x2p'] = edge_softmax(g['x2p'], e_x2p)
                # message passing
                g['x2p'].update_all(fn.u_mul_e('Wh_x2p', 'a_x2p', 'm_x2p'),
                                    fn.sum('m_x2p', 'ft_x2p'))
            
            # x2a    
            if g['x2a'].number_of_edges() > 0:
                Attn_src_x2a = (Whx2a * self.x2a_src).sum(dim=-1).unsqueeze(-1)
                Attn_dst_x2a = (Wha * self.x2a_dst).sum(dim=-1).unsqueeze(-1)
                g['x2a'].srcdata.update({'Attn_src_x2a': Attn_src_x2a})
                g['x2a'].dstdata.update({'Attn_dst_x2a': Attn_dst_x2a})

                g['x2a'].apply_edges(fn.u_add_v('Attn_src_x2a', 'Attn_dst_x2a', 'e_x2a'))
                e_x2a = self.leaky_relu(g['x2a'].edata.pop('e_x2a'))

                # compute softmax
                g['x2a'].edata['a_x2a'] = edge_softmax(g['x2a'], e_x2a)
                # message passing
                g['x2a'].update_all(fn.u_mul_e('Wh_x2a', 'a_x2a', 'm_x2a'),
                                    fn.sum('m_x2a', 'ft_x2a'))
        
        if 'state' in feat_dict:
            # p2s
            if 'P' in feat_dict:
                if g['p2s'].number_of_edges() > 0:
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
            if 'A' in feat_dict:
                # a2s
                if g['a2s'].number_of_edges() > 0:
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
            if 'X' in feat_dict:
                # x2s
                Attn_src_x2s = (Whx2s * self.x2s_src).sum(dim=-1).unsqueeze(-1)
                Attn_dst_x2s = (Whin * self.x2s_dst).sum(dim=-1).unsqueeze(-1)
                g['x2s'].srcdata.update({'Attn_src_x2s': Attn_src_x2s})
                g['x2s'].dstdata.update({'Attn_dst_x2s': Attn_dst_x2s})

                g['x2s'].apply_edges(fn.u_add_v('Attn_src_x2s', 'Attn_dst_x2s', 'e_x2s'))
                e_x2s = self.leaky_relu(g['x2s'].edata.pop('e_x2s'))

                # compute softmax
                g['x2s'].edata['a_x2s'] = edge_softmax(g['x2s'], e_x2s)
                # message passing
                g['x2s'].update_all(fn.u_mul_e('Wh_x2s', 'a_x2s', 'm_x2s'),
                                    fn.sum('m_x2s', 'ft_x2s'))
        '''
        Combine features from subgraphs
            Sum up to hi'
            Need to check if subgraph is valid
        '''
        valid_ntypes = []

        if 'P' in feat_dict:
            valid_ntypes.append('P')
            # new feature of P
            Whp_new = g.nodes['P'].data['Wh_P'].clone()

            if g['p2p'].number_of_edges() > 0:
                Whp_new += g.nodes['P'].data['ft_p2p']
            if 'A' in feat_dict:
                if g['a2p'].number_of_edges() > 0:
                    Whp_new += g.nodes['P'].data['ft_a2p']
            if 'X' in feat_dict:
                if g['x2p'].number_of_edges() > 0:
                    Whp_new += g.nodes['P'].data['ft_x2p']
                
            g.nodes['P'].data['h'] = Whp_new

        if 'A' in feat_dict:
            valid_ntypes.append('A')
            # new feature of A
            Wha_new = g.nodes['A'].data['Wh_A'].clone()

            if g['p2a'].number_of_edges() > 0:
                Wha_new += g.nodes['A'].data['ft_p2a']
            if g['a2a'].number_of_edges() > 0:
                Wha_new += g.nodes['A'].data['ft_a2a']
            if 'X' in feat_dict:
                if g['x2a'].number_of_edges() > 0:
                    Wha_new += g.nodes['A'].data['ft_x2a']
                
            g.nodes['A'].data['h'] = Wha_new
        
        if 'X' in feat_dict:
            valid_ntypes.append('X')
            # new feature of X
            Whx_new = g.nodes['X'].data['Wh_X'].clone()
            if g['p2x'].number_of_edges() > 0:
                Whx_new += g.nodes['X'].data['ft_p2x']
            if g['a2x'].number_of_edges() > 0:
                Whx_new += g.nodes['X'].data['ft_a2x']
            if g['x2x'].number_of_edges() > 0:
                Whx_new += g.nodes['X'].data['ft_x2x']
            g.nodes['X'].data['h'] = Whx_new
            
        if 'state' in feat_dict:
            valid_ntypes.append('state')
            # new feature of state
            Whstate_new = g.nodes['state'].data['Wh_in'].clone()
            if 'P' in feat_dict:
                if g['p2s'].number_of_edges() > 0:
                    Whstate_new += g.nodes['state'].data['ft_p2s']
            if 'A' in feat_dict:
                if g['a2s'].number_of_edges() > 0:
                    Whstate_new += g.nodes['state'].data['ft_a2s']

            g.nodes['state'].data['h'] = Whstate_new
        
        # deal with relu activation
        if self.use_relu:
            
            return {ntype: self.relu(g.nodes[ntype].data['h']) for ntype in valid_ntypes}
        else:
            # if last_layer:
            #     return g
            return {ntype: g.nodes[ntype].data['h'] for ntype in valid_ntypes}

# merge = 'cat' or 'avg'
class MultiHeteroGATLayerReal(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, device='cpu', action_have_vision=False, merge='cat'):
        super(MultiHeteroGATLayerReal, self).__init__()

        self._num_heads = num_heads
        self._merge = merge

        if self._merge == 'cat':
            self.gat_conv = HeteroGATLayerReal(in_dim, out_dim, num_heads, device=device, action_have_vision=action_have_vision)
        else:
            self.gat_conv = HeteroGATLayerReal(in_dim, out_dim, num_heads, device=device, use_relu=False, action_have_vision=action_have_vision)

    def forward(self, g, feat_dict, last_layer=False):
        tmp = self.gat_conv(g, feat_dict, last_layer=last_layer)
        results = {}

        if self._merge == 'cat':
            # concat on the output feature dimension (dim=1)
            # for ntype in g.ntypes:
            for ntype in tmp.keys():
                results[ntype] = torch.flatten(tmp[ntype], 1)
        else:
            # merge using average
            # for ntype in g.ntypes:
            for ntype in tmp.keys():
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
        self.device = 'cuda'
        self.fc = nn.ModuleDict({
            'P': nn.Linear(in_dim['P'], out_dim['P'] * num_heads, device=self.device),
            'A': nn.Linear(in_dim['A'], out_dim['A'] * num_heads, device=self.device),
            'p2p': nn.Linear(in_dim['P'], out_dim['P'] * num_heads, device=self.device),
            'p2a': nn.Linear(in_dim['P'], out_dim['A'] * num_heads, device=self.device),
            'a2p': nn.Linear(in_dim['A'], out_dim['P'] * num_heads, device=self.device),
            'a2a': nn.Linear(in_dim['A'], out_dim['A'] * num_heads, device=self.device),
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
