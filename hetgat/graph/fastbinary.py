# -*- coding: utf-8 -*-
"""
@author: dmartin99

Version: 2021-6-24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.ops import edge_softmax
import numpy as np
# from dgl.nn.functional import edge_softmax

import sys
sys.path.insert(0, '/home/eseraj3/HetNet_MARL_Communication/test/IC3Net/utils.py')
from utils import binary_comm_loss, get_comm_loss_bitless_k

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).double()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# in_dim: dict of input feature dimension for each node
# out_dim: dict of output feature dimension for each node
class HeteroGATLayerBinary(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, msg_dim, l_alpha=0.2, use_relu=True, use_gumble=True):
        super(HeteroGATLayerBinary, self).__init__()
        self._num_heads  = num_heads
        self._in_dim = in_dim
        self._out_dim = out_dim

        self.fc = nn.ModuleDict({
            'P': nn.Linear(in_dim['P'], out_dim['P'] * num_heads),
            'A': nn.Linear(in_dim['A'], out_dim['A'] * num_heads),
            'p2s': nn.Linear(in_dim['P'], out_dim['state'] * num_heads),
            'a2s': nn.Linear(in_dim['A'], out_dim['state'] * num_heads),
            'in': nn.Linear(in_dim['state'], out_dim['state'] * num_heads)
            })

        self.encoder = nn.ModuleDict({
                'P': nn.Linear(in_dim['P'], msg_dim),
                'A': nn.Linear(in_dim['A'], msg_dim)
            })

        self.decoder = nn.ModuleDict({
                'P': nn.Linear(msg_dim, out_dim['P'] * num_heads),
                'A': nn.Linear(msg_dim, out_dim['A'] * num_heads)
            })

        self.use_gumble = use_gumble
        if use_gumble:
            self.binarize = nn.Linear(1, 2)
            self.bin = torch.Tensor([0,1])

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
        # for state-related edges
        Whp2s = self.fc['p2s'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['P'].data['Wh_p2s'] = Whp2s

        Wha2s = self.fc['a2s'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['A'].data['Wh_a2s'] = Wha2s

        Whin = self.fc['in'](feat_dict['state'].double()).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['state'].data['Wh_in'] = Whin

        '''
        From hi to Wehi    ## Encoder 1 ##
        '''
        # message of P, NxH
        Wehp = self.encoder['P'](feat_dict['P'])
        #g.nodes['P'].data['Weh_P'] = Wehp

        # message of A
        Weha = self.encoder['A'](feat_dict['A'])
        #g.nodes['A'].data['Weh_A'] = Weha

        '''
        From Wehi to 010 (msgi)   ## Encoder 2 ##
        '''
        # binarization of P
        if self.use_gumble:
            Wehp = Wehp.unsqueeze(-1)   # NxHx1
            Wehp_b4 = self.binarize(Wehp) # NxHx2
            msg_P = F.gumbel_softmax(Wehp_b4, tau=1, hard=True) # NxHx2
            msg_P = torch.matmul(msg_P, self.bin) # NxH
        else:
            msg_P = STEFunction.apply(Wehp)

        g.nodes['P'].data['msg'] = msg_P


        # binarization of A
        if self.use_gumble:
            Weha = Weha.unsqueeze(-1)   # NxHx1
            Weha_b4 = self.binarize(Weha) # NxHx2
            msg_A = F.gumbel_softmax(Weha_b4, tau=1, hard=True)  # NxHx2
            msg_A = torch.matmul(msg_A, self.bin) # NxH

        else:
            msg_A = STEFunction.apply(Weha)

        g.nodes['A'].data['msg'] = msg_A

        '''
        From 010 (msgi) to mi 		## Decoder 1 ##
        '''
        # decode from P to P
        m_p2p = self.decoder['P'](g.nodes['P'].data['msg']).view(-1, self._num_heads, self._out_dim['P'])
        g.nodes['P'].data['msg_toP'] = m_p2p

        # decode from P to A
        m_p2a = self.decoder['A'](g.nodes['P'].data['msg']).view(-1, self._num_heads, self._out_dim['A'])
        g.nodes['P'].data['msg_toA'] = m_p2a

        # decode from A to P
        m_a2p = self.decoder['P'](g.nodes['A'].data['msg']).view(-1, self._num_heads, self._out_dim['P'])
        g.nodes['A'].data['msg_toP'] = m_a2p

        # decode from A to A
        m_a2a = self.decoder['A'](g.nodes['A'].data['msg']).view(-1, self._num_heads, self._out_dim['A'])
        g.nodes['A'].data['msg_toA'] = m_a2a

        '''
        Attention computation on subgraphs
        '''
        # p2p
        if g['p2p'].number_of_edges() > 0:
            Attn_src_p2p = (m_p2p * self.p2p_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_p2p = (Whp * self.p2p_dst).sum(dim=-1).unsqueeze(-1)
            g['p2p'].srcdata.update({'Attn_src_p2p': Attn_src_p2p})
            g['p2p'].dstdata.update({'Attn_dst_p2p': Attn_dst_p2p})

            g['p2p'].apply_edges(fn.u_add_v('Attn_src_p2p', 'Attn_dst_p2p', 'e_p2p'))
            e_p2p = self.leaky_relu(g['p2p'].edata.pop('e_p2p'))

            # compute softmax
            g['p2p'].edata['a_p2p'] = edge_softmax(g['p2p'], e_p2p)
            # message passing
            g['p2p'].update_all(fn.u_mul_e('msg_toP', 'a_p2p', 'm_p2p'),
                                fn.sum('m_p2p', 'ft_p2p'))

        # p2a
        if g['p2a'].number_of_edges() > 0:
            Attn_src_p2a = (m_p2a * self.p2a_src).sum(dim=-1).unsqueeze(-1)
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
            g['p2a'].update_all(fn.u_mul_e('msg_toA', 'a_p2a', 'm_p2a'),
                                fn.sum('m_p2a', 'ft_p2a'))
            # results =  g.nodes['A'].data['ft_p2a']

        # a2p
        if g['a2p'].number_of_edges() > 0:
            Attn_src_a2p = (m_a2p * self.a2p_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_a2p = (Whp * self.a2p_dst).sum(dim=-1).unsqueeze(-1)
            g['a2p'].srcdata.update({'Attn_src_a2p': Attn_src_a2p})
            g['a2p'].dstdata.update({'Attn_dst_a2p': Attn_dst_a2p})

            g['a2p'].apply_edges(fn.u_add_v('Attn_src_a2p', 'Attn_dst_a2p', 'e_a2p'))
            e_a2p = self.leaky_relu(g['a2p'].edata.pop('e_a2p'))

            # compute softmax
            g['a2p'].edata['a_a2p'] = edge_softmax(g['a2p'], e_a2p)
            # message passing
            g['a2p'].update_all(fn.u_mul_e('msg_toP', 'a_a2p', 'm_a2p'),
                                fn.sum('m_a2p', 'ft_a2p'))

        # a2a
        if g['a2a'].number_of_edges() > 0:
            Attn_src_a2a = (m_a2a * self.a2a_src).sum(dim=-1).unsqueeze(-1)
            Attn_dst_a2a = (Wha * self.a2a_dst).sum(dim=-1).unsqueeze(-1)
            g['a2a'].srcdata.update({'Attn_src_a2a': Attn_src_a2a})
            g['a2a'].dstdata.update({'Attn_dst_a2a': Attn_dst_a2a})

            g['a2a'].apply_edges(fn.u_add_v('Attn_src_a2a', 'Attn_dst_a2a', 'e_a2a'))
            e_a2a = self.leaky_relu(g['a2a'].edata.pop('e_a2a'))

            # compute softmax
            g['a2a'].edata['a_a2a'] = edge_softmax(g['a2a'], e_a2a)
            # message passing
            g['a2a'].update_all(fn.u_mul_e('msg_toA', 'a_a2a', 'm_a2a'),
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
class MultiHeteroGATLayerBinary(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, msg_dim, merge='cat'):
        super(MultiHeteroGATLayerBinary, self).__init__()

        self._num_heads = num_heads
        self._merge = merge

        if self._merge == 'cat':
            self.gat_conv = HeteroGATLayerBinary(in_dim, out_dim, num_heads, msg_dim)
        else:
            self.gat_conv = HeteroGATLayerBinary(in_dim, out_dim, num_heads, msg_dim, use_relu = False)

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
class HeteroGATLayerLossyBinary(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, msg_dim, l_alpha=0.2,
            use_relu=True, use_gumble=True, comm_range_P=-1, comm_range_A=-1,
            min_comm_loss=0, max_comm_loss=0.3):
        super(HeteroGATLayerLossyBinary, self).__init__()
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
            'p2s': nn.Linear(in_dim['P'], out_dim['state'] * num_heads),
            'a2s': nn.Linear(in_dim['A'], out_dim['state'] * num_heads),
            'in': nn.Linear(in_dim['state'], out_dim['state'] * num_heads)
            })

        self.encoder = nn.ModuleDict({
                'P': nn.Linear(in_dim['P'], msg_dim),
                'A': nn.Linear(in_dim['A'], msg_dim)
            })

        self.decoder = nn.ModuleDict({
                'P': nn.Linear(msg_dim, out_dim['P'] * num_heads),
                'A': nn.Linear(msg_dim, out_dim['A'] * num_heads)
            })

        self.use_gumble = use_gumble
        if use_gumble:
            self.binarize = nn.Linear(1, 2)
            self.bin = torch.Tensor([0,1])

        self.softmax = nn.Softmax(dim=1)
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

            lossy_h = edges.src['msg'].clone()
            lossy_h = binary_comm_loss(dist, lossy_h, bitless_k)

            # Calculate lossy(u) * e
            lossy_Wh = self.decoder[receiver_type](lossy_h).view(-1, self._num_heads, self._out_dim[receiver_type])

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
        # for state-related edges
        Whp2s = self.fc['p2s'](feat_dict['P']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['P'].data['Wh_p2s'] = Whp2s

        Wha2s = self.fc['a2s'](feat_dict['A']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['A'].data['Wh_a2s'] = Wha2s

        Whin = self.fc['in'](feat_dict['state'].double()).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['state'].data['Wh_in'] = Whin

        '''
        From hi to Wehi    ## Encoder 1 ##
        '''
        # message of P, NxH
        Wehp = self.encoder['P'](feat_dict['P'])
        #g.nodes['P'].data['Weh_P'] = Wehp

        # message of A
        Weha = self.encoder['A'](feat_dict['A'])
        #g.nodes['A'].data['Weh_A'] = Weha

        '''
        From Wehi to 010 (msgi)   ## Encoder 2 ##
        '''
        # binarization of P
        if self.use_gumble:
            Wehp = Wehp.unsqueeze(-1)   # NxHx1
            Wehp_b4 = self.binarize(Wehp) # NxHx2
            msg_P = F.gumbel_softmax(Wehp_b4, tau=1, hard=True) # NxHx2
            msg_P = torch.matmul(msg_P, self.bin) # NxH
        else:
            msg_P = STEFunction.apply(Wehp)

        g.nodes['P'].data['msg'] = msg_P

        # binarization of A
        if self.use_gumble:
            Weha = Weha.unsqueeze(-1)   # NxHx1
            Weha_b4 = self.binarize(Weha) # NxHx2
            msg_A = F.gumbel_softmax(Weha_b4, tau=1, hard=True)  # NxHx2
            msg_A = torch.matmul(msg_A, self.bin) # NxH
        else:
            msg_A = STEFunction.apply(Weha)

        g.nodes['A'].data['msg'] = msg_A

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
            # results =  g.nodes['A'].data['ft_p2a']

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
class MultiHeteroGATLayerLossyBinary(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, msg_dim, merge='cat',
            comm_range_P=-1, comm_range_A=-1, min_comm_loss=0,
            max_comm_loss=0.3):
        super(MultiHeteroGATLayerLossyBinary, self).__init__()

        self._num_heads = num_heads
        self._merge = merge

        if self._merge == 'cat':
            self.gat_conv = HeteroGATLayerLossyBinary(in_dim, out_dim, num_heads,
                msg_dim, comm_range_P=comm_range_P, comm_range_A=comm_range_A,
                min_comm_loss=min_comm_loss, max_comm_loss=max_comm_loss)
        else:
            self.gat_conv = HeteroGATLayerLossyBinary(in_dim, out_dim, num_heads,
                msg_dim, use_relu=False, comm_range_P=comm_range_P,
                comm_range_A=comm_range_A, min_comm_loss=min_comm_loss,
                max_comm_loss=max_comm_loss)

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
