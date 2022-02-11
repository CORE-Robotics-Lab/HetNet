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
Version: 2021-3-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import dgl.function as fn

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

    def __init__(self, in_dim, out_dim, l_alpha = 0.2, use_relu = True):
        super(HeteroGATLayerReal, self).__init__()

        self.fc = nn.ModuleDict({
            'P': nn.Linear(in_dim['P'], out_dim['P']),
            'A': nn.Linear(in_dim['A'], out_dim['A']),
                'p2p': nn.Linear(in_dim['P'], out_dim['P']),
                'p2a': nn.Linear(in_dim['P'], out_dim['A']),
                'a2p': nn.Linear(in_dim['A'], out_dim['P']),
                'a2a': nn.Linear(in_dim['A'], out_dim['A']),
                'p2s': nn.Linear(in_dim['P'], out_dim['state']),
                'a2s': nn.Linear(in_dim['A'], out_dim['state']),
                'in': nn.Linear(in_dim['state'], out_dim['state'])
            })

        # self.fc = nn.ModuleDict({
        #     'P': nn.Linear(29, out_dim['P']),
        #     'A': nn.Linear(25, out_dim['A']),
        #     'p2p': nn.Linear(29, out_dim['P']),
        #     'p2a': nn.Linear(29, out_dim['A']),
        #     'a2p': nn.Linear(25, out_dim['P']),
        #     'a2a': nn.Linear(25, out_dim['A']),
        #     'p2s': nn.Linear(29, out_dim['state']),
        #     'a2s': nn.Linear(25, out_dim['state']),
        #     'in': nn.Linear(in_dim['state'], out_dim['state'])
        # })

        self.leaky_relu = nn.LeakyReLU(negative_slope = l_alpha)
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()

        # attention coefficients
        self.attn_fc_p2p = nn.Linear(2 * out_dim['P'], 1, bias=False)
        self.attn_fc_p2a = nn.Linear(2 * out_dim['A'], 1, bias=False)
        self.attn_fc_a2p = nn.Linear(2 * out_dim['P'], 1, bias=False)
        self.attn_fc_a2a = nn.Linear(2 * out_dim['A'], 1, bias=False)

        # state node
        self.attn_fc_p2s = nn.Linear(2 * out_dim['state'], 1, bias=False)
        self.attn_fc_a2s = nn.Linear(2 * out_dim['state'], 1, bias=False)

        #self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters
        References
            1 https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html
            2 https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
        """
        gain = nn.init.calculate_gain('relu')

        # fc weights
        nn.init.xavier_normal_(self.fc['P'].weight, gain=gain)
        nn.init.xavier_normal_(self.fc['A'].weight, gain=gain)
        nn.init.xavier_normal_(self.fc['p2p'].weight, gain=gain)
        nn.init.xavier_normal_(self.fc['p2a'].weight, gain=gain)
        nn.init.xavier_normal_(self.fc['a2p'].weight, gain=gain)
        nn.init.xavier_normal_(self.fc['a2a'].weight, gain=gain)
        nn.init.xavier_normal_(self.fc['p2s'].weight, gain=gain)
        nn.init.xavier_normal_(self.fc['a2s'].weight, gain=gain)
        nn.init.xavier_normal_(self.fc['in'].weight, gain=gain)

        # fc biases
        # https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/graphconv.html#GraphConv
        nn.init.zeros_(self.fc['P'].bias)
        nn.init.zeros_(self.fc['A'].bias)
        nn.init.zeros_(self.fc['p2p'].bias)
        nn.init.zeros_(self.fc['p2a'].bias)
        nn.init.zeros_(self.fc['a2p'].bias)
        nn.init.zeros_(self.fc['a2a'].bias)
        nn.init.zeros_(self.fc['p2s'].bias)
        nn.init.zeros_(self.fc['a2s'].bias)
        nn.init.zeros_(self.fc['in'].bias)

        # attention
        nn.init.xavier_normal_(self.attn_fc_p2p.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc_p2a.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc_a2p.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc_a2a.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_fc_p2s.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc_a2s.weight, gain=gain)

    def attn_p2p(self, edges):
        z2 = torch.cat([edges.src['Wh_p2p'], edges.dst['Wh_P']], dim=1)
        a = self.attn_fc_p2p(z2)
        return {'e_p2p': self.leaky_relu(a)}

    def message_p2p(self, edges):
        return {'z_p2p': edges.src['Wh_p2p'],
                'e_p2p': edges.data['e_p2p']}

    def reduce_p2p(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_p2p'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_p2p'], dim=1)
        return {'h_recv': h}

    def attn_p2a(self, edges):
        z2 = torch.cat([edges.src['Wh_p2a'], edges.dst['Wh_A']], dim=1)
        a = self.attn_fc_p2a(z2)
        return {'e_p2a': self.leaky_relu(a)}

    def message_p2a(self, edges):
        return {'z_p2a': edges.src['Wh_p2a'],
                'e_p2a': edges.data['e_p2a']}

    def reduce_p2a(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_p2a'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_p2a'], dim=1)
        return {'h_recv': h}

    def attn_a2p(self, edges):
        z2 = torch.cat([edges.src['Wh_a2p'], edges.dst['Wh_P']], dim=1)
        a = self.attn_fc_a2p(z2)
        return {'e_a2p': self.leaky_relu(a)}

    def message_a2p(self, edges):
        return {'z_a2p': edges.src['Wh_a2p'],
                'e_a2p': edges.data['e_a2p']}

    def reduce_a2p(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_a2p'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_a2p'], dim=1)
        return {'h_recv': h}

    def attn_a2a(self, edges):
        z2 = torch.cat([edges.src['Wh_a2a'], edges.dst['Wh_A']], dim=1)
        a = self.attn_fc_a2a(z2)
        return {'e_a2a': self.leaky_relu(a)}

    def message_a2a(self, edges):
        return {'z_a2a': edges.src['Wh_a2a'],
                'e_a2a': edges.data['e_a2a']}

    def reduce_a2a(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_a2a'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_a2a'], dim=1)
        return {'h_recv': h}

    # P to state
    def attn_p2s(self, edges):
        z2 = torch.cat([edges.src['Wh_p2s'], edges.dst['Wh_in']], dim=1)
        a = self.attn_fc_p2s(z2)
        return {'e_p2s': self.leaky_relu(a)}

    def message_p2s(self, edges):
        return {'z_p2s': edges.src['Wh_p2s'],
                'e_p2s': edges.data['e_p2s']}

    def reduce_p2s(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_p2s'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_p2s'], dim=1)
        return {'h_recv': h}

    # A to state
    def attn_a2s(self, edges):
        z2 = torch.cat([edges.src['Wh_a2s'], edges.dst['Wh_in']], dim=1)
        a = self.attn_fc_a2s(z2)
        return {'e_a2s': self.leaky_relu(a)}

    def message_a2s(self, edges):
        return {'z_a2s': edges.src['Wh_a2s'],
                'e_a2s': edges.data['e_a2s']}

    def reduce_a2s(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_a2s'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_a2s'], dim=1)
        return {'h_recv': h}

    def forward(self, g, feat_dict):
        '''
        From hi to Whi
        '''
        # feature of P
        Whp = self.fc['P'](feat_dict['P'])
        g.nodes['P'].data['Wh_P'] = Whp

        # feature of A
        Wha = self.fc['A'](feat_dict['A'])
        g.nodes['A'].data['Wh_A'] = Wha

        '''
        Feature transform for each edge type (communication channel)
        '''
        # p2p
        Whp2p = self.fc['p2p'](feat_dict['P'])
        g.nodes['P'].data['Wh_p2p'] = Whp2p.reshape((2,-1))

        # p2a
        Whp2a = self.fc['p2a'](feat_dict['P'])
        g.nodes['P'].data['Wh_p2a'] = Whp2a.reshape((2,-1))

        # a2p
        Wha2p = self.fc['a2p'](feat_dict['A'])
        g.nodes['A'].data['Wh_a2p'] = Wha2p.reshape((1,-1))

        # a2a
        Wha2a = self.fc['a2a'](feat_dict['A'])
        g.nodes['A'].data['Wh_a2a'] = Wha2a.reshape((1,-1))

        # for state-related edges
        Whp2s = self.fc['p2s'](feat_dict['P'])
        g.nodes['P'].data['Wh_p2s'] = Whp2s.reshape((2,-1))

        Wha2s = self.fc['a2s'](feat_dict['A'])
        g.nodes['A'].data['Wh_a2s'] = Wha2s.reshape((1,-1))

        Whin = self.fc['in'](feat_dict['state'].double())
        g.nodes['state'].data['Wh_in'] = Whin.reshape((1,-1))

        '''
        Use m_src{to_dst_type} and whi_dst to calculate e(src,dst)
        '''
        g['p2p'].apply_edges(self.attn_p2p)
        g['p2a'].apply_edges(self.attn_p2a)
        g['a2p'].apply_edges(self.attn_a2p)
        g['a2a'].apply_edges(self.attn_a2a)

        # for state-related edges
        g['p2s'].apply_edges(self.attn_p2s)
        g['a2s'].apply_edges(self.attn_a2s)

        '''
        Send mi using message funcs     Receive mj
            mi is the same as Wh_edge_type
            per-edge-type-message passing
        And from mj to sum mj	reduce_func
        '''
        funcs = {}
        funcs['p2p'] = (self.message_p2p, self.reduce_p2p)
        funcs['p2a'] = (self.message_p2a, self.reduce_p2a)
        funcs['a2p'] = (self.message_a2p, self.reduce_a2p)
        funcs['a2a'] = (self.message_a2a, self.reduce_a2a)

        # for state-related edges
        funcs['p2s'] = (self.message_p2s, self.reduce_p2s)
        funcs['a2s'] = (self.message_a2s, self.reduce_a2s)
        funcs['in'] = (fn.copy_src('Wh_in', 'z_in'), fn.sum('z_in', 'h_recv'))

        g.multi_update_all(funcs, 'sum')

        '''
        Sum up to hi'
            Need to check if h_recv exists
        '''
        # new feature of P
        if 'h_recv' in g.nodes['P'].data.keys():
            Whp_new = g.nodes['P'].data['Wh_P'] + g.nodes['P'].data['h_recv']
        else:
            Whp_new = g.nodes['P'].data['Wh_P']
        g.nodes['P'].data['h'] = Whp_new

        # new feature of A
        if 'h_recv' in g.nodes['A'].data.keys():
            Wha_new = g.nodes['A'].data['Wh_A'] + g.nodes['A'].data['h_recv']
        else:
            Wha_new = g.nodes['A'].data['Wh_A']
        g.nodes['A'].data['h'] = Wha_new

        # new feature of state
        Whstate_new = g.nodes['state'].data['h_recv']
        g.nodes['state'].data['h'] = Whstate_new

        # deal with relu activation
        if self.use_relu:
            return {ntype : self.relu(g.nodes[ntype].data['h']) for ntype in g.ntypes}
        else:
            return {ntype : g.nodes[ntype].data['h'] for ntype in g.ntypes}

# merge = 'cat' or 'avg'
class MultiHeteroGATLayerReal(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeteroGATLayerReal, self).__init__()

        self.num_heads = num_heads
        self.merge = merge

        self.heads = nn.ModuleList()

        if self.merge == 'cat':
            for i in range(self.num_heads):
                self.heads.append(HeteroGATLayerReal(in_dim, out_dim))
        else:
            for i in range(self.num_heads):
                self.heads.append(HeteroGATLayerReal(in_dim, out_dim, use_relu = False))

    def forward(self, g, feat_dict):
        tmp = {}
        for ntype in g.ntypes:
            tmp[ntype] = []

        for i in range(self.num_heads):
            head_out = self.heads[i](g, feat_dict)

            for ntype in feat_dict:
                tmp[ntype].append(head_out[ntype])

        results = {}
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            for ntype in g.ntypes:
                results[ntype] = torch.cat(tmp[ntype], dim=1)
        else:
            # merge using average
            for ntype in g.ntypes:
                results[ntype] = torch.mean(torch.stack(tmp[ntype]), dim=0)

        return results

# in_dim: dict of input feature dimension for each node
# out_dim: dict of output feature dimension for each node
class HeteroGATReal(nn.Module):

    def __init__(self, in_dim, out_dim, l_alpha=0.2, use_relu=True):
        super(HeteroGATReal, self).__init__()

        self.fc = nn.ModuleDict({
            'P': nn.Linear(in_dim['P'], out_dim['P']),
            'A': nn.Linear(in_dim['A'], out_dim['A']),
            'p2p': nn.Linear(in_dim['P'], out_dim['P']),
            'p2a': nn.Linear(in_dim['P'], out_dim['A']),
            'a2p': nn.Linear(in_dim['A'], out_dim['P']),
            'a2a': nn.Linear(in_dim['A'], out_dim['A'])
        })

        self.leaky_relu = nn.LeakyReLU(negative_slope=l_alpha)
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()

        # attention coefficients
        self.attn_fc_p2p = nn.Linear(2 * out_dim['P'], 1, bias=False)
        self.attn_fc_p2a = nn.Linear(2 * out_dim['A'], 1, bias=False)
        self.attn_fc_a2p = nn.Linear(2 * out_dim['P'], 1, bias=False)
        self.attn_fc_a2a = nn.Linear(2 * out_dim['A'], 1, bias=False)

        # self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters
        References
            1 https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html
            2 https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
        """
        gain = nn.init.calculate_gain('relu')

        # fc weights
        nn.init.xavier_normal_(self.fc['P'].weight, gain=gain)
        nn.init.xavier_normal_(self.fc['A'].weight, gain=gain)
        nn.init.xavier_normal_(self.fc['p2p'].weight, gain=gain)
        nn.init.xavier_normal_(self.fc['p2a'].weight, gain=gain)
        nn.init.xavier_normal_(self.fc['a2p'].weight, gain=gain)
        nn.init.xavier_normal_(self.fc['a2a'].weight, gain=gain)

        # fc biases
        # https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/graphconv.html#GraphConv
        nn.init.zeros_(self.fc['P'].bias)
        nn.init.zeros_(self.fc['A'].bias)
        nn.init.zeros_(self.fc['p2p'].bias)
        nn.init.zeros_(self.fc['p2a'].bias)
        nn.init.zeros_(self.fc['a2p'].bias)
        nn.init.zeros_(self.fc['a2a'].bias)

        # attention
        nn.init.xavier_normal_(self.attn_fc_p2p.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc_p2a.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc_a2p.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc_a2a.weight, gain=gain)

    def attn_p2p(self, edges):
        z2 = torch.cat([edges.src['Wh_p2p'], edges.dst['Wh_P']], dim=1)
        a = self.attn_fc_p2p(z2)
        return {'e_p2p': self.leaky_relu(a)}

    def message_p2p(self, edges):
        return {'z_p2p': edges.src['Wh_p2p'],
                'e_p2p': edges.data['e_p2p']}

    def reduce_p2p(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_p2p'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_p2p'], dim=1)
        return {'h_recv': h}

    def attn_p2a(self, edges):
        z2 = torch.cat([edges.src['Wh_p2a'], edges.dst['Wh_A']], dim=1)
        a = self.attn_fc_p2a(z2)
        return {'e_p2a': self.leaky_relu(a)}

    def message_p2a(self, edges):
        return {'z_p2a': edges.src['Wh_p2a'],
                'e_p2a': edges.data['e_p2a']}

    def reduce_p2a(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_p2a'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_p2a'], dim=1)
        return {'h_recv': h}

    def attn_a2p(self, edges):
        z2 = torch.cat([edges.src['Wh_a2p'], edges.dst['Wh_P']], dim=1)
        a = self.attn_fc_a2p(z2)
        return {'e_a2p': self.leaky_relu(a)}

    def message_a2p(self, edges):
        return {'z_a2p': edges.src['Wh_a2p'],
                'e_a2p': edges.data['e_a2p']}

    def reduce_a2p(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_a2p'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_a2p'], dim=1)
        return {'h_recv': h}

    def attn_a2a(self, edges):
        z2 = torch.cat([edges.src['Wh_a2a'], edges.dst['Wh_A']], dim=1)
        a = self.attn_fc_a2a(z2)
        return {'e_a2a': self.leaky_relu(a)}

    def message_a2a(self, edges):
        return {'z_a2a': edges.src['Wh_a2a'],
                'e_a2a': edges.data['e_a2a']}

    def reduce_a2a(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_a2a'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_a2a'], dim=1)
        return {'h_recv': h}


    def forward(self, g, feat_dict):
        '''
        From hi to Whi
        '''
        # feature of P
        Whp = self.fc['P'](feat_dict['P'])
        g.nodes['P'].data['Wh_P'] = Whp

        # feature of A
        Wha = self.fc['A'](feat_dict['A'])
        g.nodes['A'].data['Wh_A'] = Wha

        '''
        Feature transform for each edge type (communication channel)
        '''
        # p2p
        Whp2p = self.fc['p2p'](feat_dict['P'])
        g.nodes['P'].data['Wh_p2p'] = Whp2p.reshape((2,-1))

        # p2a
        Whp2a = self.fc['p2a'](feat_dict['P'])
        g.nodes['P'].data['Wh_p2a'] = Whp2a.reshape((2,-1))

        # a2p
        Wha2p = self.fc['a2p'](feat_dict['A'])
        g.nodes['A'].data['Wh_a2p'] = Wha2p.reshape((1,-1))

        # a2a
        Wha2a = self.fc['a2a'](feat_dict['A'])
        g.nodes['A'].data['Wh_a2a'] = Wha2a.reshape((1,-1))

        '''
        Use m_src{to_dst_type} and whi_dst to calculate e(src,dst)
        '''
        g['p2p'].apply_edges(self.attn_p2p)
        g['p2a'].apply_edges(self.attn_p2a)
        g['a2p'].apply_edges(self.attn_a2p)
        g['a2a'].apply_edges(self.attn_a2a)

        '''
        Send mi using message funcs     Receive mj
            mi is the same as Wh_edge_type
            per-edge-type-message passing
        And from mj to sum mj	reduce_func
        '''
        funcs = {}
        funcs['p2p'] = (self.message_p2p, self.reduce_p2p)
        funcs['p2a'] = (self.message_p2a, self.reduce_p2a)
        funcs['a2p'] = (self.message_a2p, self.reduce_a2p)
        funcs['a2a'] = (self.message_a2a, self.reduce_a2a)

        g.multi_update_all(funcs, 'sum')

        '''
        Sum up to hi'
            Need to check if h_recv exists
        '''
        # new feature of P
        if 'h_recv' in g.nodes['P'].data.keys():
            Whp_new = g.nodes['P'].data['Wh_P'] + g.nodes['P'].data['h_recv']
        else:
            Whp_new = g.nodes['P'].data['Wh_P']
        g.nodes['P'].data['h'] = Whp_new

        # new feature of A
        if 'h_recv' in g.nodes['A'].data.keys():
            Wha_new = g.nodes['A'].data['Wh_A'] + g.nodes['A'].data['h_recv']
        else:
            Wha_new = g.nodes['A'].data['Wh_A']
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

        self.num_heads = num_heads
        self.merge = merge

        self.heads = nn.ModuleList()

        if self.merge == 'cat':
            for i in range(self.num_heads):
                self.heads.append(HeteroGATReal(in_dim, out_dim))
        else:
            for i in range(self.num_heads):
                self.heads.append(HeteroGATReal(in_dim, out_dim, use_relu=False))

    def forward(self, g, feat_dict):
        tmp = {}
        for ntype in g.ntypes:
            tmp[ntype] = []

        for i in range(self.num_heads):
            head_out = self.heads[i](g, feat_dict)

            for ntype in feat_dict:
                tmp[ntype].append(head_out[ntype])

        results = {}
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            for ntype in g.ntypes:
                results[ntype] = torch.cat(tmp[ntype], dim=1)
        else:
            # merge using average
            for ntype in g.ntypes:
                results[ntype] = torch.mean(torch.stack(tmp[ntype]), dim=0)

        return results
