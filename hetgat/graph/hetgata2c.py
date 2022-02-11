# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 23:06:14 2020

@author: pheno

Hetero graph conv layer for UAV coordination
    with state node as centralized Critic

Version: 2020-11-29
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

# in_dim: dict of input feature dimension for each node
# out_dim: dict of output feature dimension for each node
# msg_dim: dimension of binarized message
class HeteroGATLayerA2C(nn.Module):

    def __init__(self, in_dim, out_dim, msg_dim, l_alpha = 0.2, use_relu = True):
        super(HeteroGATLayerA2C, self).__init__()

        self.fc = nn.ModuleDict({
                'P': nn.Linear(in_dim['P'], out_dim['P']),
                'A': nn.Linear(in_dim['P'], out_dim['P']),
                'p2s': nn.Linear(in_dim['P'], out_dim['state']),
                'a2s': nn.Linear(in_dim['P'], out_dim['state']),
                'in': nn.Linear(in_dim['state'], out_dim['state'])
            })

        self.encoder = nn.ModuleDict({
                'P': nn.Linear(in_dim['P'], msg_dim),
                'A': nn.Linear(in_dim['P'], msg_dim)
            })

        self.binarize = nn.Linear(1, 2)

        self.bin = torch.Tensor([0,1]).cuda()

        self.decoder = nn.ModuleDict({
                'P': nn.Linear(msg_dim, out_dim['P']),
                'A': nn.Linear(msg_dim, out_dim['P'])
            })

        self.leaky_relu = nn.LeakyReLU(negative_slope = l_alpha)
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()

        # attention coefficients
        self.attn_fc_p2p = nn.Linear(2 * out_dim['P'], 1, bias=False)
        self.attn_fc_p2a = nn.Linear(2 * out_dim['P'], 1, bias=False)
        self.attn_fc_a2p = nn.Linear(2 * out_dim['P'], 1, bias=False)
        self.attn_fc_a2a = nn.Linear(2 * out_dim['P'], 1, bias=False)

        # state node
        self.attn_fc_p2s = nn.Linear(2 * out_dim['state'], 1, bias=False)
        self.attn_fc_a2s = nn.Linear(2 * out_dim['state'], 1, bias=False)

    def attn_p2p(self, edges):
        z2 = torch.cat([edges.src['msg_toP'], edges.dst['Wh_P']], dim=1)
        a = self.attn_fc_p2p(z2)
        return {'e_p2p': self.leaky_relu(a)}

    def message_p2p(self, edges):
        return {'z_p2p': edges.src['msg_toP'],
                'e_p2p': edges.data['e_p2p']}

    def reduce_p2p(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_p2p'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_p2p'], dim=1)
        return {'h_recv': h}

    def attn_p2a(self, edges):
        z2 = torch.cat([edges.src['msg_toA'], edges.dst['Wh_A']], dim=1)
        a = self.attn_fc_p2a(z2)
        return {'e_p2a': self.leaky_relu(a)}

    def message_p2a(self, edges):
        return {'z_p2a': edges.src['msg_toA'],
                'e_p2a': edges.data['e_p2a']}

    def reduce_p2a(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_p2a'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_p2a'], dim=1)
        return {'h_recv': h}

    def attn_a2p(self, edges):
        z2 = torch.cat([edges.src['msg_toP'], edges.dst['Wh_P']], dim=1)
        a = self.attn_fc_a2p(z2)
        return {'e_a2p': self.leaky_relu(a)}

    def message_a2p(self, edges):
        return {'z_a2p': edges.src['msg_toP'],
                'e_a2p': edges.data['e_a2p']}

    def reduce_a2p(self, nodes):
        alpha = F.softmax(nodes.mailbox['e_a2p'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z_a2p'], dim=1)
        return {'h_recv': h}

    def attn_a2a(self, edges):
        z2 = torch.cat([edges.src['msg_toA'], edges.dst['Wh_A']], dim=1)
        a = self.attn_fc_a2a(z2)
        return {'e_a2a': self.leaky_relu(a)}

    def message_a2a(self, edges):
        return {'z_a2a': edges.src['msg_toA'],
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
        feat_dict['P'] = feat_dict['P'].reshape(2, 32)
        Whp = self.fc['P'](feat_dict['P'])
        g.nodes['P'].data['Wh_P'] = Whp

        # feature of A
        feat_dict['A'] = feat_dict['A'].reshape(1, 32)
        Wha = self.fc['A'](feat_dict['A'])
        g.nodes['A'].data['Wh_A'] = Wha

        # for state-related edges
        Whp2s = self.fc['p2s'](feat_dict['P'])
        g.nodes['P'].data['Wh_p2s'] = Whp2s

        Wha2s = self.fc['a2s'](feat_dict['A'])
        g.nodes['A'].data['Wh_a2s'] = Wha2s

        feat_dict['state'] = feat_dict['state'].reshape(1,-1).double()
        Whin = self.fc['in'](feat_dict['state'])
        g.nodes['state'].data['Wh_in'] = Whin

        # for ntype in g.ntypes:
        #     Wh = self.fc[ntype](feat_dict[ntype])
        #     g.nodes[ntype].data['Wh_%s' % ntype] = Wh

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
        Wehp = Wehp.unsqueeze(-1)   # NxHx1
        Wehp_b4 = self.binarize(Wehp) # NxHx2
        msg_P = F.gumbel_softmax(Wehp_b4, tau=1, hard=True) # NxHx2
        # If the second argument is 1-dimensional, a 1 is appended to its
        # dimension for the purpose of the batched matrix multiple and
        # removed after.
        msg_P = torch.matmul(msg_P, self.bin) # NxH
        g.nodes['P'].data['msg'] = msg_P

        # binarization of A
        Weha = Weha.unsqueeze(-1)   # NxHx1
        Weha_b4 = self.binarize(Weha) # NxHx2
        msg_A = F.gumbel_softmax(Weha_b4, tau=1, hard=True) # NxHx2
        # If the second argument is 1-dimensional, a 1 is appended to its
        # dimension for the purpose of the batched matrix multiple and
        # removed after.
        msg_A = torch.matmul(msg_A, self.bin) # NxH
        g.nodes['A'].data['msg'] = msg_A

        '''
        From 010 (msgi) to mi 		## Decoder 1 ##
        '''
        # decode from P to P
        m_p2p = self.decoder['P'](g.nodes['P'].data['msg'])
        g.nodes['P'].data['msg_toP'] = m_p2p

        # decode from P to A
        m_p2a = self.decoder['A'](g.nodes['P'].data['msg'])
        g.nodes['P'].data['msg_toA'] = m_p2a

        # decode from A to P
        m_a2p = self.decoder['P'](g.nodes['A'].data['msg'])
        g.nodes['A'].data['msg_toP'] = m_a2p

        # decode from A to A
        m_a2a = self.decoder['A'](g.nodes['A'].data['msg'])
        g.nodes['A'].data['msg_toA'] = m_a2a

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
            per-edge-type-message passing
        And Form mj to sum mj	reduce_func
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
class MultiHeteroGATLayerA2C(nn.Module):

    def __init__(self, in_dim, out_dim, msg_dim, num_heads, merge='cat'):
        super(MultiHeteroGATLayerA2C, self).__init__()

        self.num_heads = num_heads
        self.merge = merge

        self.heads = nn.ModuleList()

        if self.merge == 'cat':
            for i in range(self.num_heads):
                self.heads.append(HeteroGATLayerA2C(in_dim, out_dim, msg_dim))
        else:
            for i in range(self.num_heads):
                self.heads.append(HeteroGATLayerA2C(in_dim, out_dim, msg_dim,
                                                    use_relu = False))

    def forward(self, g, feat_dict):
        tmp = {}
        for ntype in feat_dict:
            tmp[ntype] = []

        for i in range(self.num_heads):
            head_out = self.heads[i](g, feat_dict)

            for ntype in feat_dict:
                tmp[ntype].append(head_out[ntype])

        results = {}
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            for ntype in feat_dict:
                results[ntype] = torch.cat(tmp[ntype], dim=1)
        else:
            # merge using average
            # this is usually the last layer to predict logits (before softmax)
            # so no relu used
            for ntype in feat_dict:
                #results[ntype] = self.relu(torch.mean(torch.stack(tmp[ntype]), dim=0))
                results[ntype] = torch.mean(torch.stack(tmp[ntype]), dim=0)

        return results
