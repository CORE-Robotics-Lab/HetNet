# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 23:53:25 2020

@author: pheno

Version: 2020-11-29
1. add A2C support
"""

#import random
#from collections import namedtuple
import time
import numpy as np
#import networkx as nx

import dgl
import torch

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

'''
Helper function for building heterograph
    pos: ordered list of one hot of each agents position
    num_P: number of perception UAVs
    num_A: number of action UAVs
    PnP, PnA, AnA: lists of communication pairs of different types
        [[0,2],[1,3],...]
        UAV id starts with 0
        P and A share the same idx sequence
        PnP and AnA have both edges of [0, 1] & [1, 0]
        PnA only have one (no AnP)
    with_state: w/or w/o including state summary node
    comm_range_*: range of communication from agents of * type
'''

def build_hetgraph(pos, num_P=2, num_A=1, PnP=None, PnA=None, AnA=None,
        with_state=False, with_self_loop=False, with_two_state=False,
        comm_range_P=-1, comm_range_A=-1):
    pos_coords = [cartesian_from_one_hot(x) for x in pos]
    pos_dist = {}

    # P_pos_coords = torch.Tensor(pos_coords[:num_P])
    # A_pos_coords = torch.Tensor(pos_coords[num_P:num_P + num_A])

    PnP, PnA, AnP, AnA = [], [], [], []
    PnP_dist, PnA_dist, AnP_dist, AnA_dist = [], [], [], []

    P_i = range(num_P)
    A_i = range(num_P, num_P + num_A)

    for p1 in P_i:
        for x2 in range(num_P + num_A):
            if p1 != x2:
                key = (min(p1, x2), max(p1, x2))
                comm_dist = pos_dist.get(key, np.linalg.norm(pos_coords[p1] - pos_coords[x2], ord=2))
                pos_dist[key] = comm_dist

                if comm_range_P == -1 or comm_dist <= comm_range_P:
                    if x2 < num_P:
                        PnP.append([p1, x2])
                        PnP_dist.append(comm_dist)
                    else:
                        PnA.append([p1, x2])
                        PnA_dist.append(comm_dist)

    for a1 in A_i:
        for x2 in range(num_P + num_A):
            if a1 != x2:
                key = (min(a1, x2), max(a1, x2))
                comm_dist = pos_dist.get(key, np.linalg.norm(pos_coords[a1] - pos_coords[x2], ord=2))
                pos_dist[key] = comm_dist

                if comm_range_A == -1 or comm_dist <= comm_range_A:
                    if x2 < num_P:
                        AnP.append([a1, x2])
                        AnP_dist.append(comm_dist)
                    else:
                        AnA.append([a1, x2])
                        AnA_dist.append(comm_dist)

    if with_state:
        if with_two_state:
            num_nodes_dict = {'P': num_P, 'A': num_A, 'state': 2}
        else:
            num_nodes_dict = {'P': num_P, 'A': num_A, 'state': 1}
    else:
        num_nodes_dict = {'P': num_P, 'A': num_A}

    data_dict = {}

    p2p_u, a2p_u, p2a_u, a2a_u = [], [], [], []
    p2p_v, a2p_v, p2a_v, a2a_v = [], [], [], []

    for i in range(len(PnP)):
        p2p_u.append(PnP[i][0])
        p2p_v.append(PnP[i][1])

    if with_self_loop:
        for i in range(num_P):
            p2p_u.append(i)
            p2p_v.append(i)

    for i in range(len(PnA)):
        p2a_u.append(PnA[i][0])
        p2a_v.append(PnA[i][1] - num_P)

    for i in range(len(AnP)):
        a2p_u.append(AnP[i][0] - num_P)
        a2p_v.append(AnP[i][1])

    for i in range(len(AnA)):
        a2a_u.append(AnA[i][0] - num_P)
        a2a_v.append(AnA[i][1] - num_P)

    if with_self_loop:
        for i in range(num_A):
            a2a_u.append(i)
            a2a_v.append(i)

    data_dict[('P', 'p2p', 'P')] = (p2p_u, p2p_v)
    data_dict[('P', 'p2a', 'A')] = (p2a_u, p2a_v)
    data_dict[('A', 'a2p', 'P')] = (a2p_u, a2p_v)
    data_dict[('A', 'a2a', 'A')] = (a2a_u, a2a_v)

    if with_state:
        if with_two_state:
            # state node #0 is P state node
            data_dict[('P','p2s','state')] = (list(range(num_P)),
                                              [0 for i in range(num_P)])
            # state node #1 is A state node
            data_dict[('A','a2s','state')] = (list(range(num_A)),
                                              [1 for i in range(num_A)])
            data_dict[('state', 'in', 'state')] = ([0, 1], [0, 1])
        else:
            data_dict[('P','p2s','state')] = (list(range(num_P)),
                                              np.zeros(num_P, dtype=np.int64))
            data_dict[('A','a2s','state')] = (list(range(num_A)),
                                              np.zeros(num_A, dtype=np.int64))
            data_dict[('state', 'in', 'state')] = ([0], [0])

    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    
    g['p2p'].edata.update({'dist': torch.Tensor(PnP_dist)})
    g['p2a'].edata.update({'dist': torch.Tensor(PnA_dist)})
    g['a2p'].edata.update({'dist': torch.Tensor(AnP_dist)})
    g['a2a'].edata.update({'dist': torch.Tensor(AnA_dist)})

# tring to add features before batching
    # g['p2p'].srcdata.update({'point': P_pos_coords})
    # g['p2p'].dstdata.update({'point': P_pos_coords})

    # g['p2a'].srcdata.update({'point': P_pos_coords})
    # g['p2a'].dstdata.update({'point': A_pos_coords})

    # g['a2p'].srcdata.update({'point': A_pos_coords})
    # g['a2p'].dstdata.update({'point': P_pos_coords})

    # g['a2a'].srcdata.update({'point': A_pos_coords})
    # g['a2a'].dstdata.update({'point': A_pos_coords})

    return g

def cartesian_from_one_hot(one_hot):
    dim = np.sqrt(len(one_hot))
    hot_one = np.argmax(one_hot)

    x = int(hot_one % dim)
    y = int(np.floor(hot_one / dim))

    return np.array([x, y])

'''
Used for sanity check
'''
def fake_node_helper(num_P, num_A, h_dim_P, h_dim_A):
    feat_dict = {}

    feat_dict['P'] = np.random.rand(num_P, h_dim_P)

    feat_dict['A'] = np.random.rand(num_A, h_dim_A)

    return feat_dict

'''
Used for sanity check
'''
def fake_raw_input(num_P, num_A, in_dim_raw):
    raw_f_d = {}

    raw_f_d['P_s'] = np.random.rand(num_P, in_dim_raw['Sensor'][0],
                                    in_dim_raw['Sensor'][1])

    raw_f_d['P'] = np.random.rand(num_P, in_dim_raw['Status'])

    raw_f_d['A'] = np.random.rand(num_A, in_dim_raw['Status'])

    return raw_f_d

if __name__ == '__main__':
    num_P = 2
    num_A = 3
    PnP = []
    #PnP = [[0,1]]
    PnA = [[0,2],[1,3],[1,4]] # first is P, second is A
    AnA = [[3,4],[4,3]]

    hetg = build_hetgraph(num_P, num_A, PnP, PnA, AnA, with_state = True)
    print(hetg)
    print(hetg['p2p'].number_of_edges())

    f_d = fake_node_helper(num_P, num_A, h_dim_P = 6, h_dim_A = 3)
    print(f_d)

    in_dim_raw = {'Status': 5,
                  'Sensor': (16,16)}

    r_f_d = fake_raw_input(num_P, num_A, in_dim_raw)
    print(r_f_d)
