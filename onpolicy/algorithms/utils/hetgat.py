from tkinter.messagebox import NO
from turtle import forward
import torch.nn as nn
import torch
from onpolicy.algorithms.hetgat_mappo.graphs.fastreal import MultiHeteroGATLayerReal
"""HetGAT module"""

class HetGATLayer(nn.Module):
    def __init__(self, args, outputs_dim, device = torch.device('cpu')):
        super(HetGATLayer, self).__init__()
        self.hidden_size = args.hidden_size
        self.hetnet_hidden_size = args.hetnet_hidden_dim
        self.hetnet_num_heads = args.hetnet_num_heads
        self.device = device
        self.args = args
        self.outputs_dim = outputs_dim
        self.n_types = self.args.n_types # implement a better to deal with different number of agent types

        if self.n_types == 2:
            in_dim = {'P':self.hidden_size, 'A':self.hidden_size, 'X':self.hidden_size, 'state':4}
            hid_dim = {'P':self.hetnet_hidden_size, 'A':self.hetnet_hidden_size, 'state':self.hetnet_hidden_size}
            hid_dim_input = {'P':self.hetnet_hidden_size*self.hetnet_num_heads,
                            'A':self.hetnet_hidden_size*self.hetnet_num_heads,
                            'state':self.hetnet_hidden_size*self.hetnet_num_heads}
            out_dim = {'P':self.hidden_size, 'A':self.hidden_size, 'state':8}

            if self.args.tensor_obs:
                in_dim['P'] = 128 + 16 # input dimension of perception agents is equal to the sum of state dim and obs dim
                in_dim['A'] = 128 # dimension

        elif self.n_types == 3:
            in_dim = {'P':self.hidden_size, 'A':self.hidden_size, 'X':self.hidden_size, 'state':9}
            hid_dim = {'P':self.hetnet_hidden_size, 'A':self.hetnet_hidden_size,
                    'X': self.hetnet_hidden_size, 'state':self.hetnet_hidden_size}
            hid_dim_input = {'P':self.hetnet_hidden_size*self.hetnet_num_heads, 
                            'A':self.hetnet_hidden_size*self.hetnet_num_heads,
                            'X': self.hetnet_hidden_size*self.hetnet_num_heads,
                            'state':self.hetnet_hidden_size*self.hetnet_num_heads}
            out_dim = {'P': self.outputs_dim, 'A':self.outputs_dim, 'X':self.outputs_dim, 'state':9}

        else:
            raise NotImplementedError

        self.hetgat_layer1 = MultiHeteroGATLayerReal(in_dim, hid_dim, self.hetnet_num_heads, device=self.device,
                                                     action_have_vision=True)
        self.hetgat_layer2 = MultiHeteroGATLayerReal(hid_dim_input, out_dim, self.hetnet_num_heads, device=self.device,
                                                     merge='avg')

        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, feat_dict, cent_obs, graph_batch, batch_size):


        #
        # feat_dict = {}
        # # P, A, X = torch.split(x, [self.args.num_P, self.args.num_A, self.args.num_X], dim=0)
        # P, A = torch.split(x.view(batch_size, self.args.num_P+self.args.num_A,-1), [self.args.num_P, self.args.num_A], dim=-2)
        # feat_dict['P'] = P
        # if A.numel() is not 0:
        #     feat_dict['A'] = A

        if cent_obs != None:
            feat_dict['state'] = cent_obs
        # TODO: split features x into dict['P'] and dict['A']

        h1 = self.hetgat_layer1(graph_batch.to(self.device), feat_dict)
        q_out= self.hetgat_layer2(graph_batch.to(self.device), h1, last_layer=True)
        
        if self.args.num_A == 0 and self.args.num_X == 0:
            x = q_out['P'].view(batch_size,self.args.num_P,-1)
        elif self.args.num_X == 0:
            x = torch.cat((q_out['P'].view(batch_size,self.args.num_P,-1), q_out['A'].view(batch_size,self.args.num_A,-1)), dim=1)
        # x = self.norm(q_out)
        if cent_obs is not None:
            state = q_out['state']
        else:
            state = None
        return x, state
