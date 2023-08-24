
import torch
import dgl

from onpolicy.algorithms.utils.util import check
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.hetgat_mappo.algorithm.gat_actor_critic import GAT_Actor, GAT_Critic


class MAPPO_GAT_Policy(R_MAPPOPolicy):
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=...):
        super(MAPPO_GAT_Policy, self).__init__(args, obs_space, cent_obs_space, act_space, device)

        self.actor = GAT_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = GAT_Critic(args, self.share_obs_space, self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.args = args
        
    def build_hetgraph_3class(self, inputs):
        """ Builds heterograph with 3 agent types. """

        if self.args.with_state:
            if self.args.with_two_state:
                num_nodes_dict = {'P': self.args.num_P, 'A': self.args.num_A, 'X': self.args.num_X, 'state': 3}
            else:
                num_nodes_dict = {'P': self.args.num_P, 'A': self.args.num_A, 'X': self.args.num_X, 'state': 1}
        else:
            num_nodes_dict = {'P': self.args.num_P, 'A': self.args.num_A,'X': self.args.num_X}
        
        PnP, PnA, AnP, AnA, PnX, AnX, XnX, XnP, XnA = [], [], [], [],[], [], [], [],[]

        P_i = range(self.args.num_P)
        A_i = range(self.args.num_P, self.args.num_P + self.args.num_A)
        X_i = range(self.args.num_P + self.args.num_A, self.args.num_P + self.args.num_A+self.args.num_X)

        for p1 in P_i:
            for x2 in range(self.args.num_P + self.args.num_A + self.args.num_X):
                if p1 != x2:
                    if x2 < self.args.num_P:
                        PnP.append([p1, x2]) 
                    elif x2 < self.args.num_P + self.args.num_A:
                        PnA.append([p1, x2])
                    else:
                        PnX.append([p1, x2])
            
        for a1 in A_i:
            for x2 in range(self.args.num_P + self.args.num_A + self.args.num_X):
                if a1 != x2:
                    if x2 < self.args.num_P:
                        AnP.append([a1, x2])
                    elif x2 < self.args.num_P + self.args.num_A:
                        AnA.append([a1, x2])
                    else:
                        AnX.append([a1, x2])
                        
        for x1 in X_i:
            for x2 in range(self.args.num_P + self.args.num_A + self.args.num_X):
                if x2 < self.args.num_P:
                    XnP.append([x1, x2])
                elif x2 < self.args.num_P + self.args.num_A:
                    XnA.append([x1, x2])
                else:
                    XnX.append([x1, x2])
                    
        data_dict = {}
        p2p_u, a2p_u, p2a_u, a2a_u = [], [], [], []
        p2p_v, a2p_v, p2a_v, a2a_v = [], [], [], []
        p2x_u, a2x_u, x2x_u, x2p_u, x2a_u = [], [], [], [], []
        p2x_v, a2x_v, x2x_v, x2p_v, x2a_v = [], [], [], [], []
        
        for i in range(len(PnP)):
            p2p_u.append(PnP[i][0])
            p2p_v.append(PnP[i][1])
        
        for i in range(len(PnA)):
            p2a_u.append(PnA[i][0])
            p2a_v.append(PnA[i][1] - self.args.num_P)

        for i in range(len(AnP)):
            a2p_u.append(AnP[i][0] - self.args.num_P)
            a2p_v.append(AnP[i][1])

        for i in range(len(AnA)):
            a2a_u.append(AnA[i][0] - self.args.num_P)
            a2a_v.append(AnA[i][1] - self.args.num_P)
        # X-related edges
        for i in range(len(PnX)):
            p2x_u.append(PnX[i][0])
            p2x_v.append(PnX[i][1] - self.args.num_P - self.args.num_A)        

        for i in range(len(AnX)):
            a2x_u.append(AnX[i][0] - self.args.num_P)
            a2x_v.append(AnX[i][1] - self.args.num_P - self.args.num_A)    

        for i in range(len(XnX)):
            x2x_u.append(XnX[i][0] - self.args.num_P - self.args.num_A)
            x2x_v.append(XnX[i][1] - self.args.num_P - self.args.num_A) 

        for i in range(len(XnP)):
            x2p_u.append(XnP[i][0] - self.args.num_P - self.args.num_A)
            x2p_v.append(XnP[i][1]) 

        for i in range(len(XnA)):
            x2a_u.append(XnA[i][0] - self.args.num_P - self.args.num_A)
            x2a_v.append(XnA[i][1] - self.args.num_P)
            
        if self.args.with_state:
            # state node #0 is P state node
            data_dict[('P','p2s','state')] = (list(range(self.args.num_P)), [0 for i in range(self.args.num_P)])

            # state node #1 is A state node
            data_dict[('A','a2s','state')] = (list(range(self.args.num_A)),
                                            [1 for i in range(self.args.num_A)])
            data_dict[('state', 'in', 'state')] = ([0, 1], [0, 1])
        
        data_dict[('P', 'p2p', 'P')] = (p2p_u, p2p_v)
        data_dict[('P', 'p2a', 'A')] = (p2a_u, p2a_v)
        data_dict[('A', 'a2p', 'P')] = (a2p_u, a2p_v)
        data_dict[('A', 'a2a', 'A')] = (a2a_u, a2a_v)
        
        data_dict[('P', 'p2x', 'X')] = (p2x_u, p2x_v)
        data_dict[('A', 'a2x', 'X')] = (a2x_u, a2x_v)
        data_dict[('X', 'x2x', 'X')] = (x2x_u, x2x_v)
        data_dict[('X', 'x2p', 'P')] = (x2p_u, x2p_v)
        data_dict[('X', 'x2a', 'A')] = (x2a_u, x2a_v)
        
        g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict, device=self.device)
        
        # TODO: generalize edge data update for more than 1 A nodes
        # g['p2p'].edata.update({'dist': inputs[:self.args.num_P]})
        # g['p2a'].edata.update({'dist': inputs[:self.args.num_P]})
        # g['a2p'].edata.update({'dist': inputs[self.args.num_P:].expand(g.num_edges('a2p'), -1)})
        # g['a2a'].edata.update({'dist': torch.Tensor(AnA_dist)})
        '''commented out the input stuff'''
        g.nodes['P'].data['input'] = inputs[:self.args.num_P]
        g.nodes['A'].data['input'] = inputs[self.args.num_P:self.args.num_P+self.args.num_A]
        g.nodes['X'].data['input'] = inputs[self.args.num_P+self.args.num_A:]

        return g
    
    def build_hetgraph_2class(self, inputs):
        """ Builds heterograph with 2 agent types """

        if self.args.with_state:
            if self.args.with_two_state:
                num_nodes_dict = {'P': self.args.num_P, 'A': self.args.num_A, 'state': 3}
            else:
                num_nodes_dict = {'P': self.args.num_P, 'A': self.args.num_A, 'state': 1}
        else:
            num_nodes_dict = {'P': self.args.num_P, 'A': self.args.num_A}
        
        PnP, PnA, AnP, AnA = [], [], [], []
        P_i = range(self.args.num_P)
        A_i = range(self.args.num_P, self.args.num_P + self.args.num_A)
        
        for p1 in P_i:
            for x2 in range(self.args.num_P + self.args.num_A):
                if p1 != x2:
                    if x2 < self.args.num_P:
                        PnP.append([p1, x2]) 
                    else:
                        PnA.append([p1, x2])
            
        for a1 in A_i:
            for x2 in range(self.args.num_P + self.args.num_A + self.args.num_X):
                if a1 != x2:
                    if x2 < self.args.num_P:
                        AnP.append([a1, x2])
                    else:
                        AnA.append([a1, x2])
                    
        data_dict = {}
        p2p_u, a2p_u, p2a_u, a2a_u = [], [], [], []
        p2p_v, a2p_v, p2a_v, a2a_v = [], [], [], []
        
        for i in range(len(PnP)):
            p2p_u.append(PnP[i][0])
            p2p_v.append(PnP[i][1])
        
        for i in range(len(PnA)):
            p2a_u.append(PnA[i][0])
            p2a_v.append(PnA[i][1] - self.args.num_P)

        for i in range(len(AnP)):
            a2p_u.append(AnP[i][0] - self.args.num_P)
            a2p_v.append(AnP[i][1])

        for i in range(len(AnA)):
            a2a_u.append(AnA[i][0] - self.args.num_P)
            a2a_v.append(AnA[i][1] - self.args.num_P)
            
        if self.args.with_state:
            # state node #0 is P state node
            data_dict[('P','p2s','state')] = (list(range(self.args.num_P)), [0 for i in range(self.args.num_P)])
            # state node #1 is A state node
            data_dict[('A','a2s','state')] = (list(range(self.args.num_A)), [1 for i in range(self.args.num_A)])
            data_dict[('state', 'in', 'state')] = ([0, 1], [0, 1])
        
        data_dict[('P', 'p2p', 'P')] = (p2p_u, p2p_v)
        data_dict[('P', 'p2a', 'A')] = (p2a_u, p2a_v)
        data_dict[('A', 'a2p', 'P')] = (a2p_u, a2p_v)
        data_dict[('A', 'a2a', 'A')] = (a2a_u, a2a_v)
        
        g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict, device=self.device)
        
        # TODO: generalize edge data update for more than 1 A nodes
        # g['p2p'].edata.update({'dist': inputs[:self.args.num_P]})
        # g['p2a'].edata.update({'dist': inputs[:self.args.num_P]})
        # g['a2p'].edata.update({'dist': inputs[self.args.num_P:].expand(g.num_edges('a2p'), -1)})
        # g['a2a'].edata.update({'dist': torch.Tensor(AnA_dist)})
        '''commented out the input stuff, not sure if this messes up graph node orders'''

        # g.nodes['P'].data['input'] = inputs[:self.args.num_P]
        # g.nodes['A'].data['input'] = inputs[self.args.num_P:self.args.num_P+self.args.num_A]
        # g.nodes['X'].data['input'] = inputs[self.args.num_P+self.args.num_A:].view(self.args.num_X, self.input_shape)

        return g



    def build_hetgraph_1class(self, inputs):
        '''builds heterograph with 1 agent types, TODO: implement this with homogeneous graph'''
        if self.args.with_state:
            if self.args.with_two_state:
                num_nodes_dict = {'P': self.args.num_P, 'state': 3}
            else:
                num_nodes_dict = {'P': self.args.num_P, 'state': 1}
        else:
            num_nodes_dict = {'P': self.args.num_P}
        
        PnP = []
        P_i = range(self.args.num_P)
        
        for p1 in P_i:
            for x2 in range(self.args.num_P):
                if p1 != x2:
                    if x2 < self.args.num_P:
                        PnP.append([p1, x2])

        data_dict = {}
        p2p_u = []
        p2p_v = []
        
        for i in range(len(PnP)):
            p2p_u.append(PnP[i][0])
            p2p_v.append(PnP[i][1])
            
        if self.args.with_state:
            # state node #0 is P state node
            data_dict[('P','p2s','state')] = (list(range(self.args.num_P)),
                                            [0 for i in range(self.args.num_P)])
            # data_dict[('state', 'in', 'state')] = ([0, 1], [0, 1])
            data_dict[('state', 'in', 'state')] = ([0], [0])
        
        data_dict[('P', 'p2p', 'P')] = (p2p_u, p2p_v)
        
        g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict, device=self.device)
        
        # TODO: generalize edge data update for more than 1 A nodes
        # g['p2p'].edata.update({'dist': inputs[:self.args.num_P]})
        # g['p2a'].edata.update({'dist': inputs[:self.args.num_P]})
        # g['a2p'].edata.update({'dist': inputs[self.args.num_P:].expand(g.num_edges('a2p'), -1)})
        # g['a2a'].edata.update({'dist': torch.Tensor(AnA_dist)})
        '''commented out the input stuff, not sure if this messes up graph node orders'''
        g.nodes['P'].data['input'] = inputs[:self.args.num_P]
        # g.nodes['A'].data['input'] = inputs[self.args.num_P:self.args.num_P+self.args.num_A].view(self.args.num_A, self.input_shape)
        # g.nodes['X'].data['input'] = inputs[self.args.num_P+self.args.num_A:].view(self.args.num_X, self.input_shape)
        return g
        pass
    
    def build_hetgraph(self, inputs):
        # tensor: 3x3x2 (lists/object)
        # vectorized: 3x29
        # inputs = check(inputs).to(**self.actor.tpdv)
        # g = self.build_hetgraph_3class(inputs)
        
        if self.args.n_types == 1:
            g = self.build_hetgraph_1class(inputs)
        elif self.args.n_types == 2:
            g = self.build_hetgraph_2class(inputs)
        elif self.args.n_types == 3:
            g = self.build_hetgraph_3class(inputs)
        else:
            raise NotImplementedError
        g = g.to(self.device)
        return g

    def build_batch_hetgraph(self, inputs):
        # inputs = check(inputs).to(**self.actor.tpdv) #
        # inputs = inputs.view(self.args.episode_length, self.args.num_P+self.args.num_A, -1)
        graphs = []
        for i in range(self.args.episode_length):
            # input = inputs.select(0,i)
            g = self.build_hetgraph(inputs)
            # g = g.to(self.device)
            graphs.append(g)
        g_batch = dgl.batch(graphs)
        return g_batch
    
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_obs_actor, rnn_states_critic, masks,
                    available_actions=None, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.

        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        g = self.build_hetgraph(obs)
        rnn_states_critic = check(rnn_states_critic).to(**self.actor.tpdv)
        if not self.args.with_state:
            state = None
        else:
            state = cent_obs

        actions, action_log_probs, rnn_states_actor, rnn_obs_actor, state_features = self.actor(obs, state,
                                                                                                rnn_states_actor,
                                                                                                rnn_obs_actor,
                                                                                                masks,
                                                                                                available_actions,
                                                                                                deterministic,
                                                                                                graph=g)
        # state_features = state_features.detach().cpu().numpy()
        # state_features = state_features.expand(cent_obs.shape[0], cent_obs.shape[1])
        if self.args.with_state:
            values, rnn_states_critic = self.critic(state_features, rnn_states_critic, masks)
        else:
            values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)

        return values, actions, action_log_probs, rnn_states_actor, rnn_obs_actor, rnn_states_critic
    
    def get_values(self, cent_obs, obs, rnn_states_actor, rnn_obs_actor, rnn_states_critic, masks, available_actions=None):
        values, _, _, _, _, _= self.get_actions(cent_obs, obs, rnn_states_actor, rnn_obs_actor, rnn_states_critic,
                                                masks, available_actions)
        return values
    
    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_obs_actor,  rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        g_batch = self.build_batch_hetgraph(obs)
        rnn_states_critic = check(rnn_states_critic).to(**self.actor.tpdv)

        if not self.args.with_state:
            state = None
        else:
            state = cent_obs

        action_log_probs, dist_entropy, state_features = self.actor.evaluate_actions(obs,
                                                                                     state,
                                                                                     rnn_states_actor,
                                                                                     rnn_obs_actor,
                                                                                     action,
                                                                                     masks,
                                                                                     available_actions,
                                                                                     active_masks,
                                                                                     graph_batch=g_batch,
                                                                                     batch_size=self.args.episode_length)
        # state_features = state_features.detach().cpu().numpy()
        # state_features = state_features.expand(cent_obs.shape[0], cent_obs.shape[1])
        # values, _ = self.critic(cent_obs, rnn_states_critic, masks)

        if self.args.with_state:
            values, _ = self.critic(state_features, rnn_states_critic, masks)
        else:
            values, _ = self.critic(cent_obs, rnn_states_critic, masks)

        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, rnn_obs, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """

        g = self.build_hetgraph(obs)
        actions, _, rnn_states_actor, rnn_obs_actor, _ = self.actor(obs, None, rnn_states_actor, rnn_obs, masks,
                                                                    available_actions, deterministic, graph=g)
        return actions, rnn_states_actor, rnn_obs_actor
