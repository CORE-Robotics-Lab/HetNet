
import torch
import torch.nn as nn
import numpy as np

from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.rnn import LSTMLayer
from onpolicy.algorithms.utils.hetgat import HetGATLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.utils.util import get_shape_from_obs_space


class GAT_Actor(nn.Module):
    """ Actor network class for MAPPO with graph attention. Output actions given heterogeneous state and observations.
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(GAT_Actor, self).__init__()

        self.args = args

        self.hidden_size = args.hidden_size
        self.hetnet_hidden_size = args.hetnet_hidden_dim
        self.hetnet_num_heads = args.hetnet_num_heads
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self.args.tensor_obs:
            if self.args.vision == 1 or self.args.vision == 2:
                self.prepro_obs = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                                                nn.ReLU(inplace=True),
                                                nn.AdaptiveAvgPool2d((1, 1)),)
            else:
                self.prepro_obs = nn.Sequential(nn.Flatten(1, -1),
                                                nn.Linear(3, 6),
                                                nn.Linear(6, 16))

            self.prepro_stat = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3)),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2)),
                                             nn.ReLU(inplace=True),
                                             nn.AdaptiveAvgPool2d((1, 1)),)

            self.f_module_obs = nn.LSTMCell(16, 16)
            self.f_module_stat = nn.LSTMCell(128, 128)

            if self.args.upsample_x2:
                self.upsample_stat = nn.Upsample(scale_factor=2, mode='nearest')

        else:
            obs_shape = get_shape_from_obs_space(obs_space)
            base = CNNBase if len(obs_shape) == 3 else MLPBase
            self.base = base(args, obs_shape)

            if args.use_LSTM:
                self.rnn = LSTMLayer(args, self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            else:
                self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.gat = HetGATLayer(args, self.hidden_size, device=device)
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        #self.critic = nn.Linear(9, 1)

        self.relu = nn.ReLU()
        self.to(device)
        
    def forward(self, obs, cent_obs, rnn_states, rnn_obs, masks ,
                available_actions=None, deterministic=False, graph=None, batch_size=1):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param cent_obs: global state input for SSN, shoudl be disconnected during execution 
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """

        if cent_obs is not None:
            cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self.args.tensor_obs:
            x_per_stat, x_act_stat, x_per_obs = self.get_states_obs_from_tensor(obs)
            x_per_stat, x_act_stat, x_per_obs = check(x_per_stat).to(**self.tpdv), check(x_act_stat).to(**self.tpdv), \
                                                check(x_per_obs).to(**self.tpdv)

            if self.args.upsample_x2:
                x_per_stat = self.upsample_stat(x_per_stat[None, :])  # Send to upsample layer with extra batch dimension
                x_per_stat = torch.squeeze(x_per_stat, 0)  # Remove extra batch dimension

                if self.args.num_A != 0:
                    x_act_stat = self.upsample_stat(x_act_stat[None, :])
                    x_act_stat = torch.squeeze(x_act_stat, 0)

            state_per_stat = self.prepro_stat(torch.unsqueeze(x_per_stat, dim=1))
            if self.args.num_A != 0:
                state_act_stat = self.prepro_stat(torch.unsqueeze(x_act_stat, dim=1))

            if self.args.vision == 1 or self.args.vision == 2:
                x_per_obs = self.prepro_obs(x_per_obs)
            else:
                x_per_obs = self.prepro_obs(x_per_obs.squeeze())

        else:
            obs = check(obs).to(**self.tpdv)
            actor_features = self.base(obs)

        # to LSTM
        if self.args.use_LSTM:
            if self.args.tensor_obs:
                rnn_states = rnn_states.squeeze()
                rnn_obs = check(rnn_obs).to(**self.tpdv)
                rnn_obs = rnn_obs.squeeze()

                # Forward pass LSTM features associated with the state of the perception agents
                hidden_state, cell_state = torch.split(rnn_states, [self.args.num_P + self.args.num_A] * 2)
                hidden_state_per_stat, cell_state_per_stat = hidden_state[:self.args.num_P], \
                                                             cell_state[:self.args.num_P]
                hidden_state_per_stat, cell_state_per_stat = self.f_module_stat(state_per_stat.squeeze(),
                                                                                (hidden_state_per_stat,
                                                                                 cell_state_per_stat))

                # Forward pass LSTM features associated with the state of the action agents
                if self.args.num_A != 0:
                    hidden_state_act_stat, cell_state_act_stat = hidden_state[self.args.num_P:],\
                                                                 cell_state[self.args.num_P:]
                    hidden_state_act_stat, cell_state_act_stat = self.f_module_stat(
                        state_act_stat.squeeze().reshape(self.args.num_A, -1), # in case only one action agent
                        (hidden_state_act_stat,
                         cell_state_act_stat))

                # forward pass the LSTM features associated with the observations of the perception agents
                hidden_state_per_obs, cell_state_per_obs = torch.split(rnn_obs, [self.args.num_P] * 2)
                hidden_state_per_obs, cell_state_per_obs = self.f_module_obs(x_per_obs.squeeze(),
                                                                             (hidden_state_per_obs, cell_state_per_obs))

                if self.args.num_A != 0:
                    rnn_states = torch.cat((hidden_state_per_stat, hidden_state_act_stat,
                                            cell_state_per_stat, cell_state_act_stat))
                else:
                    rnn_states = torch.cat((hidden_state_per_stat, cell_state_per_stat))

                rnn_obs = torch.cat((hidden_state_per_obs, cell_state_per_obs))

                feat_dict = {}
                feat_dict['P'] = torch.cat([hidden_state_per_stat, hidden_state_per_obs], dim=1)
                if self.args.num_A != 0:
                    feat_dict['A'] = hidden_state_act_stat

            else:
                actor_features, rnn_states = self.rnn(actor_features, rnn_states)

        else:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actor_features, state_features = self.gat(feat_dict, cent_obs, graph, batch_size)
        actor_features = actor_features.squeeze()

        # action
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        
        return actions, action_log_probs, rnn_states, rnn_obs, state_features

    def get_states_obs_from_tensor(self, x):
        if len(x.shape) == 2: # if this is for a sample

            x_per_stat = []
            x_act_stat = []
            x_per_obs = []
            for i in range(self.args.num_P):
                x_per_stat.append(x[i][1])
            for i in range(self.args.num_P, self.args.num_P + self.args.num_A):
                x_act_stat.append(x[i][1])

            for i in range(self.args.num_P):
                x_per_obs.append(x[i][0])

            return torch.tensor(np.array(x_per_stat)), torch.tensor(np.array(x_act_stat)),\
                   torch.tensor(np.array(x_per_obs))

        else: # if this is for a batch of samples
            x = list(x)
            get_x_per_stat = lambda x_in: [x_in[i][1]for i in range(self.args.num_P)]
            get_x_act_stat = lambda x_in: [x_in[i][1]for i in range(self.args.num_P, self.args.num_P + self.args.num_A)]
            x_per_obs = lambda x_in: [x_in[i][0] for i in range(self.args.num_P)]

            x_per_stat = np.stack(list(map(get_x_per_stat, x)))
            x_act_stat = np.stack(list(map(get_x_act_stat, x)))
            x_per_obs = np.stack(list(map(x_per_obs, x)))

            return torch.unsqueeze(torch.tensor(np.array(x_per_stat)), dim=2), \
                   torch.unsqueeze(torch.tensor(np.array(x_act_stat)), dim=2), \
                   torch.unsqueeze(torch.tensor(np.array(x_per_obs)), dim=2)

    def evaluate_actions(self, obs, cent_obs, rnn_states, rnn_obs, action, masks, available_actions=None,
                         active_masks=None, graph_batch=None, batch_size=1):
        """
        Compute log probability and entropy of given actions.

        :param obs: (torch.Tensor) observation inputs into network.
        :param cent_obs: global state input for SSN, shoudl be disconnected during execution 
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """

        if cent_obs is not None:
            cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if self.args.tensor_obs:
            x_per_stat, x_act_stat, x_per_obs = self.get_states_obs_from_tensor(obs)
            x_per_stat, x_act_stat, x_per_obs = check(x_per_stat).to(**self.tpdv), check(x_act_stat).to(**self.tpdv), \
                                                check(x_per_obs).to(**self.tpdv)

            state_per_stat = self.prepro_stat(x_per_stat.reshape(-1, *x_per_stat.shape[2:]))
            if self.args.num_A != 0:
                state_act_stat = self.prepro_stat(x_act_stat.reshape(-1, *x_act_stat.shape[2:]))

            x_per_obs = self.prepro_obs(x_per_obs.reshape(-1, *x_per_obs.shape[3:]))

        else:
            obs = check(obs).to(**self.tpdv)
            actor_features = self.base(obs)

        if self.args.use_LSTM:
            if self.args.tensor_obs:
                rnn_states = rnn_states.squeeze()
                rnn_obs = check(rnn_obs).to(**self.tpdv)
                rnn_obs = rnn_obs.squeeze()

                # Forward pass LSTM features associated with the state of the perception agents
                hidden_state, cell_state = torch.split(rnn_states, [self.args.num_P + self.args.num_A] * 2, dim=1)

                hidden_state_per_stat, cell_state_per_stat = hidden_state[:, :self.args.num_P, :].reshape(-1, 128), \
                                                             cell_state[:, :self.args.num_P, :].reshape(-1, 128)

                hidden_state_per_stat, cell_state_per_stat = self.f_module_stat(state_per_stat.squeeze(),
                                                                                (hidden_state_per_stat,
                                                                                 cell_state_per_stat))

                # Forward pass LSTM features associated with the state of the action agents
                if self.args.num_A != 0:
                    hidden_state_act_stat, cell_state_act_stat = hidden_state[:, self.args.num_P:, :].reshape(-1, 128), \
                                                                 cell_state[:, self.args.num_P:, :].reshape(-1, 128)

                    hidden_state_act_stat, cell_state_act_stat = self.f_module_stat(state_act_stat.squeeze(),
                                                                                    (hidden_state_act_stat,
                                                                                     cell_state_act_stat))

                # forward pass the LSTM features associated with the observations of the perception agents
                hidden_state_per_obs, cell_state_per_obs = torch.split(rnn_obs, [self.args.num_P] * 2, dim=1)

                hidden_state_per_obs, cell_state_per_obs = self.f_module_obs(x_per_obs.squeeze(),
                                                                             (hidden_state_per_obs.reshape(-1, 16),
                                                                              cell_state_per_obs.reshape(-1, 16)))

                feat_dict = {}
                # concat to 128 + 16 -> 144; the dimension of feat_dict is (batch_size*num_agents, 144)
                feat_dict['P'] = torch.cat([hidden_state_per_stat, hidden_state_per_obs], dim=1)
                if self.args.num_A != 0:
                    feat_dict['A'] = hidden_state_act_stat

            else:
                actor_features, rnn_states = self.rnn(actor_features, rnn_states)
        else:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # need to figure what cent_obs/shared_obs should be?
        actor_features, state_features = self.gat(feat_dict, cent_obs, graph_batch, batch_size)
        actor_features = actor_features.reshape(actor_features.shape[0]*actor_features.shape[1], -1)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions,
                                                                   active_masks=active_masks
                                                                   if self._use_policy_active_masks else None)
        return action_log_probs, dist_entropy, state_features

    
class GAT_Critic(nn.Module):
    """ Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO)
    or local observations (IPPO).

    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(GAT_Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart

        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            if args.use_LSTM:
                # self.rnn = LSTMLayer(args, self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
                self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            else:
                self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        critic_features = self.base(cent_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            if self.args.use_LSTM:
                critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
            else:
                critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        values = self.v_out(critic_features)

        return values, rnn_states
