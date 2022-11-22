import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check, complete_graph_edge_index
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.nn import DGCNLayers, MLPBlock, NNLayers, DGCNBlock
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy

class DGCNActor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """ 
        class constructor for attributes for the actor model 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device

        self.num_agents = args.num_agents
        self.n_rollout_threads = args.n_rollout_threads
        self.n_dgcn_layers = args.n_dgcn_layers
        self.num_somu_lstm = args.num_somu_lstm
        self.num_scmu_lstm = args.num_scmu_lstm
        self.somu_lstm_hidden_size = args.somu_lstm_hidden_size
        self.scmu_lstm_hidden_size = args.scmu_lstm_hidden_size
        self.somu_multi_att_num_heads = args.somu_multi_att_num_heads
        self.scmu_multi_att_num_heads = args.scmu_multi_att_num_heads
        self.actor_fc_output_dims = args.actor_fc_output_dims
        self.n_actor_layers = args.n_actor_layers

        obs_shape = get_shape_from_obs_space(obs_space)
        if isinstance(obs_shape, list):
            self.obs_dims = obs_shape[0]
        else:
            self.obs_dims = obs_shape

        # model architecture for mappo dgcn actor

        # dgcn layers

        self.dgcn_layers = DGCNLayers(input_channels=self.obs_dims, block=DGCNBlock, output_channels=[self.obs_dims for i in range(self.n_dgcn_layers)], concat=False, activation_func="relu", weight_initialisation="default")

        # list of lstms for self observation memory unit (somu) for each agent
        # somu_lstm_input_size is the dimension of the observations
        self.somu_lstm_list = [nn.ModuleList([nn.LSTM(input_size=self.obs_dims, hidden_size=self.somu_lstm_hidden_size, num_layers=1, batch_first=True, dropout=0).to(device) 
                               for _ in range(self.num_somu_lstm)]) for _ in range(self.num_agents)]

        # list of lstms for self communication memory unit (scmu) for each agent
        # somu_lstm_input_size is the last layer of dgcn layer
        self.scmu_lstm_list = [nn.ModuleList([nn.LSTM(input_size=self.obs_dims, hidden_size=self.scmu_lstm_hidden_size, num_layers=1, batch_first=True, dropout=0).to(device) 
                               for _ in range(self.num_scmu_lstm)]) for _ in range(self.num_agents)]

        # weights to generate query, key and value for somu and scmu for each agent
        self.somu_query_layer_list = nn.ModuleList([MLPBlock(input_shape=self.somu_lstm_hidden_size, output_shape=self.somu_lstm_hidden_size, activation_func="relu", dropout_p= 0, weight_initialisation="default") 
                                                   for _ in range(self.num_agents)]).to(device)  
        self.somu_key_layer_list = nn.ModuleList([MLPBlock(input_shape=self.somu_lstm_hidden_size, output_shape=self.somu_lstm_hidden_size, activation_func="relu", dropout_p=0, weight_initialisation="default") 
                                                 for _ in range(self.num_agents)]).to(device) 
        self.somu_value_layer_list = nn.ModuleList([MLPBlock(input_shape=self.somu_lstm_hidden_size, output_shape=self.somu_lstm_hidden_size, activation_func="relu", dropout_p=0, weight_initialisation="default") 
                                                   for _ in range(self.num_agents)]).to(device)
        self.scmu_query_layer_list = nn.ModuleList([MLPBlock(input_shape=self.scmu_lstm_hidden_size, output_shape=self.scmu_lstm_hidden_size, activation_func="relu", dropout_p=0, weight_initialisation="default")
                                                   for _ in range(self.num_agents)]).to(device)
        self.scmu_key_layer_list = nn.ModuleList([MLPBlock(input_shape=self.scmu_lstm_hidden_size, output_shape=self.scmu_lstm_hidden_size, activation_func="relu", dropout_p=0, weight_initialisation="default") 
                                                 for _ in range(self.num_agents)]).to(device)
        self.scmu_value_layer_list = nn.ModuleList([MLPBlock(input_shape=self.scmu_lstm_hidden_size, output_shape=self.scmu_lstm_hidden_size, activation_func="relu", dropout_p=0, weight_initialisation="default") 
                                                   for _ in range(self.num_agents)]).to(device)

        # multi-head self attention layer for somu and scmu to selectively choose between the lstms outputs
        self.somu_multi_att_layer_list = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.somu_lstm_hidden_size, num_heads=self.somu_multi_att_num_heads, dropout=0, batch_first=True) for _ in range(self.num_agents)]).to(device)
        self.scmu_multi_att_layer_list = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.scmu_lstm_hidden_size, num_heads=self.scmu_multi_att_num_heads, dropout=0, batch_first=True) for _ in range(self.num_agents)]).to(device)

        # hidden fc layers for to generate actions for each agent
        # input channels are observations + concatenated outputs of somu_multi_att_layer and scmu_multi_att_layer and last layer of dgcn
        # fc_output_dims is the list of sizes of output channels fc_block
        self.actor_fc_layers_list = nn.ModuleList([NNLayers(input_channels=self.obs_dims + self.num_somu_lstm * self.somu_lstm_hidden_size + self.num_scmu_lstm * self.scmu_lstm_hidden_size + self.obs_dims, block=MLPBlock,
                                                   output_channels=[self.actor_fc_output_dims for i in range(self.n_actor_layers)], activation_func='relu', dropout_p=0, weight_initialisation="default") 
                                                   for _ in range(self.num_agents)]).to(device)

        # final action layer for each agent
        self.act_list = nn.ModuleList([ACTLayer(action_space, self.actor_fc_output_dims, self._use_orthogonal, self._gain) for _ in range(self.num_agents)]).to(device)
        
        self.to(device)
        
    def forward(self, obs, available_actions=None, deterministic=False, knn=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.
        :param knn: (bool) whether to use k nearest neighbour to set up edge index.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            available_actions = available_actions.view(self.n_rollout_threads, self.num_agents, -1)
        # obtain reshaped observation 
        obs = obs.view(self.n_rollout_threads, self.num_agents, -1)
        # store actions and actions_log_probs per env. shape([n_rollout_threads, num_agents, action_dims])
        actions_list = [[] for _ in range(self.n_rollout_threads)]
        action_log_probs_list = [[] for _ in range(self.n_rollout_threads)]

        # iterate over number of env rollouts
        for i in range(self.n_rollout_threads):
            if knn:
                raise NotImplementedError
            else:
                # obtain edge index
                edge_index = complete_graph_edge_index(self.num_agents) 
                edge_index = torch.tensor(edge_index, dtype = torch.long, device=self.device).t().contiguous()

            # observation per env (shape: [num_agents, obs_dims])
            obs_env = obs[i]
            # obs_env --> dgcn_layers (shape: [num_agents, num_layers, obs_dims])
            dgcn_output = self.dgcn_layers(obs_env, edge_index)

            # iterate over agents 
            for j in range(self.num_agents):
                # empty list to store ouputs for somu and scmu
                somu_lstm_output_list = []
                scmu_lstm_output_list = []
                # iterate over each somu_lstm in somu_lstm_list
                for k in range(self.num_somu_lstm):
                    # observation per env per agent (shape: [1, obs_dims])
                    obs_env_agent = torch.unsqueeze(obs_env[j], dim=0)
                    # obs_env_agent --> somu_lstm (shape: [1, somu_lstm_hidden_size])
                    somu_lstm_output, _ = self.somu_lstm_list[j][k](obs_env_agent)
                    somu_lstm_output_list.append(somu_lstm_output)
                # iterate over each somu lstm in scmu_lstm_list
                for k in range(self.num_scmu_lstm):
                    # last layer of dgcn_output per agent (shape: [1, obs_dims])
                    dgcn_output_agent = torch.unsqueeze(dgcn_output[j, -1, :], dim=0)
                    # dgcn_output_agent --> scmu_lstm (shape: [1, scmu_lstm_hidden_size])
                    scmu_lstm_output, _ = self.scmu_lstm_list[j][k](dgcn_output_agent)
                    scmu_lstm_output_list.append(scmu_lstm_output)

                # concatenate lstm ouput based on number of lstms (shape: [num_somu_lstm / num_scmu_lstm, somu_lstm_hidden_size / scmu_lstm_hidden_size])
                somu_lstm_output = torch.cat(somu_lstm_output_list, dim=0)
                scmu_lstm_output = torch.cat(scmu_lstm_output_list, dim=0)

                # obtain query, key and value for somu_lstm and scmu_lstm_outputs (shape: [num_somu_lstm / num_scmu_lstm, somu_lstm_hidden_size / scmu_lstm_hidden_size])
                q_somu = self.somu_query_layer_list[i](somu_lstm_output)
                k_somu = self.somu_key_layer_list[i](somu_lstm_output)
                v_somu = self.somu_value_layer_list[i](somu_lstm_output)
                q_scmu = self.scmu_query_layer_list[i](scmu_lstm_output)
                k_scmu = self.scmu_key_layer_list[i](scmu_lstm_output)
                v_scmu = self.scmu_value_layer_list[i](scmu_lstm_output)

                # q, k, v --> multihead attention (shape: [num_somu_lstm / num_scmu_lstm, somu_lstm_hidden_size / scmu_lstm_hidden_size])
                somu_multi_att_output = self.somu_multi_att_layer_list[j](q_somu, k_somu, v_somu)[0] 
                scmu_multi_att_output = self.somu_multi_att_layer_list[j](q_scmu, k_scmu, v_scmu)[0]

                # reshape output (shape: [1, (num_somu_lstm / num_scmu_lstm) * (somu_lstm_hidden_size / scmu_lstm_hidden_size)])
                somu_output = somu_multi_att_output.view(1, -1)
                scmu_output = scmu_multi_att_output.view(1, -1)

                # concatenate outputs from dgcn, somu and scmu (shape: [1, obs_dims + num_somu_lstm * somu_lstm_hidden_size + num_scmu_lstm * scmu_lstm_hidden_size + obs_dims])
                output = torch.cat((obs_env_agent, dgcn_output_agent, somu_output, scmu_output), dim=-1)
                # output --> actor_fc_layers (shape: [1, actor_fc_output_dims])
                output = self.actor_fc_layers_list[j](output)
                # actor_fc_layers --> act (shape: [1, action_space_dim])
                actions, action_log_probs = self.act_list[j](output, torch.unsqueeze(available_actions[i, j], dim=0) if available_actions is not None else None, deterministic)
                # append to actions_list and action_log_probs_list
                actions_list[i].append(actions)
                action_log_probs_list[i].append(action_log_probs)

            # concatenate across agents (shape: [num_agents, action_space_dim])
            actions_list[i] = torch.cat(actions_list[i], dim=0)
            action_log_probs_list[i] = torch.cat(action_log_probs_list[i], dim=0)

        # (shape: [n_rollout_threads * num_agents, action_space_dim])
        return torch.cat(actions_list, dim=0), torch.cat(action_log_probs_list, dim=0)

    def evaluate_actions(self, obs, action, available_actions=None, active_masks=None, knn=False):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        :param knn: (bool) whether to use k nearest neighbour to set up edge index.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            available_actions = available_actions.view(self.n_rollout_threads, self.num_agents, -1)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
            active_masks = active_masks.view(self.n_rollout_threads, self.num_agents, -1)
        # obtain reshaped observation and actions 
        obs = obs.view(self.n_rollout_threads, self.num_agents, -1)
        action = action.view(self.n_rollout_threads, self.num_agents, -1)
        # store actions and actions_log_probs per env. shape([n_rollout_threads, num_agents, action_dims])
        action_log_probs_list = [[] for _ in range(self.n_rollout_threads)]
        dist_entropy_list = [[] for _ in range(self.n_rollout_threads)]

        # iterate over number of env rollouts
        for i in range(self.n_rollout_threads):
            if knn:
                raise NotImplementedError
            else:
                # obtain edge index
                edge_index = complete_graph_edge_index(self.num_agents) 
                edge_index = torch.tensor(edge_index, dtype = torch.long).t().contiguous()

            # observation per env (shape: [num_agents, obs_dims])
            obs_env = obs[i]
            # obs_env --> dgcn_layers (shape: [num_agents, num_layers, obs_dims])
            dgcn_output = self.dgcn_layers(obs_env, edge_index)

            # iterate over agents 
            for j in range(self.num_agents):
                # empty list to store ouputs for somu and scmu
                somu_lstm_output_list = []
                scmu_lstm_output_list = []
                # iterate over each somu_lstm in somu_lstm_list
                for k in range(self.num_somu_lstm):
                    # observation per env per agent (shape: [1, obs_dims])
                    obs_env_agent = torch.unsqueeze(obs_env[j], dim=0)
                    # obs_env_agent --> somu_lstm (shape: [1, somu_lstm_hidden_size])
                    somu_lstm_output, _ = self.somu_lstm_list[j][k](obs_env_agent)
                    somu_lstm_output_list.append(somu_lstm_output)
                # iterate over each somu lstm in scmu_lstm_list
                for k in range(self.num_scmu_lstm):
                    # last layer of dgcn_output per agent (shape: [1, obs_dims])
                    dgcn_output_agent = torch.unsqueeze(dgcn_output[j, -1, :], dim=0)
                    # dgcn_output_agent --> scmu_lstm (shape: [1, scmu_lstm_hidden_size])
                    scmu_lstm_output, _ = self.scmu_lstm_list[j][k](dgcn_output_agent)
                    scmu_lstm_output_list.append(scmu_lstm_output)

                # concatenate lstm ouput based on number of lstms (shape: [num_somu_lstm / num_scmu_lstm, somu_lstm_hidden_size / scmu_lstm_hidden_size])
                somu_lstm_output = torch.cat(somu_lstm_output_list, dim=0)
                scmu_lstm_output = torch.cat(scmu_lstm_output_list, dim=0)

                # obtain query, key and value for somu_lstm and scmu_lstm_outputs (shape: [num_somu_lstm / num_scmu_lstm, somu_lstm_hidden_size / scmu_lstm_hidden_size])
                q_somu = self.somu_query_layer_list[i](somu_lstm_output)
                k_somu = self.somu_key_layer_list[i](somu_lstm_output)
                v_somu = self.somu_value_layer_list[i](somu_lstm_output)
                q_scmu = self.scmu_query_layer_list[i](scmu_lstm_output)
                k_scmu = self.scmu_key_layer_list[i](scmu_lstm_output)
                v_scmu = self.scmu_value_layer_list[i](scmu_lstm_output)

                # q, k, v --> multihead attention (shape: [num_somu_lstm / num_scmu_lstm, somu_lstm_hidden_size / scmu_lstm_hidden_size])
                somu_multi_att_output = self.somu_multi_att_layer_list[j](q_somu, k_somu, v_somu)[0]  
                scmu_multi_att_output = self.somu_multi_att_layer_list[j](q_scmu, k_scmu, v_scmu)[0] 

                # reshape output (shape: [1, (num_somu_lstm / num_scmu_lstm) * (somu_lstm_hidden_size / scmu_lstm_hidden_size)])
                somu_output = somu_multi_att_output.view(1, -1)
                scmu_output = scmu_multi_att_output.view(1, -1)

                # concatenate outputs from dgcn, somu and scmu (shape: [1, obs_dims + num_somu_lstm * somu_lstm_hidden_size + num_scmu_lstm * scmu_lstm_hidden_size + obs_dims])
                output = torch.cat((obs_env_agent, dgcn_output_agent, somu_output, scmu_output), dim=-1)
                # output --> actor_fc_layers (shape: [1, actor_fc_output_dims])
                output = self.actor_fc_layers_list[j](output)
                # actor_fc_layers --> act (shape: [1, action_space_dim])
                action_log_probs, dist_entropy = self.act_list[j].evaluate_actions(output, self.actions[i, j], torch.unsqueeze(available_actions[i, j], dim=0) if available_actions is not None else None, 
                                                                                   active_masks = torch.unsqueeze(active_masks[i, j], dim=0) if self._use_policy_active_masks and active_masks is not None else None)
                # append to actions_list and action_log_probs_list
                action_log_probs_list[i].append(action_log_probs)
                dist_entropy_list[i].append(dist_entropy)

            # concatenate across agents (shape: [num_agents, action_space_dim])
            action_log_probs_list[i] = torch.cat(action_log_probs_list[i], dim=0)
            dist_entropy_list[i] = torch.cat(dist_entropy_list[i], dim=0)

        # (shape: [n_rollout_threads * num_agents, action_space_dim])
        return torch.cat(action_log_probs_list, dim=0), torch.cat(dist_entropy_list, dim=0)


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
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
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states