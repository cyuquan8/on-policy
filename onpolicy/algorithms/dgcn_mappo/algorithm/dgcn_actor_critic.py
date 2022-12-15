import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from onpolicy.algorithms.utils.util import init, check, complete_graph_edge_index
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.single_act import SingleACTLayer
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
        self.somu_num_layers = args.somu_num_layers
        self.scmu_num_layers = args.scmu_num_layers
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
        self.somu_lstm_list = [nn.LSTM(input_size=self.obs_dims, hidden_size=self.somu_lstm_hidden_size, num_layers=self.somu_num_layers, batch_first=True).to(device) for _ in range(self.num_agents)]

        # list of lstms for self communication memory unit (scmu) for each agent
        # somu_lstm_input_size is the last layer of dgcn layer
        self.scmu_lstm_list = [nn.LSTM(input_size=self.obs_dims, hidden_size=self.scmu_lstm_hidden_size, num_layers=self.scmu_num_layers, batch_first=True).to(device) for _ in range(self.num_agents)]

        # multi-head self attention layer for somu and scmu to selectively choose between the lstms outputs
        self.somu_multi_att_layer_list = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.somu_lstm_hidden_size, num_heads=self.somu_multi_att_num_heads, dropout=0, batch_first=True, device=device) 
                                                        for _ in range(self.num_agents)])
        self.scmu_multi_att_layer_list = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.scmu_lstm_hidden_size, num_heads=self.scmu_multi_att_num_heads, dropout=0, batch_first=True, device=device) 
                                                        for _ in range(self.num_agents)])

        # hidden fc layers for to generate actions for each agent
        # input channels are observations + concatenated outputs of somu_multi_att_layer and scmu_multi_att_layer and last layer of dgcn
        # fc_output_dims is the list of sizes of output channels fc_block
        self.actor_fc_layers_list = nn.ModuleList([NNLayers(input_channels=self.obs_dims + self.obs_dims + self.somu_num_layers * self.somu_lstm_hidden_size + self.scmu_num_layers * self.scmu_lstm_hidden_size, block=MLPBlock,
                                                   output_channels=[self.actor_fc_output_dims for i in range(self.n_actor_layers)], activation_func='relu', dropout_p=0, weight_initialisation="default") 
                                                   for _ in range(self.num_agents)]).to(device)

        # final action layer for each agent
        self.act_list = nn.ModuleList([SingleACTLayer(action_space, self.actor_fc_output_dims, self._use_orthogonal, self._gain) for _ in range(self.num_agents)]).to(device)
        
        self.to(device)
        
    def forward(self, obs, somu_hidden_states_actor, somu_cell_states_actor, scmu_hidden_states_actor, scmu_cell_states_actor, available_actions=None, deterministic=False, knn=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param somu_hidden_states_actor: (np.ndarray / torch.Tensor) hidden states for somu network.
        :param somu_cell_states_actor: (np.ndarray / torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_actor: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_actor: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.
        :param knn: (bool) whether to use k nearest neighbour to set up edge index.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        if knn:
            raise NotImplementedError
        else:
            # obtain edge index
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
            obs = check(obs).to(**self.tpdv) # shape: (batch_size, num_agents, obs_dims)   
            batch_size = obs.shape[0]
            obs_gnn = Batch.from_data_list([Data(x=obs[i, :, :], edge_index=edge_index) for i in range(batch_size)]).to(self.device)
        somu_hidden_states_actor = check(somu_hidden_states_actor).to(**self.tpdv) # shape: (batch_size, num_agents, somu_num_layers, somu_lstm_hidden_size)
        somu_cell_states_actor = check(somu_cell_states_actor).to(**self.tpdv) # shape: (batch_size, num_agents, somu_num_layers, somu_lstm_hidden_size)
        scmu_hidden_states_actor = check(scmu_hidden_states_actor).to(**self.tpdv) # shape: (batch_size, num_agents, scmu_num_layers, scmu_lstm_hidden_size)
        scmu_cell_states_actor = check(scmu_cell_states_actor).to(**self.tpdv) # shape: (batch_size, num_agents, scmu_num_layers, scmu_lstm_hidden_size)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv) # shape: (batch_size, num_agents, action_space_dim)
        # store somu, scmu, actions and action_log_probs
        somu_lstm_hidden_state_list = []
        somu_lstm_cell_state_list = []
        scmu_lstm_hidden_state_list = []
        scmu_lstm_cell_state_list = []
        actions_list = []
        action_log_probs_list = []
       
        # obs_gnn.x [shape: (batch_size * num_agents, obs_dims)] --> dgcn_layers [shape: (batch_size, num_agents, n_dgcn_layers + 1, obs_dims)]
        dgcn_output = self.dgcn_layers(x=obs_gnn.x, edge_index=obs_gnn.edge_index).view(batch_size, self.num_agents, self.n_dgcn_layers + 1, self.obs_dims)
       
        # iterate over agents 
        for i in range(self.num_agents):
            # obs[:, i, :].unsqueeze(dim=1) [shape: (batch_size, agent_index=sequence_length=1, obs_dims)], 
            # (h_0 [shape: (somu_num_layers, batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_num_layers, batch_size, somu_lstm_hidden_state)]) -->
            # somu_output [shape: (batch_size, agent_index=sequence_length=1, somu_lstm_hidden_size)], 
            # (h_n [shape: (somu_num_layers, batch_size, somu_lstm_hidden_state)], c_n [shape: (somu_num_layers, batch_size, somu_lstm_hidden_state)])
            somu_output, somu_hidden_cell_states_tup = self.somu_lstm_list[i](obs[:, i, :].unsqueeze(dim=1), (somu_hidden_states_actor.transpose(0, 2)[:, i, :, :].contiguous(), 
                                                                                                              somu_cell_states_actor.transpose(0, 2)[:, i, :, :].contiguous()))
            somu_lstm_hidden_state_list.append(somu_hidden_cell_states_tup[0].transpose(0, 1))
            somu_lstm_cell_state_list.append(somu_hidden_cell_states_tup[1].transpose(0, 1))
            # dgcn_output[:, i, -1, :].unsqueeze(dim=1) [shape: (batch_size, agent_index=sequence_length=1, obs_dims)] (last layer of dgcn for given agent), 
            # (h_0 [shape: (scmu_num_layers, batch_size, scmu_lstm_hidden_state)], c_0 [shape: (scmu_num_layers, batch_size, scmu_lstm_hidden_state)]) -->
            # scmu_output [shape: (batch_size, agent_index=sequence_length=1, scmu_lstm_hidden_state)], 
            # (h_n [shape: (scmu_num_layers, batch_size, scmu_lstm_hidden_state)], c_n [shape: (scmu_num_layers, batch_size, scmu_lstm_hidden_state)])
            scmu_output, scmu_hidden_cell_states_tup = self.scmu_lstm_list[i](dgcn_output[:, i, -1, :].unsqueeze(dim=1), (scmu_hidden_states_actor.transpose(0, 2)[:, i, :, :].contiguous(), 
                                                                                                                          scmu_cell_states_actor.transpose(0, 2)[:, i, :, :].contiguous()))
            scmu_lstm_hidden_state_list.append(scmu_hidden_cell_states_tup[0].transpose(0, 1))
            scmu_lstm_cell_state_list.append(scmu_hidden_cell_states_tup[1].transpose(0, 1))
            # somu_hidden_state / scmu_hidden_state [shape: (batch_size, somu_num_layers / scmu_num_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)] 
            # --> multihead self attention [shape: (batch_size, somu_num_layers / scmu_num_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
            somu_output = self.somu_multi_att_layer_list[i](somu_lstm_hidden_state_list[i], somu_lstm_hidden_state_list[i], somu_lstm_hidden_state_list[i])[0]
            scmu_output = self.somu_multi_att_layer_list[i](scmu_lstm_hidden_state_list[i], scmu_lstm_hidden_state_list[i], scmu_lstm_hidden_state_list[i])[0]

            # concatenate obs and outputs from dgcn, somu and scmu [shape: (batch_size, obs_dims + obs_dims + num_somu_lstm * somu_lstm_hidden_size + num_scmu_lstm * scmu_lstm_hidden_size)]
            output = torch.cat((obs[:, i, :], dgcn_output[:, i, -1, :], somu_output.reshape(batch_size, -1), scmu_output.reshape(batch_size, -1)), dim=-1)
            # output [shape: (batch_size, obs_dims + obs_dims + num_somu_lstm * somu_lstm_hidden_size + num_scmu_lstm * scmu_lstm_hidden_size)] --> actor_fc_layers [shape: (batch_size, actor_fc_output_dims)]
            output = self.actor_fc_layers_list[i](output)
            # actor_fc_layers --> act [shape: (batch_size, action_space_dim)]
            actions, action_log_probs = self.act_list[i](output, available_actions[:, i, :] if available_actions is not None else None, deterministic)
            # append actions and action_log_probs to respective lists
            actions_list.append(actions)
            action_log_probs_list.append(action_log_probs)
       
        # [shape: (batch_size, num_agents, action_space_dim)], [shape: (batch_size, num_agents, somu_num_layers / scmu_num_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
        return torch.stack(actions_list, dim=0).transpose(0, 1), torch.stack(action_log_probs_list, dim=0).transpose(0, 1), torch.stack(somu_lstm_hidden_state_list, dim=0).transpose(0, 1), \
               torch.stack(somu_lstm_cell_state_list, dim=0).transpose(0, 1), torch.stack(scmu_lstm_hidden_state_list, dim=0).transpose(0, 1), torch.stack(scmu_lstm_cell_state_list, dim=0).transpose(0, 1)

    def evaluate_actions(self, obs, somu_hidden_states_actor, somu_cell_states_actor, scmu_hidden_states_actor, scmu_cell_states_actor, action, available_actions=None, active_masks=None, knn=False):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param somu_hidden_states_actor: (torch.Tensor) hidden states for somu network.
        :param somu_cell_states_actor: (torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_actor: (torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_actor: (torch.Tensor) hidden states for scmu network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        :param knn: (bool) whether to use k nearest neighbour to set up edge index.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if knn:
            raise NotImplementedError
        else:
            # obtain edge index
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
            obs = check(obs).to(**self.tpdv) # shape: (batch_size * num_agents, obs_dims)   
            obs = obs.view(-1, self.num_agents, self.obs_dims) # shape: (batch_size, num_agents, obs_dims)
            batch_size = obs.shape[0]
            obs_gnn = Batch.from_data_list([Data(x=obs[i, :, :], edge_index=edge_index) for i in range(batch_size)]).to(self.device)
        action = check(action).to(**self.tpdv) # shape: (batch_size * num_agents, action_space_dim)
        somu_hidden_states_actor = check(somu_hidden_states_actor).to(**self.tpdv) # shape: (batch_size * num_agents, somu_num_layers, somu_lstm_hidden_size)
        somu_cell_states_actor = check(somu_cell_states_actor).to(**self.tpdv) # shape: (batch_size * num_agents, somu_num_layers, somu_lstm_hidden_size)
        scmu_hidden_states_actor = check(scmu_hidden_states_actor).to(**self.tpdv) # shape: (batch_size * num_agents, scmu_num_layers, scmu_lstm_hidden_size)
        scmu_cell_states_actor = check(scmu_cell_states_actor).to(**self.tpdv) # shape: (batch_size * num_agents, scmu_num_layers, scmu_lstm_hidden_size)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv) # shape: (batch_size * num_agents, action_space_dim)
            available_actions = available_actions.view(batch_size, self.num_agents, -1) # shape: (batch_size, num_agents, action_space_dim)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv) # shape: (batch_size * num_agents, 1)
            active_masks = active_masks.view(batch_size, self.num_agents, -1) # shape: (batch_size, num_agents, 1)
        action = action = action.view(batch_size, self.num_agents, -1) # shape: (batch_size, num_agents, action_space_dim)
        somu_hidden_states_actor = somu_hidden_states_actor.view(batch_size, self.num_agents, self.somu_num_layers, self.somu_lstm_hidden_size) # shape: (batch_size, num_agents, somu_num_layers, somu_lstm_hidden_size)
        somu_cell_states_actor = somu_cell_states_actor.view(batch_size, self.num_agents, self.somu_num_layers, self.somu_lstm_hidden_size) # shape: (batch_size, num_agents, somu_num_layers, somu_lstm_hidden_size)
        scmu_hidden_states_actor = scmu_hidden_states_actor.view(batch_size, self.num_agents, self.scmu_num_layers, self.scmu_lstm_hidden_size) # shape: (batch_size, num_agents, scmu_num_layers, scmu_lstm_hidden_size)
        scmu_cell_states_actor = scmu_cell_states_actor.view(batch_size, self.num_agents, self.scmu_num_layers, self.scmu_lstm_hidden_size) # shape: (batch_size, num_agents, scmu_num_layers, scmu_lstm_hidden_size)
        # store actions and actions_log_probs
        action_log_probs_list = []
        dist_entropy_list = []

        # obs_gnn.x [shape: (batch_size * num_agents, obs_dims)] --> dgcn_layers [shape: (batch_size, num_agents, n_dgcn_layers + 1, obs_dims)]
        dgcn_output = self.dgcn_layers(x=obs_gnn.x, edge_index=obs_gnn.edge_index).view(batch_size, self.num_agents, self.n_dgcn_layers + 1, self.obs_dims)

         # iterate over agents 
        for i in range(self.num_agents):
            # obs[:, i, :].unsqueeze(dim=1) [shape: (batch_size, agent_index=sequence_length=1, obs_dims)], 
            # (h_0 [shape: (somu_num_layers, batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_num_layers, batch_size, somu_lstm_hidden_state)]) -->
            # somu_output [shape: (batch_size, agent_index=sequence_length=1, somu_lstm_hidden_size)], 
            # (h_n [shape: (somu_num_layers, batch_size, somu_lstm_hidden_state)], c_n [shape: (somu_num_layers, batch_size, somu_lstm_hidden_state)])
            somu_output, somu_hidden_cell_states_tup = self.somu_lstm_list[i](obs[:, i, :].unsqueeze(dim=1), (somu_hidden_states_actor.transpose(0, 2)[:, i, :, :].contiguous(), 
                                                                                                              somu_cell_states_actor.transpose(0, 2)[:, i, :, :].contiguous()))
            # dgcn_output[:, i, -1, :].unsqueeze(dim=1) [shape: (batch_size, agent_index=sequence_length=1, obs_dims)] (last layer of dgcn for given agent), 
            # (h_0 [shape: (scmu_num_layers, batch_size, scmu_lstm_hidden_state)], c_0 [shape: (scmu_num_layers, batch_size, scmu_lstm_hidden_state)]) -->
            # scmu_output [shape: (batch_size, agent_index=sequence_length=1, scmu_lstm_hidden_state)], 
            # (h_n [shape: (scmu_num_layers, batch_size, scmu_lstm_hidden_state)], c_n [shape: (scmu_num_layers, batch_size, scmu_lstm_hidden_state)])
            scmu_output, scmu_hidden_cell_states_tup = self.scmu_lstm_list[i](dgcn_output[:, i, -1, :].unsqueeze(dim=1), (scmu_hidden_states_actor.transpose(0, 2)[:, i, :, :].contiguous(), 
                                                                                                                          scmu_cell_states_actor.transpose(0, 2)[:, i, :, :].contiguous()))
            # somu_hidden_state.transpose(0, 1) / scmu_hidden_state.transpose(0, 1) [shape: (batch_size, somu_num_layers / scmu_num_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)] 
            # --> multihead self attention [shape: (batch_size, somu_num_layers / scmu_num_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
            somu_output = self.somu_multi_att_layer_list[i](somu_hidden_cell_states_tup[0].transpose(0, 1), somu_hidden_cell_states_tup[0].transpose(0, 1), somu_hidden_cell_states_tup[0].transpose(0, 1))[0]
            scmu_output = self.somu_multi_att_layer_list[i](scmu_hidden_cell_states_tup[0].transpose(0, 1), scmu_hidden_cell_states_tup[0].transpose(0, 1), scmu_hidden_cell_states_tup[0].transpose(0, 1))[0]

            # concatenate obs and outputs from dgcn, somu and scmu [shape: (batch_size, obs_dims + obs_dims + num_somu_lstm * somu_lstm_hidden_size + num_scmu_lstm * scmu_lstm_hidden_size)]
            output = torch.cat((obs[:, i, :], dgcn_output[:, i, -1, :], somu_output.reshape(batch_size, -1), scmu_output.reshape(batch_size, -1)), dim=-1)
            # output [shape: (batch_size, obs_dims + obs_dims + num_somu_lstm * somu_lstm_hidden_size + num_scmu_lstm * scmu_lstm_hidden_size)] --> actor_fc_layers [shape: (batch_size, actor_fc_output_dims)]
            output = self.actor_fc_layers_list[i](output)
            # actor_fc_layers --> act [shape: (batch_size, action_space_dim)], [shape: () == scalar]
            action_log_probs, dist_entropy = self.act_list[i].evaluate_actions(output, action[:, i, :], available_actions[:, i, :] if available_actions is not None else None, 
                                                                               active_masks[:, i, :] if self._use_policy_active_masks and active_masks is not None else None)
            # append action_log_probs and dist_entropy to respective lists
            action_log_probs_list.append(action_log_probs)
            dist_entropy_list.append(dist_entropy)

        # [shape: (batch_size * num_agents, action_space_dim)] and [shape: () == scalar]
        return torch.stack(action_log_probs_list, dim=0).transpose(0, 1).reshape(batch_size * self.num_agents, -1), torch.stack(dist_entropy_list, dim=0).mean()

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