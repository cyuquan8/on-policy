import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.utils.nn import (
    MLPBlock, 
    NNLayers
)
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.magic_gain import GraphAttentionGAIN
from onpolicy.algorithms.utils.magic_gat import GraphAttentionGAT
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space


class MAGIC_Actor(nn.Module):
    """
    MAGIC Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(MAGIC_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.data_chunk_length = args.data_chunk_length
        self.num_agents = args.num_agents
        self.message_encoder = args.magic_message_encoder
        self.message_decoder = args.magic_message_decoder
        self.use_gat_encoder = args.magic_use_gat_encoder
        self.gat_encoder_out_size = args.magic_gat_encoder_out_size
        self.gat_encoder_num_heads = args.magic_gat_encoder_num_heads
        self.gat_encoder_normalize = args.magic_gat_encoder_normalize
        self.first_graph_complete = args.magic_first_graph_complete
        self.second_graph_complete = args.magic_second_graph_complete
        self.learn_second_graph = args.magic_learn_second_graph
        self.gat_hidden_size = args.magic_gat_hidden_size
        self.gat_num_heads = args.magic_gat_num_heads
        self.gat_num_heads_out = args.magic_gat_num_heads_out
        self.self_loop_type1 = args.magic_self_loop_type1
        self.self_loop_type2 = args.magic_self_loop_type2
        self.first_gat_normalize = args.magic_first_gat_normalize
        self.second_gat_normalize = args.magic_second_gat_normalize
        self.comm_init = args.magic_comm_init
        self.comm_mask_zero = args.magic_comm_mask_zero
        self.directed = args.magic_directed
        self.gat_architecture = args.magic_gat_architecture
        self.n_gnn_fc_layers = args.magic_n_gnn_fc_layers
        self.gnn_train_eps = args.magic_gnn_train_eps
        self.gnn_norm = args.magic_gnn_norm

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        assert self._recurrent_N == 1

        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            raise NotImplementedError("CNN-based observations not implemented for MAGIC")
        if isinstance(obs_shape, (list, tuple)):
            self.obs_dims = obs_shape[0]
        else:
            self.obs_dims = obs_shape
        self.act_dims = get_shape_from_act_space(action_space)

        # encoder
        self.obs_encoder = nn.Linear(self.obs_dims, self.hidden_size)
        # lstm cell
        self.lstm_cell= nn.LSTMCell(self.hidden_size, self.hidden_size)
        # message encoder
        if self.message_encoder:
            self.msg_encoder = nn.Linear(self.hidden_size, self.hidden_size)
        # gat encoder for scheduler
        if self.use_gat_encoder:
            if self.gat_architecture == 'gat':
                self.gat_encoder = GraphAttentionGAT(
                    in_features=self.hidden_size, 
                    out_features=self.gat_encoder_out_size, 
                    dropout=0, 
                    negative_slope=0.2,
                    num_agents=self.num_agents, 
                    num_heads=self.gat_encoder_num_heads, 
                    self_loop_type=1, 
                    average=True, 
                    normalize=self.gat_encoder_normalize,
                    device=device
                )
            elif self.gat_architecture == 'gain':
                self.gat_encoder = GraphAttentionGAIN(
                    in_features=self.hidden_size,  
                    out_features=self.gat_encoder_out_size, 
                    dropout=0, 
                    negative_slope=0.2, 
                    num_agents=self.num_agents,
                    n_gnn_fc_layers=self.n_gnn_fc_layers, 
                    num_heads=self.gat_encoder_num_heads,
                    eps=1., 
                    gnn_train_eps=self.gnn_train_eps,
                    gnn_norm=self.gnn_norm,
                    self_loop_type=1,
                    average=True, 
                    normalize=self.gat_encoder_normalize,
                    device=device
                )
        # sub-schedulers
        if not self.first_graph_complete:
            if self.use_gat_encoder:
                self.sub_scheduler_mlp1 = NNLayers(
                    input_channels=self.gat_encoder_out_size * 2, 
                    block=MLPBlock, 
                    output_channels=[self.gat_encoder_out_size // 2, self.gat_encoder_out_size // 2, 2],
                    norm_type='none', 
                    activation_func='relu', 
                    dropout_p=0, 
                    weight_initialisation="default"
                )
            else:
                self.sub_scheduler_mlp1 = NNLayers(
                    input_channels=self.hidden_size * 2, 
                    block=MLPBlock, 
                    output_channels=[self.hidden_size // 2, self.hidden_size // 8, 2],
                    norm_type='none', 
                    activation_func='relu', 
                    dropout_p=0, 
                    weight_initialisation="default"
                )
        if self.learn_second_graph and not self.second_graph_complete:
            if self.use_gat_encoder:
                self.sub_scheduler_mlp2 = NNLayers(
                    input_channels=self.gat_encoder_out_size * 2, 
                    block=MLPBlock, 
                    output_channels=[self.gat_encoder_out_size // 2, self.gat_encoder_out_size // 2, 2],
                    norm_type='none', 
                    activation_func='relu', 
                    dropout_p=0, 
                    weight_initialisation="default"
                )
            else:
                self.sub_scheduler_mlp2 = NNLayers(
                    input_channels=self.hidden_size * 2, 
                    block=MLPBlock, 
                    output_channels=[self.hidden_size // 2, self.hidden_size // 8, 2],
                    norm_type='none', 
                    activation_func='relu', 
                    dropout_p=0, 
                    weight_initialisation="default"
                )
        # sub-processors
        if self.gat_architecture == 'gat':
            self.sub_processor1 = GraphAttentionGAT(
                in_features=self.hidden_size, 
                out_features=self.gat_hidden_size, 
                dropout=0, 
                negative_slope=0.2,
                num_agents=self.num_agents, 
                num_heads=self.gat_num_heads, 
                self_loop_type=self.self_loop_type1, 
                average=False, 
                normalize=self.first_gat_normalize,
                device=device
            )  
            self.sub_processor2 = GraphAttentionGAT(
                in_features=self.gat_hidden_size * self.gat_num_heads, 
                out_features=self.hidden_size, 
                dropout=0, 
                negative_slope=0.2,
                num_agents=self.num_agents, 
                num_heads=self.gat_num_heads_out, 
                self_loop_type=self.self_loop_type2, 
                average=True, 
                normalize=self.second_gat_normalize,
                device=device
            )
        elif self.gat_architecture == 'gain':
            self.sub_processor1 = GraphAttentionGAIN(
                in_features=self.hidden_size, 
                out_features=self.gat_hidden_size, 
                dropout=0, 
                negative_slope=0.2,
                num_agents=self.num_agents, 
                n_gnn_fc_layers=self.n_gnn_fc_layers, 
                num_heads=self.gat_num_heads,
                eps=1., 
                gnn_train_eps=self.gnn_train_eps,
                gnn_norm=self.gnn_norm, 
                self_loop_type=self.self_loop_type1, 
                average=False, 
                normalize=self.first_gat_normalize,
                device=device
            )
            self.sub_processor2 = GraphAttentionGAIN(
                in_features=self.gat_hidden_size * self.gat_num_heads, 
                out_features=self.hidden_size, 
                dropout=0, 
                negative_slope=0.2,
                num_agents=self.num_agents, 
                n_gnn_fc_layers=self.n_gnn_fc_layers, 
                num_heads=self.gat_num_heads_out,
                eps=1., 
                gnn_train_eps=self.gnn_train_eps,
                gnn_norm=self.gnn_norm, 
                self_loop_type=self.self_loop_type2, 
                average=True, 
                normalize=self.second_gat_normalize,
                device=device
            )
        # message decoder
        if self.message_decoder:
            self.msg_decoder = nn.Linear(self.hidden_size, self.hidden_size)
        # decoder
        self.act = ACTLayer(action_space, self.hidden_size * 2, self._use_orthogonal, self._gain)

        # initialize weights as 0
        if self.comm_init == 'zeros':
            def init_linear(self, m):
                """
                Function to initialize the parameters in nn.Linear as 0 
                """
                if type(m) == nn.Linear:
                    m.weight.data.fill_(0.)
                    m.bias.data.fill_(0.)
            if self.message_encoder:
                self.msg_encoder.weight.data.zero_()
            if self.message_decoder:
                self.msg_decoder.weight.data.zero_()
            if not self.first_graph_complete:
                self.sub_scheduler_mlp1.apply(init_linear)
            if self.learn_second_graph and not self.second_graph_complete:
                self.sub_scheduler_mlp2.apply(init_linear)

        self.to(device)

    def get_complete_graph(self, agent_mask):
        """
        Function to generate a complete graph, and mask it with agent_mask
        """
        # adjacency [shape: (batch_size, num_agents, num_agents)]
        adj = torch.ones(
            size=(agent_mask.shape[0], self.num_agents, self.num_agents), 
            dtype=self.tpdv['dtype'], 
            device=self.tpdv['device']
        )
        # [shape: (batch_size, num_agents, 1)] --> [shape: (batch_size, num_agents, num_agents)]
        agent_mask = agent_mask.expand(-1, -1, self.num_agents)
        # mask adjacency [shape: (batch_size, num_agents, num_agents)]
        agent_mask_transpose = agent_mask.transpose(1, 2)
        adj = adj * agent_mask * agent_mask_transpose
        
        return adj

    def sub_scheduler(self, sub_scheduler_mlp, hidden_state, agent_mask, directed=True):
        """
        Function to perform a sub-scheduler

        Arguments: 
            sub_scheduler_mlp (nn.Sequential): the MLP layers in a sub-scheduler
            hidden_state (tensor): the encoded messages input to the sub-scheduler (batch_size, num_agents, hidden_size)
            agent_mask (tensor): (batch_size, num_agents, 1)
            directed (bool): decide if generate directed graphs

        Return:
            adj (tensor): a adjacency matrix which is the communication graph (batch_size, num_agents, num_agents)
        """
        # [shape: batch_size, num_agents, 1, hidden_size] --> [shape: batch_size, num_agents, num_agents, hidden_size]
        hard_attn_input_1 = hidden_state.unsqueeze(2).repeat(1, 1, self.num_agents, 1)
        # [shape: batch_size, 1, num_agents, hidden_size] --> [shape: batch_size, num_agents, num_agents, hidden_size]
        hard_attn_input_2 = hidden_state.unsqueeze(1).repeat(1, self.num_agents, 1, 1)
        # [shape: batch_size, num_agents, num_agents, hidden_size * 2]
        hard_attn_input = torch.cat((hard_attn_input_1, hard_attn_input_2), dim=-1)
        # [shape: batch_size, num_agents, num_agents, 2]
        if directed:
            hard_attn_output = F.gumbel_softmax(sub_scheduler_mlp(hard_attn_input), hard=True)
        else:
            hard_attn_output = F.gumbel_softmax(
                0.5 * sub_scheduler_mlp(hard_attn_input) + 0.5 * sub_scheduler_mlp(hard_attn_input.permute(0, 2, 1, 3)), 
                hard=True
            )
        # [shape: batch_size, num_agents, num_agents, 1]
        hard_attn_output = torch.narrow(hard_attn_output, 3, 1, 1)
        # [shape: (batch_size, num_agents, 1)] --> [shape: (batch_size, num_agents, num_agents)]
        agent_mask = agent_mask.expand(-1, -1, self.num_agents)
        # mask adjacency [shape: (batch_size, num_agents, num_agents)]
        agent_mask_transpose = agent_mask.transpose(1, 2)
        adj = hard_attn_output.squeeze(-1) * agent_mask * agent_mask_transpose

        return adj

    def forward(self, obs, lstm_hidden_states, lstm_cell_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param lstm_hidden_states: (np.ndarray / torch.Tensor) hidden states for LSTM.
        :param lstm_cell_states: (np.ndarray / torch.Tensor) cell states for LSTM.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return lstm_hidden_states: (torch.Tensor) updated LSTM hidden states.
        :return lstm_cell_states: (torch.Tensor) updated LSTM cell states.
        """
        # [shape: (batch_size * num_agents, obs_dims)]
        obs = check(obs).to(**self.tpdv)
        batch_size = obs.shape[0] // self.num_agents
        # obtain batch, [shape: (batch_size * num_agents)]
        if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
            batch = torch.arange(batch_size).repeat_interleave(self.num_agents).to(self.tpdv['device'])
        # [shape: (batch_size * num_agents, recurrent_N = 1, hidden_size)] --> 
        # [shape: (batch_size * num_agents, hidden_size)]
        lstm_hidden_states = check(lstm_hidden_states).to(**self.tpdv).squeeze(1)
        lstm_cell_states = check(lstm_cell_states).to(**self.tpdv).squeeze(1)
        # [shape: (batch_size * num_agents, 1)]
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            # [shape: (batch_size * num_agents, act_dims)]
            available_actions = check(available_actions).to(**self.tpdv)

        # obs --> encoded_obs [shape: (batch_size * num_agents, hidden_size)]
        encoded_obs = self.obs_encoder(obs)
        # encoded_obs [shape: (batch_size * num_agents, hidden_size)]
        # lstm_hidden_states / lstm_cell_states [shape: (batch_size * num_agents, hidden_size)]
        # masks [shape: (batch_size * num_agents, 1)] --> 
        # lstm_hidden_states / lstm_cell_states [shape: (batch_size * num_agents, hidden_size)]
        lstm_hidden_states, lstm_cell_states = self.lstm_cell(
            encoded_obs, 
            (lstm_hidden_states * masks, lstm_cell_states * masks)
        )
        # message encoder lstm_hidden_states [shape: (batch_size * num_agents, hidden_size)] --> 
        # comm [shape: (batch_size * num_agents, hidden_size)]
        comm = lstm_hidden_states.clone()
        if self.message_encoder:
            comm = self.msg_encoder(comm)
        # agent_mask to block communication if desired [shape: (batch_size, num_agents, 1)]
        if self.comm_mask_zero:
            agent_mask = torch.zeros(
                size=(batch_size, self.num_agents, 1), 
                dtype=self.tpdv['dtype'], 
                device=self.tpdv['device']
            )
        else:
            agent_mask = torch.ones(
                size=(batch_size, self.num_agents, 1), 
                dtype=self.tpdv['dtype'], 
                device=self.tpdv['device']
            )
        # mask communcation from dead agents [shape: (batch_size, num_agents, hidden_size)]
        comm = comm.reshape(batch_size, self.num_agents, self.hidden_size)
        comm = comm * agent_mask
        comm_ori = comm.clone()

        # sub-scheduler 1
        if not self.first_graph_complete:
            if self.use_gat_encoder:
                # masked complete adjacency [shape: (batch_size, num_agents, num_agents)]
                adj_complete = self.get_complete_graph(agent_mask)
                # gat encoder [shape: batch_size, num_agents, gat_encoder_out_size]
                if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
                    encoded_state1 = self.gat_encoder(comm, adj_complete, batch)
                else:
                    encoded_state1 = self.gat_encoder(comm, adj_complete)
                # scheduled adjacency [shape: (batch_size, num_agents, num_agents)] 
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, encoded_state1, agent_mask, self.directed)
            else:
                # scheduled adjacency [shape: (batch_size, num_agents, num_agents)]
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask, self.directed)
        else:
            # masked complete adjacency [shape: (batch_size, num_agents, num_agents)]
            adj1 = self.get_complete_graph(agent_mask)
        # sub-processor 1 [shape: (batch_size, num_agents, hidden_size)] --> 
        # [shape: (batch_size, num_agents, gat_hidden_size * gat_num_heads)]
        if self.gat_architecture == "gain":
            if self.gnn_norm == "graphnorm":
                comm = self.sub_processor1(comm, adj1, batch)
            else:
                comm = self.sub_processor1(comm, adj1)
        else:
            comm = F.elu(self.sub_processor1(comm, adj1))

        # sub-scheduler 2
        if self.learn_second_graph and not self.second_graph_complete:
            if self.use_gat_encoder:
                if self.first_graph_complete:
                    # masked complete adjacency [shape: (batch_size, num_agents, num_agents)]
                    adj_complete = self.get_complete_graph(agent_mask)
                    # gat encoder [shape: batch_size, num_agents, gat_encoder_out_size]
                    if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
                        encoded_state2 = self.gat_encoder(comm_ori, adj_complete, batch)
                    else:
                        encoded_state2 = self.gat_encoder(comm_ori, adj_complete)
                else:
                    # reuse encoded state from sub-scheduler 1 [shape: batch_size, num_agents, gat_encoder_out_size]
                    encoded_state2 = encoded_state1
                # scheduled adjacency [shape: (batch_size, num_agents, num_agents)]
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, encoded_state2, agent_mask, self.directed)
            else:
                # scheduled adjacency [shape: (batch_size, num_agents, num_agents)]
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, comm_ori, agent_mask, self.directed)
        elif not self.learn_second_graph and not self.second_graph_complete:
            # reuse scheduled adjacency from sub-scheduler 1 [shape: (batch_size, num_agents, num_agents)]
            adj2 = adj1
        else:
            # masked complete adjacency [shape: (batch_size, num_agents, num_agents)]
            adj2 = self.get_complete_graph(agent_mask)
        # sub-processor 2 [shape: (batch_size, num_agents, gat_hidden_size * gat_num_heads)] --> 
        # [shape: (batch_size, num_agents, hidden_size)]
        if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
            comm = self.sub_processor2(comm, adj2, batch)
        else:
            comm = self.sub_processor2(comm, adj2)
        
        # mask communication to dead agents [shape: (batch_size, num_agents, hidden_size)]
        comm = comm * agent_mask
        # message decoder [shape: (batch_size, num_agents, hidden_size)]
        if self.message_decoder:
            comm = self.msg_decoder(comm)
        # [shape: (batch_size * num_agents, hidden_size * 2)] -->
        # actions, action_log_probs [shape: (batch_size * num_agents, act_dims)]
        actions, action_log_probs = self.act(
            torch.cat((lstm_hidden_states, comm.reshape(batch_size * self.num_agents, self.hidden_size)), dim=-1), 
            available_actions, 
            deterministic
        )

        # [shape: (batch_size * num_agents, act_dims)], [shape: (batch_size * num_agents, recurrent_N = 1, hidden_size)]
        return actions, action_log_probs, lstm_hidden_states.unsqueeze(1), lstm_cell_states.unsqueeze(1)

    def evaluate_actions(self, obs, lstm_hidden_states, lstm_cell_states, action, masks, available_actions=None, 
                         active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param lstm_hidden_states: (torch.Tensor) hidden states for LSTM.
        :param lstm_cell_states: (torch.Tensor) cell states for LSTM.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        # [shape: (mini_batch_size * num_agents * data_chunk_length, obs_dims)] --> 
        # [shape: (mini_batch_size * num_agents, data_chunk_length, obs_dims)]
        obs = check(obs).to(**self.tpdv)
        obs = obs.reshape(-1, self.data_chunk_length, self.obs_dims)
        mini_batch_size = obs.shape[0] // self.num_agents
        # obtain batch, [shape: (mini_batch_size * data_chunk_length * num_agents)]
        if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
            batch = torch.arange(mini_batch_size).repeat_interleave(self.num_agents).to(self.tpdv['device'])
        # [shape: (mini_batch_size * num_agents, recurrent_N = 1, hidden_size)] --> 
        # [shape: (mini_batch_size * num_agents, hidden_size)]
        lstm_hidden_states = check(lstm_hidden_states).to(**self.tpdv).squeeze(1)
        lstm_cell_states = check(lstm_cell_states).to(**self.tpdv).squeeze(1)
        # [shape: (mini_batch_size * num_agents * data_chunk_length, act_dims)] -->
        # [shape: (mini_batch_size * num_agents, data_chunk_length, act_dims)]
        action = check(action).to(**self.tpdv)
        action = action.reshape(mini_batch_size * self.num_agents, self.data_chunk_length, -1)
        # [shape: (mini_batch_size * num_agents * data_chunk_length, 1)] --> 
        # [shape: (mini_batch_size * num_agents, data_chunk_length, 1)]
        masks = check(masks).to(**self.tpdv)
        masks = masks.reshape(mini_batch_size * self.num_agents, self.data_chunk_length, -1)
        if available_actions is not None:
            # [shape: (mini_batch_size * num_agents * data_chunk_length, act_dims)] -->
            # [shape: (mini_batch_size * num_agents, data_chunk_length, act_dims)]
            available_actions = check(available_actions).to(**self.tpdv)
            available_actions = available_actions.reshape(mini_batch_size * self.num_agents, self.data_chunk_length, -1)
        if active_masks is not None:
            # [shape: (mini_batch_size * num_agents * data_chunk_length, 1)] -->
            # [shape: (mini_batch_size * num_agents, data_chunk_length, 1)]
            active_masks = check(active_masks).to(**self.tpdv)
            active_masks = active_masks.reshape(mini_batch_size * self.num_agents, self.data_chunk_length, -1)
        # store actions and actions_log_probs
        action_log_probs_list = []
        dist_entropy_list = []

        # obs --> encoded_obs [shape: (mini_batch_size * num_agents, data_chunk_length, hidden_size)]
        encoded_obs = self.obs_encoder(obs)

        # iterate over data_chunk_length 
        for j in range(self.data_chunk_length):
            # encoded_obs[:, j] [shape: (mini_batch_size * num_agents, hidden_size)]
            # lstm_hidden_states / lstm_cell_states [shape: (mini_batch_size * num_agents, hidden_size)]
            # masks [shape: (mini_batch_size * num_agents, 1)] --> 
            # lstm_hidden_states / lstm_cell_states [shape: (mini_batch_size * num_agents, hidden_size)]
            lstm_hidden_states, lstm_cell_states = self.lstm_cell(
                encoded_obs[:, j], 
                (lstm_hidden_states * masks[:, j], lstm_cell_states * masks[:, j])
            )
            # message encoder lstm_hidden_states [shape: (mini_batch_size * num_agents, hidden_size)] --> 
            # comm [shape: (mini_batch_size * num_agents, hidden_size)]
            comm = lstm_hidden_states.clone()
            if self.message_encoder:
                comm = self.msg_encoder(comm)
            # agent_mask to block communication if desired [shape: (mini_batch_size, num_agents, 1)]
            if self.comm_mask_zero:
                agent_mask = torch.zeros(
                    size=(mini_batch_size, self.num_agents, 1), 
                    dtype=self.tpdv['dtype'], 
                    device=self.tpdv['device']
                )
            else:
                if active_masks is not None:
                    agent_mask = active_masks[:, j].reshape(mini_batch_size, self.num_agents, 1)
                else:
                    agent_mask = torch.ones(
                        size=(mini_batch_size, self.num_agents, 1), 
                        dtype=self.tpdv['dtype'], 
                        device=self.tpdv['device']
                    )
            # mask communcation from dead agents [shape: (mini_batch_size, num_agents, hidden_size)]
            comm = comm.reshape(mini_batch_size, self.num_agents, self.hidden_size)
            comm = comm * agent_mask
            comm_ori = comm.clone()

            # sub-scheduler 1
            if not self.first_graph_complete:
                if self.use_gat_encoder:
                    # masked complete adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                    adj_complete = self.get_complete_graph(agent_mask)
                    # gat encoder [shape: mini_batch_size, num_agents, gat_encoder_out_size]
                    if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
                        encoded_state1 = self.gat_encoder(comm, adj_complete, batch)
                    else:
                        encoded_state1 = self.gat_encoder(comm, adj_complete)
                    # scheduled adjacency [shape: (mini_batch_size, num_agents, num_agents)] 
                    adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, encoded_state1, agent_mask, self.directed)
                else:
                    # scheduled adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                    adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask, self.directed)
            else:
                # masked complete adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                adj1 = self.get_complete_graph(agent_mask)
            # sub-processor 1 [shape: (mini_batch_size, num_agents, hidden_size)] --> 
            # [shape: (mini_batch_size, num_agents, gat_hidden_size * gat_num_heads)]
            if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
                comm = F.elu(self.sub_processor1(comm, adj1, batch))
            else:
                comm = F.elu(self.sub_processor1(comm, adj1))

            # sub-scheduler 2
            if self.learn_second_graph and not self.second_graph_complete:
                if self.use_gat_encoder:
                    if self.first_graph_complete:
                        # masked complete adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                        adj_complete = self.get_complete_graph(agent_mask)
                        # gat encoder [shape: mini_batch_size, num_agents, gat_encoder_out_size]
                        if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
                            encoded_state2 = self.gat_encoder(comm_ori, adj_complete, batch)
                        else:
                            encoded_state2 = self.gat_encoder(comm_ori, adj_complete)
                    else:
                        # reuse encoded state from sub-scheduler 1 
                        # [shape: mini_batch_size, num_agents, gat_encoder_out_size]
                        encoded_state2 = encoded_state1
                    # scheduled adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                    adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, encoded_state2, agent_mask, self.directed)
                else:
                    # scheduled adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                    adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, comm_ori, agent_mask, self.directed)
            elif not self.learn_second_graph and not self.second_graph_complete:
                # reuse scheduled adjacency from sub-scheduler 1 [shape: (mini_batch_size, num_agents, num_agents)]
                adj2 = adj1
            else:
                # masked complete adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                adj2 = self.get_complete_graph(agent_mask)
            # sub-processor 2 [shape: (mini_batch_size, num_agents, gat_hidden_size * gat_num_heads)] --> 
            # [shape: (mini_batch_size, num_agents, hidden_size)]
            if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
                comm = self.sub_processor2(comm, adj2, batch)
            else:
                comm = self.sub_processor2(comm, adj2)
            
            # mask communication to dead agents [shape: (mini_batch_size, num_agents, hidden_size)]
            comm = comm * agent_mask
            # message decoder [shape: (mini_batch_size, num_agents, hidden_size)]
            if self.message_decoder:
                comm = self.msg_decoder(comm)
            # [shape: (mini_batch_size * num_agents, hidden_size * 2)] -->
            # [shape: (mini_batch_size * num_agents, act_dims)], [shape: () == scalar] 
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                torch.cat(
                    (lstm_hidden_states, comm.reshape(mini_batch_size * self.num_agents, self.hidden_size)), 
                    dim=-1
                ),
                action[:, j], 
                available_actions[:, j],
                active_masks=active_masks[:, j] if self._use_policy_active_masks else None
            )
            # append action_log_probs and dist_entropy to respective lists
            action_log_probs_list.append(action_log_probs)
            dist_entropy_list.append(dist_entropy)

        # [shape: (mini_batch_size * num_agents * data_chunk_length, act_dims)]
        # [shape: () == scalar]
        return torch.stack(action_log_probs_list, dim=1)\
                    .reshape(mini_batch_size * self.num_agents * self.data_chunk_length, -1), \
               torch.stack(dist_entropy_list, dim=0).mean()

class MAGIC_Critic(nn.Module):
    """
    MAGIC Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(MAGIC_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self.data_chunk_length = args.data_chunk_length
        self.num_agents = args.num_agents
        self.message_encoder = args.magic_message_encoder
        self.message_decoder = args.magic_message_decoder
        self.use_gat_encoder = args.magic_use_gat_encoder
        self.gat_encoder_out_size = args.magic_gat_encoder_out_size
        self.gat_encoder_num_heads = args.magic_gat_encoder_num_heads
        self.gat_encoder_normalize = args.magic_gat_encoder_normalize
        self.first_graph_complete = args.magic_first_graph_complete
        self.second_graph_complete = args.magic_second_graph_complete
        self.learn_second_graph = args.magic_learn_second_graph
        self.gat_hidden_size = args.magic_gat_hidden_size
        self.gat_num_heads = args.magic_gat_num_heads
        self.gat_num_heads_out = args.magic_gat_num_heads_out
        self.self_loop_type1 = args.magic_self_loop_type1
        self.self_loop_type2 = args.magic_self_loop_type2
        self.first_gat_normalize = args.magic_first_gat_normalize
        self.second_gat_normalize = args.magic_second_gat_normalize
        self.comm_init = args.magic_comm_init
        self.comm_mask_zero = args.magic_comm_mask_zero
        self.directed = args.magic_directed
        self.gat_architecture = args.magic_gat_architecture
        self.n_gnn_fc_layers = args.magic_n_gnn_fc_layers
        self.gnn_train_eps = args.magic_gnn_train_eps
        self.gnn_norm = args.magic_gnn_norm

        self._use_orthogonal = args.use_orthogonal
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        assert self._recurrent_N == 1

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if len(cent_obs_shape) == 3:
            raise NotImplementedError("CNN-based observations not implemented for MAGIC")
        if isinstance(cent_obs_shape, (list, tuple)):
            self.obs_dims = cent_obs_shape[0]
        else:
            self.obs_dims = cent_obs_shape

        # encoder
        self.obs_encoder = nn.Linear(self.obs_dims, self.hidden_size)
        # lstm cell
        self.lstm_cell= nn.LSTMCell(self.hidden_size, self.hidden_size)
        # message encoder
        if self.message_encoder:
            self.msg_encoder = nn.Linear(self.hidden_size, self.hidden_size)
        # gat encoder for scheduler
        if self.use_gat_encoder:
            if self.gat_architecture == 'gat':
                self.gat_encoder = GraphAttentionGAT(
                    in_features=self.hidden_size, 
                    out_features=self.gat_encoder_out_size, 
                    dropout=0, 
                    negative_slope=0.2,
                    num_agents=self.num_agents, 
                    num_heads=self.gat_encoder_num_heads, 
                    self_loop_type=1, 
                    average=True, 
                    normalize=self.gat_encoder_normalize,
                    device=device
                )
            elif self.gat_architecture == 'gain':
                self.gat_encoder = GraphAttentionGAIN(
                    in_features=self.hidden_size,  
                    out_features=self.gat_encoder_out_size, 
                    dropout=0, 
                    negative_slope=0.2, 
                    num_agents=self.num_agents,
                    n_gnn_fc_layers=self.n_gnn_fc_layers, 
                    num_heads=self.gat_encoder_num_heads,
                    eps=1., 
                    gnn_train_eps=self.gnn_train_eps,
                    gnn_norm=self.gnn_norm,
                    self_loop_type=1,
                    average=True, 
                    normalize=self.gat_encoder_normalize,
                    device=device
                )
        # sub-schedulers
        if not self.first_graph_complete:
            if self.use_gat_encoder:
                self.sub_scheduler_mlp1 = NNLayers(
                    input_channels=self.gat_encoder_out_size * 2, 
                    block=MLPBlock, 
                    output_channels=[self.gat_encoder_out_size // 2, self.gat_encoder_out_size // 2, 2],
                    norm_type='none', 
                    activation_func='relu', 
                    dropout_p=0, 
                    weight_initialisation="default"
                )
            else:
                self.sub_scheduler_mlp1 = NNLayers(
                    input_channels=self.hidden_size * 2, 
                    block=MLPBlock, 
                    output_channels=[self.hidden_size // 2, self.hidden_size // 8, 2],
                    norm_type='none', 
                    activation_func='relu', 
                    dropout_p=0, 
                    weight_initialisation="default"
                )
        if self.learn_second_graph and not self.second_graph_complete:
            if self.use_gat_encoder:
                self.sub_scheduler_mlp2 = NNLayers(
                    input_channels=self.gat_encoder_out_size * 2, 
                    block=MLPBlock, 
                    output_channels=[self.gat_encoder_out_size // 2, self.gat_encoder_out_size // 2, 2],
                    norm_type='none', 
                    activation_func='relu', 
                    dropout_p=0, 
                    weight_initialisation="default"
                )
            else:
                self.sub_scheduler_mlp2 = NNLayers(
                    input_channels=self.hidden_size * 2, 
                    block=MLPBlock, 
                    output_channels=[self.hidden_size // 2, self.hidden_size // 8, 2],
                    norm_type='none', 
                    activation_func='relu', 
                    dropout_p=0, 
                    weight_initialisation="default"
                )
        # sub-processors
        if self.gat_architecture == 'gat':
            self.sub_processor1 = GraphAttentionGAT(
                in_features=self.hidden_size, 
                out_features=self.gat_hidden_size, 
                dropout=0, 
                negative_slope=0.2,
                num_agents=self.num_agents, 
                num_heads=self.gat_num_heads, 
                self_loop_type=self.self_loop_type1, 
                average=False, 
                normalize=self.first_gat_normalize,
                device=device
            )  
            self.sub_processor2 = GraphAttentionGAT(
                in_features=self.gat_hidden_size * self.gat_num_heads, 
                out_features=self.hidden_size, 
                dropout=0, 
                negative_slope=0.2,
                num_agents=self.num_agents, 
                num_heads=self.gat_num_heads_out, 
                self_loop_type=self.self_loop_type2, 
                average=True, 
                normalize=self.second_gat_normalize,
                device=device
            )
        elif self.gat_architecture == 'gain':
            self.sub_processor1 = GraphAttentionGAIN(
                in_features=self.hidden_size, 
                out_features=self.gat_hidden_size, 
                dropout=0, 
                negative_slope=0.2,
                num_agents=self.num_agents, 
                n_gnn_fc_layers=self.n_gnn_fc_layers, 
                num_heads=self.gat_num_heads,
                eps=1., 
                gnn_train_eps=self.gnn_train_eps,
                gnn_norm=self.gnn_norm, 
                self_loop_type=self.self_loop_type1, 
                average=False, 
                normalize=self.first_gat_normalize,
                device=device
            )
            self.sub_processor2 = GraphAttentionGAIN(
                in_features=self.gat_hidden_size * self.gat_num_heads, 
                out_features=self.hidden_size, 
                dropout=0, 
                negative_slope=0.2,
                num_agents=self.num_agents, 
                n_gnn_fc_layers=self.n_gnn_fc_layers, 
                num_heads=self.gat_num_heads_out,
                eps=1., 
                gnn_train_eps=self.gnn_train_eps,
                gnn_norm=self.gnn_norm, 
                self_loop_type=self.self_loop_type2, 
                average=True, 
                normalize=self.second_gat_normalize,
                device=device
            )
        # message decoder
        if self.message_decoder:
            self.msg_decoder = nn.Linear(self.hidden_size, self.hidden_size)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # decoder
        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size * 2, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size * 2, 1))

        # initialize weights as 0
        if self.comm_init == 'zeros':
            def init_linear(self, m):
                """
                Function to initialize the parameters in nn.Linear as 0 
                """
                if type(m) == nn.Linear:
                    m.weight.data.fill_(0.)
                    m.bias.data.fill_(0.)
            if self.message_encoder:
                self.msg_encoder.weight.data.zero_()
            if self.message_decoder:
                self.msg_decoder.weight.data.zero_()
            if not self.first_graph_complete:
                self.sub_scheduler_mlp1.apply(init_linear)
            if self.learn_second_graph and not self.second_graph_complete:
                self.sub_scheduler_mlp2.apply(init_linear)

        self.to(device)

    def get_complete_graph(self, agent_mask):
        """
        Function to generate a complete graph, and mask it with agent_mask
        """
        # adjacency [shape: (batch_size, num_agents, num_agents)]
        adj = torch.ones(
            size=(agent_mask.shape[0], self.num_agents, self.num_agents), 
            dtype=self.tpdv['dtype'], 
            device=self.tpdv['device']
        )
        # [shape: (batch_size, num_agents, 1)] --> [shape: (batch_size, num_agents, num_agents)]
        agent_mask = agent_mask.expand(-1, -1, self.num_agents)
        # mask adjacency [shape: (batch_size, num_agents, num_agents)]
        agent_mask_transpose = agent_mask.transpose(1, 2)
        adj = adj * agent_mask * agent_mask_transpose
        
        return adj

    def sub_scheduler(self, sub_scheduler_mlp, hidden_state, agent_mask, directed=True):
        """
        Function to perform a sub-scheduler

        Arguments: 
            sub_scheduler_mlp (nn.Sequential): the MLP layers in a sub-scheduler
            hidden_state (tensor): the encoded messages input to the sub-scheduler (batch_size, num_agents, hidden_size)
            agent_mask (tensor): (batch_size, num_agents, 1)
            directed (bool): decide if generate directed graphs

        Return:
            adj (tensor): a adjacency matrix which is the communication graph (batch_size, num_agents, num_agents)
        """
        # [shape: batch_size, num_agents, 1, hidden_size] --> [shape: batch_size, num_agents, num_agents, hidden_size]
        hard_attn_input_1 = hidden_state.unsqueeze(2).repeat(1, 1, self.num_agents, 1)
        # [shape: batch_size, 1, num_agents, hidden_size] --> [shape: batch_size, num_agents, num_agents, hidden_size]
        hard_attn_input_2 = hidden_state.unsqueeze(1).repeat(1, self.num_agents, 1, 1)
        # [shape: batch_size, num_agents, num_agents, hidden_size * 2]
        hard_attn_input = torch.cat((hard_attn_input_1, hard_attn_input_2), dim=-1)
        # [shape: batch_size, num_agents, num_agents, 2]
        if directed:
            hard_attn_output = F.gumbel_softmax(sub_scheduler_mlp(hard_attn_input), hard=True)
        else:
            hard_attn_output = F.gumbel_softmax(
                0.5 * sub_scheduler_mlp(hard_attn_input) + 0.5 * sub_scheduler_mlp(hard_attn_input.permute(0, 2, 1, 3)), 
                hard=True
            )
        # [shape: batch_size, num_agents, num_agents, 1]
        hard_attn_output = torch.narrow(hard_attn_output, 3, 1, 1)
        # [shape: (batch_size, num_agents, 1)] --> [shape: (batch_size, num_agents, num_agents)]
        agent_mask = agent_mask.expand(-1, -1, self.num_agents)
        # mask adjacency [shape: (batch_size, num_agents, num_agents)]
        agent_mask_transpose = agent_mask.transpose(1, 2)
        adj = hard_attn_output.squeeze(-1) * agent_mask * agent_mask_transpose

        return adj

    def forward(self, cent_obs, lstm_hidden_states, lstm_cell_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param lstm_hidden_states: (np.ndarray / torch.Tensor) hidden states for LSTM.
        :param lstm_cell_states: (np.ndarray / torch.Tensor) cell states for LSTM.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return lstm_hidden_states: (torch.Tensor) updated LSTM hidden states.
        :return lstm_cell_states: (torch.Tensor) updated LSTM cell states.
        """
        # [shape: (batch_size * num_agents, obs_dims)]
        cent_obs = check(cent_obs).to(**self.tpdv)
        batch_size = cent_obs.shape[0] // self.num_agents
        # obtain batch, [shape: (batch_size * num_agents)]
        if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
            batch = torch.arange(batch_size).repeat_interleave(self.num_agents).to(self.tpdv['device'])
        # [shape: (batch_size * num_agents, recurrent_N = 1, hidden_size)] --> 
        # [shape: (batch_size * num_agents, hidden_size)]
        lstm_hidden_states = check(lstm_hidden_states).to(**self.tpdv).squeeze(1)
        lstm_cell_states = check(lstm_cell_states).to(**self.tpdv).squeeze(1)
        # [shape: (batch_size * num_agents, 1)]
        masks = check(masks).to(**self.tpdv)

        # cent_obs --> encoded_obs [shape: (batch_size * num_agents, hidden_size)]
        encoded_obs = self.obs_encoder(cent_obs)
        # encoded_obs [shape: (batch_size * num_agents, hidden_size)]
        # lstm_hidden_states / lstm_cell_states [shape: (batch_size * num_agents, hidden_size)]
        # masks [shape: (batch_size * num_agents, 1)] --> 
        # lstm_hidden_states / lstm_cell_states [shape: (batch_size * num_agents, hidden_size)]
        lstm_hidden_states, lstm_cell_states = self.lstm_cell(
            encoded_obs, 
            (lstm_hidden_states * masks, lstm_cell_states * masks)
        )
        # message encoder lstm_hidden_states [shape: (batch_size * num_agents, hidden_size)] --> 
        # comm [shape: (batch_size * num_agents, hidden_size)]
        comm = lstm_hidden_states.clone()
        if self.message_encoder:
            comm = self.msg_encoder(comm)
        # agent_mask to block communication if desired [shape: (batch_size, num_agents, 1)]
        if self.comm_mask_zero:
            agent_mask = torch.zeros(
                size=(batch_size, self.num_agents, 1), 
                dtype=self.tpdv['dtype'], 
                device=self.tpdv['device']
            )
        else:
            agent_mask = torch.ones(
                size=(batch_size, self.num_agents, 1), 
                dtype=self.tpdv['dtype'], 
                device=self.tpdv['device']
            )
        # mask communcation from dead agents [shape: (batch_size, num_agents, hidden_size)]
        comm = comm.reshape(batch_size, self.num_agents, self.hidden_size)
        comm = comm * agent_mask
        comm_ori = comm.clone()

        # sub-scheduler 1
        if not self.first_graph_complete:
            if self.use_gat_encoder:
                # masked complete adjacency [shape: (batch_size, num_agents, num_agents)]
                adj_complete = self.get_complete_graph(agent_mask)
                # gat encoder [shape: batch_size, num_agents, gat_encoder_out_size]
                if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
                    encoded_state1 = self.gat_encoder(comm, adj_complete, batch)
                else:
                    encoded_state1 = self.gat_encoder(comm, adj_complete)
                # scheduled adjacency [shape: (batch_size, num_agents, num_agents)] 
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, encoded_state1, agent_mask, self.directed)
            else:
                # scheduled adjacency [shape: (batch_size, num_agents, num_agents)]
                adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask, self.directed)
        else:
            # masked complete adjacency [shape: (batch_size, num_agents, num_agents)]
            adj1 = self.get_complete_graph(agent_mask)
        # sub-processor 1 [shape: (batch_size, num_agents, hidden_size)] --> 
        # [shape: (batch_size, num_agents, gat_hidden_size * gat_num_heads)]
        if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
            comm = F.elu(self.sub_processor1(comm, adj1, batch))
        else:
            comm = F.elu(self.sub_processor1(comm, adj1))

        # sub-scheduler 2
        if self.learn_second_graph and not self.second_graph_complete:
            if self.use_gat_encoder:
                if self.first_graph_complete:
                    # masked complete adjacency [shape: (batch_size, num_agents, num_agents)]
                    adj_complete = self.get_complete_graph(agent_mask)
                    # gat encoder [shape: batch_size, num_agents, gat_encoder_out_size]
                    if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
                        encoded_state2 = self.gat_encoder(comm_ori, adj_complete, batch)
                    else:
                        encoded_state2 = self.gat_encoder(comm_ori, adj_complete)
                else:
                    # reuse encoded state from sub-scheduler 1 [shape: batch_size, num_agents, gat_encoder_out_size]
                    encoded_state2 = encoded_state1
                # scheduled adjacency [shape: (batch_size, num_agents, num_agents)]
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, encoded_state2, agent_mask, self.directed)
            else:
                # scheduled adjacency [shape: (batch_size, num_agents, num_agents)]
                adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, comm_ori, agent_mask, self.directed)
        elif not self.learn_second_graph and not self.second_graph_complete:
            # reuse scheduled adjacency from sub-scheduler 1 [shape: (batch_size, num_agents, num_agents)]
            adj2 = adj1
        else:
            # masked complete adjacency [shape: (batch_size, num_agents, num_agents)]
            adj2 = self.get_complete_graph(agent_mask)
        # sub-processor 2 [shape: (batch_size, num_agents, gat_hidden_size * gat_num_heads)] --> 
        # [shape: (batch_size, num_agents, hidden_size)]
        if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
            comm = self.sub_processor2(comm, adj2, batch)
        else:
            comm = self.sub_processor2(comm, adj2)
        
        # mask communication to dead agents [shape: (batch_size, num_agents, hidden_size)]
        comm = comm * agent_mask
        # message decoder [shape: (batch_size, num_agents, hidden_size)]
        if self.message_decoder:
            comm = self.msg_decoder(comm)
        # [shape: (batch_size * num_agents, hidden_size * 2)] --> values [shape: (batch_size * num_agents, 1)]
        values = self.v_out(
            torch.cat((lstm_hidden_states, comm.reshape(batch_size * self.num_agents, self.hidden_size)), dim=-1)
        )

        # [shape: (batch_size * num_agents, 1)], [shape: (batch_size * num_agents, recurrent_N = 1, hidden_size)]
        return values, lstm_hidden_states.unsqueeze(1), lstm_cell_states.unsqueeze(1)

    def evaluate_actions(self, cent_obs, lstm_hidden_states, lstm_cell_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param lstm_hidden_states: (np.ndarray / torch.Tensor) hidden states for LSTM.
        :param lstm_cell_states: (np.ndarray / torch.Tensor) cell states for LSTM.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        """
        # [shape: (mini_batch_size * num_agents * data_chunk_length, obs_dims)] --> 
        # [shape: (mini_batch_size * num_agents, data_chunk_length, obs_dims)]
        cent_obs = check(cent_obs).to(**self.tpdv)
        cent_obs = cent_obs.reshape(-1, self.data_chunk_length, self.obs_dims)
        mini_batch_size = cent_obs.shape[0] // self.num_agents
        # obtain batch, [shape: (mini_batch_size * data_chunk_length * num_agents)]
        if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
            batch = torch.arange(mini_batch_size).repeat_interleave(self.num_agents).to(self.tpdv['device'])
        # [shape: (mini_batch_size * num_agents, recurrent_N = 1, hidden_size)] --> 
        # [shape: (mini_batch_size * num_agents, hidden_size)]
        lstm_hidden_states = check(lstm_hidden_states).to(**self.tpdv).squeeze(1)
        lstm_cell_states = check(lstm_cell_states).to(**self.tpdv).squeeze(1)
        # [shape: (mini_batch_size * num_agents * data_chunk_length, 1)] --> 
        # [shape: (mini_batch_size * num_agents, data_chunk_length, 1)]
        masks = check(masks).to(**self.tpdv)
        masks = masks.reshape(mini_batch_size * self.num_agents, self.data_chunk_length, -1)
        # list to store values
        values_list = []

        # cent_obs --> obs_enc [shape: (mini_batch_size * num_agents, data_chunk_length, hidden_size)]
        encoded_obs = self.obs_encoder(cent_obs)

        # iterate over data_chunk_length 
        for j in range(self.data_chunk_length):
            # encoded_obs[:, j] [shape: (mini_batch_size * num_agents, hidden_size)]
            # lstm_hidden_states / lstm_cell_states [shape: (mini_batch_size * num_agents, hidden_size)]
            # masks [shape: (mini_batch_size * num_agents, 1)] --> 
            # lstm_hidden_states / lstm_cell_states [shape: (mini_batch_size * num_agents, hidden_size)]
            lstm_hidden_states, lstm_cell_states = self.lstm_cell(
                encoded_obs[:, j], 
                (lstm_hidden_states * masks[:, j], lstm_cell_states * masks[:, j])
            )
            # message encoder lstm_hidden_states [shape: (mini_batch_size * num_agents, hidden_size)] --> 
            # comm [shape: (mini_batch_size * num_agents, hidden_size)]
            comm = lstm_hidden_states.clone()
            if self.message_encoder:
                comm = self.msg_encoder(comm)
            # agent_mask to block communication if desired [shape: (mini_batch_size, num_agents, 1)]
            if self.comm_mask_zero:
                agent_mask = torch.zeros(
                    size=(mini_batch_size, self.num_agents, 1), 
                    dtype=self.tpdv['dtype'], 
                    device=self.tpdv['device']
                )
            else:
                agent_mask = torch.ones(
                    size=(mini_batch_size, self.num_agents, 1), 
                    dtype=self.tpdv['dtype'], 
                    device=self.tpdv['device']
                )
            # mask communcation from dead agents [shape: (mini_batch_size, num_agents, hidden_size)]
            comm = comm.reshape(mini_batch_size, self.num_agents, self.hidden_size)
            comm = comm * agent_mask
            comm_ori = comm.clone()

            # sub-scheduler 1
            if not self.first_graph_complete:
                if self.use_gat_encoder:
                    # masked complete adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                    adj_complete = self.get_complete_graph(agent_mask)
                    # gat encoder [shape: mini_batch_size, num_agents, gat_encoder_out_size]
                    if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
                        encoded_state1 = self.gat_encoder(comm, adj_complete, batch)
                    else:
                        encoded_state1 = self.gat_encoder(comm, adj_complete)
                    # scheduled adjacency [shape: (mini_batch_size, num_agents, num_agents)] 
                    adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, encoded_state1, agent_mask, self.directed)
                else:
                    # scheduled adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                    adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask, self.directed)
            else:
                # masked complete adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                adj1 = self.get_complete_graph(agent_mask)
            # sub-processor 1 [shape: (mini_batch_size, num_agents, hidden_size)] --> 
            # [shape: (mini_batch_size, num_agents, gat_hidden_size * gat_num_heads)]
            if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
                comm = F.elu(self.sub_processor1(comm, adj1, batch))
            else:
                comm = F.elu(self.sub_processor1(comm, adj1))

            # sub-scheduler 2
            if self.learn_second_graph and not self.second_graph_complete:
                if self.use_gat_encoder:
                    if self.first_graph_complete:
                        # masked complete adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                        adj_complete = self.get_complete_graph(agent_mask)
                        # gat encoder [shape: mini_batch_size, num_agents, gat_encoder_out_size]
                        if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
                            encoded_state2 = self.gat_encoder(comm_ori, adj_complete, batch)
                        else:
                            encoded_state2 = self.gat_encoder(comm_ori, adj_complete)
                    else:
                        # reuse encoded state from sub-scheduler 1 
                        # [shape: mini_batch_size, num_agents, gat_encoder_out_size]
                        encoded_state2 = encoded_state1
                    # scheduled adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                    adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, encoded_state2, agent_mask, self.directed)
                else:
                    # scheduled adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                    adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, comm_ori, agent_mask, self.directed)
            elif not self.learn_second_graph and not self.second_graph_complete:
                # reuse scheduled adjacency from sub-scheduler 1 [shape: (mini_batch_size, num_agents, num_agents)]
                adj2 = adj1
            else:
                # masked complete adjacency [shape: (mini_batch_size, num_agents, num_agents)]
                adj2 = self.get_complete_graph(agent_mask)
            # sub-processor 2 [shape: (mini_batch_size, num_agents, gat_hidden_size * gat_num_heads)] --> 
            # [shape: (mini_batch_size, num_agents, hidden_size)]
            if self.gat_architecture == 'gain' and self.gnn_norm == 'graphnorm':
                comm = self.sub_processor2(comm, adj2, batch)
            else:
                comm = self.sub_processor2(comm, adj2)
            
            # mask communication to dead agents [shape: (mini_batch_size, num_agents, hidden_size)]
            comm = comm * agent_mask
            # message decoder [shape: (mini_batch_size, num_agents, hidden_size)]
            if self.message_decoder:
                comm = self.msg_decoder(comm)
            # [shape: (mini_batch_size * num_agents, hidden_size * 2)] --> 
            # values [shape: (mini_batch_size * num_agents, 1)]
            values = self.v_out(
                torch.cat(
                    (lstm_hidden_states, comm.reshape(mini_batch_size * self.num_agents, self.hidden_size)), 
                    dim=-1
                )
            )
            values_list.append(values)

        # [shape: (mini_batch_size * num_agents * data_chunk_length, 1)]
        return torch.stack(values_list, dim=1).reshape(-1, 1)