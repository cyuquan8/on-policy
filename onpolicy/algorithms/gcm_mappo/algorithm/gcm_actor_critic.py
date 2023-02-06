import torch
import torch.nn as nn

from onpolicy.algorithms.utils.nn import DNAGATv2Layers, DNAGATv2Block, GINBlock, GINLayers, MLPBlock, NNLayers
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.single_act import SingleACTLayer
from onpolicy.algorithms.utils.util import init, check, complete_graph_edge_index
from onpolicy.utils.util import get_shape_from_obs_space
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph


class GCMNetDNAGATv2Actor(nn.Module):
    """
    GCMNetDNAGATv2 Actor network class for MAPPO. Outputs actions given observations.
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

        self.data_chunk_length = args.data_chunk_length
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device

        self.num_agents = args.num_agents
        self.n_rollout_threads = args.n_rollout_threads
        self.n_gnn_layers = args.n_gnn_layers
        self.somu_n_layers = args.somu_n_layers
        self.scmu_n_layers = args.scmu_n_layers
        self.somu_lstm_hidden_size = args.somu_lstm_hidden_size
        self.scmu_lstm_hidden_size = args.scmu_lstm_hidden_size
        self.somu_multi_att_n_heads = args.somu_multi_att_n_heads
        self.scmu_multi_att_n_heads = args.scmu_multi_att_n_heads
        self.fc_output_dims = args.fc_output_dims
        self.n_fc_layers = args.n_fc_layers
        self.knn = args.knn
        self.k = args.k
        self.rni = args.rni
        self.rni_ratio = args.rni_ratio

        obs_shape = get_shape_from_obs_space(obs_space)
        if isinstance(obs_shape, list):
            self.obs_dims = obs_shape[0]
        else:
            self.obs_dims = obs_shape

        self.rni_dims = round(self.obs_dims * self.rni_ratio)

        # model architecture for mappo GCMNetDNAGATv2 actor

        # gnn layers
        self.gnn_layers = DNAGATv2Layers(input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                                         block=DNAGATv2Block, 
                                         output_channels=[self.obs_dims + self.rni_dims if self.rni\
                                                          else self.obs_dims for i in range(self.n_gnn_layers)], 
                                         concat=False)

        # list of lstms for self observation memory unit (somu) for each agent
        # somu_lstm_input_size is the dimension of the observations
        self.somu_lstm_list = nn.ModuleList([nn.LSTM(input_size=self.obs_dims, hidden_size=self.somu_lstm_hidden_size, 
                                                     num_layers=self.somu_n_layers, batch_first=True).to(device) 
                                             for _ in range(self.num_agents)])

        # list of lstms for self communication memory unit (scmu) for each agent
        # somu_lstm_input_size are all layers of gnn
        self.scmu_lstm_list = nn.ModuleList([nn.LSTM(input_size=(self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims)\
                                                                if self.rni else (self.n_gnn_layers + 1) * self.obs_dims, 
                                                     hidden_size=self.scmu_lstm_hidden_size, 
                                                     num_layers=self.scmu_n_layers, batch_first=True).to(device) 
                                             for _ in range(self.num_agents)])

        # multi-head self attention layer for somu and scmu to selectively choose between the lstms outputs
        self.somu_multi_att_layer_list = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.somu_lstm_hidden_size, 
                                                                              num_heads=self.somu_multi_att_n_heads, 
                                                                              dropout=0, batch_first=True, device=device) 
                                                        for _ in range(self.num_agents)])
        self.scmu_multi_att_layer_list = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.scmu_lstm_hidden_size, 
                                                                              num_heads=self.scmu_multi_att_n_heads, 
                                                                              dropout=0, batch_first=True, device=device) 
                                                        for _ in range(self.num_agents)])

        # shared hidden fc layers for to generate actions for each agent
        # input channels are all layers of gnn (including initial observations) 
        # + concatenated outputs of somu_multi_att_layer and scmu_multi_att_layer
        # fc_output_dims is the list of sizes of output channels fc_block
        self.fc_layers = NNLayers(input_channels=(self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) + \
                                                 self.somu_n_layers * self.somu_lstm_hidden_size + \
                                                 self.scmu_n_layers * self.scmu_lstm_hidden_size if self.rni
                                                 else (self.n_gnn_layers + 1) * self.obs_dims + \
                                                 self.somu_n_layers * self.somu_lstm_hidden_size + \
                                                 self.scmu_n_layers * self.scmu_lstm_hidden_size, 
                                  block=MLPBlock, output_channels=[self.fc_output_dims for i in range(self.n_fc_layers)], 
                                  activation_func='relu', dropout_p=0, 
                                  weight_initialisation="orthogonal" if self._use_orthogonal else "default").to(device)

        # shared final action layer for each agent
        self.act = SingleACTLayer(action_space, self.fc_output_dims, self._use_orthogonal, self._gain).to(device)
        
        self.to(device)
        
    def forward(self, obs, somu_hidden_states_actor, somu_cell_states_actor, scmu_hidden_states_actor, scmu_cell_states_actor, 
                masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param somu_hidden_states_actor: (np.ndarray / torch.Tensor) hidden states for somu network.
        :param somu_cell_states_actor: (np.ndarray / torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_actor: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_actor: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return somu_hidden_states_actor: (np.ndarray / torch.Tensor) hidden states for somu network.
        :return somu_cell_states_actor: (np.ndarray / torch.Tensor) cell states for somu network.
        :return scmu_hidden_states_actor: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :return scmu_cell_states_actor: (np.ndarray / torch.Tensor) hidden states for scmu network.
        """
        if self.knn:
            # [shape: (batch_size, num_agents, obs_dims)]
            obs = check(obs) 
            batch_size = obs.shape[0]
            # [shape: (batch_size * num_agents, obs_dims)]
            obs_temp = obs.reshape(batch_size * self.num_agents, self.obs_dims)
            batch = torch.tensor([i // self.num_agents for i in range(batch_size * self.num_agents)])
            edge_index = knn_graph(x=obs_temp, k=self.k, batch=batch, loop=True)
            obs = obs.to(**self.tpdv)  
            if self.rni:
                # zero mean std 1 gaussian noise
                # [shape: (batch_size, num_agents, rni_dims)] 
                rni = torch.normal(0, 1, size=(batch_size, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # [shape: (batch_size, num_agents, obs_dims + rni_dims)]
                obs_rni = torch.cat((obs, rni), dim=-1)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni[i, :, :] if self.rni else obs[i, :, :], edge_index=edge_index) 
                                            for i in range(batch_size)]).to(self.device)
        else:
            # obtain edge index
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
            # [shape: (batch_size, num_agents, obs_dims)]  
            obs = check(obs).to(**self.tpdv)  
            batch_size = obs.shape[0]
            if self.rni:
                # zero mean std 1 gaussian noise 
                # [shape: (batch_size, num_agents, rni_dims)] 
                rni = torch.normal(0, 1, size=(batch_size, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # [shape: (batch_size, num_agents, obs_dims + rni_dims)]
                obs_rni = torch.cat((obs, rni), dim=-1)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni[i, :, :] if self.rni else obs[i, :, :], edge_index=edge_index) 
                                            for i in range(batch_size)]).to(self.device)
        # [shape: (batch_size, num_agents, somu_n_layers, somu_lstm_hidden_size)]
        somu_hidden_states_actor = check(somu_hidden_states_actor).to(**self.tpdv)
        somu_cell_states_actor = check(somu_cell_states_actor).to(**self.tpdv)
        # [shape: (batch_size, num_agents, scmu_n_layers, scmu_lstm_hidden_size)] 
        scmu_hidden_states_actor = check(scmu_hidden_states_actor).to(**self.tpdv)
        scmu_cell_states_actor = check(scmu_cell_states_actor).to(**self.tpdv) 
        # [shape: (batch_size, num_agents, 1)]
        masks = check(masks).to(**self.tpdv).reshape(batch_size, self.num_agents, -1) 
        if available_actions is not None:
            # [shape: (batch_size, num_agents, action_space_dim)]
            available_actions = check(available_actions).to(**self.tpdv) 
        # store somu and scmu hidden states and cell states, actions and action_log_probs
        somu_lstm_hidden_state_list = []
        somu_lstm_cell_state_list = []
        scmu_lstm_hidden_state_list = []
        scmu_lstm_cell_state_list = []
        actions_list = []
        action_log_probs_list = []
       
        # obs_gnn.x [shape: (batch_size * num_agents, obs_dims / (obs_dims + rni_dims))] 
        # --> gnn_layers [shape: (batch_size, num_agents, (n_gnn_layers + 1) * obs_dims / (n_gnn_layers + 1) * (obs_dims + rni_dims))]
        gnn_output = self.gnn_layers(x=obs_gnn.x, edge_index=obs_gnn.edge_index)\
                         .reshape(batch_size, self.num_agents, (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) if self.rni else\
                                  (self.n_gnn_layers + 1) * self.obs_dims)
       
        # iterate over agents 
        for i in range(self.num_agents):
            # obs[:, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, obs_dims)],
            # masks[:, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (somu_n_layers, batch_size, 1)],
            # (h_0 [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)]) -->
            # somu_output [shape: (batch_size, sequence_length=1, somu_lstm_hidden_size)], 
            # (h_n [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)], c_n [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)])
            somu_output, somu_hidden_cell_states_tup = self.somu_lstm_list[i](obs[:, i, :].unsqueeze(dim=1), 
                                                                              (somu_hidden_states_actor.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.somu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous(), 
                                                                               somu_cell_states_actor.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.somu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous()))
            # gnn_output[:, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, (n_gnn_layers + 1) * obs_dims)],
            # masks[:, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (scmu_n_layers, batch_size, 1)], 
            # (h_0 [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)], c_0 [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)]) -->
            # scmu_output [shape: (batch_size, sequence_length=1, scmu_lstm_hidden_state)], 
            # (h_n [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)], c_n [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)])
            scmu_output, scmu_hidden_cell_states_tup = self.scmu_lstm_list[i](gnn_output[:, i, :].unsqueeze(dim=1), 
                                                                              (scmu_hidden_states_actor.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.scmu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous(), 
                                                                               scmu_cell_states_actor.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.scmu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous()))
            # (h_n [shape: (batch_size, somu_n_layers / scmu_n_layers, scmu_lstm_hidden_state)], 
            # c_n [shape: (batch_size, somu_n_layers / scmu_n_layers, scmu_lstm_hidden_state)])
            somu_lstm_hidden_state_list.append(somu_hidden_cell_states_tup[0].transpose(0, 1))
            somu_lstm_cell_state_list.append(somu_hidden_cell_states_tup[1].transpose(0, 1))
            scmu_lstm_hidden_state_list.append(scmu_hidden_cell_states_tup[0].transpose(0, 1))
            scmu_lstm_cell_state_list.append(scmu_hidden_cell_states_tup[1].transpose(0, 1))
            # somu_output / scmu_output [shape: (batch_size, sequence_length=1, somu_lstm_hidden_size / scmu_lstm_hidden_state)],
            # somu_hidden_state / scmu_hidden_state [shape: (batch_size, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)] 
            # --> multihead attention [shape: (batch_size, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
            somu_output = self.somu_multi_att_layer_list[i](somu_lstm_hidden_state_list[i], somu_output, somu_output)[0]
            scmu_output = self.somu_multi_att_layer_list[i](scmu_lstm_hidden_state_list[i], scmu_output, scmu_output)[0]

            # concatenate obs and outputs from gnn, somu and scmu 
            # [shape: (batch_size, (n_gnn_layers + 1) * obs_dims + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size) / 
            # (batch_size, (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            output = torch.cat((gnn_output[:, i, :], somu_output.reshape(batch_size, -1), scmu_output.reshape(batch_size, -1)), dim=-1)
            # output [shape: (batch_size, (n_gnn_layers + 1) * self.obs_dims + somu_lstm_hidden_size + scmu_lstm_hidden_size) / 
            # (batch_size, (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            # -->   
            # fc_layers [shape: (batch_size, fc_output_dims)]
            output = self.fc_layers(output)
            # fc_layers --> act [shape: (batch_size, action_space_dim)]
            actions, action_log_probs = self.act(output, available_actions[:, i, :] if available_actions is not None else None, deterministic)
            # append actions and action_log_probs to respective lists
            actions_list.append(actions)
            action_log_probs_list.append(action_log_probs)
       
        # [shape: (batch_size, num_agents, action_space_dim)]
        # [shape: (batch_size, num_agents, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
        return torch.stack(actions_list, dim=1), torch.stack(action_log_probs_list, dim=1), \
               torch.stack(somu_lstm_hidden_state_list, dim=1), torch.stack(somu_lstm_cell_state_list, dim=1), \
               torch.stack(scmu_lstm_hidden_state_list, dim=1), torch.stack(scmu_lstm_cell_state_list, dim=1)

    def evaluate_actions(self, obs, somu_hidden_states_actor, somu_cell_states_actor, scmu_hidden_states_actor, scmu_cell_states_actor, 
                         action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param somu_hidden_states_actor: (torch.Tensor) hidden states for somu network.
        :param somu_cell_states_actor: (torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_actor: (torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_actor: (torch.Tensor) hidden states for scmu network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.knn:
            # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims]   
            obs = check(obs)
            mini_batch_size = obs.shape[0]
            # [shape: (mini_batch_size * data_chunk_length * num_agents, obs_dims)] 
            obs_temp = obs.reshape(mini_batch_size * self.data_chunk_length * self.num_agents, self.obs_dims)
            batch = torch.tensor([i // self.num_agents for i in range(mini_batch_size * self.data_chunk_length * self.num_agents)])
            edge_index = knn_graph(x=obs_temp, k=self.k, batch=batch, loop=True)
            obs = obs.to(**self.tpdv)
            # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims)] 
            obs_batch = obs.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims)
            if self.rni:
                # zero mean std 1 gaussian noise 
                # [shape: (mini_batch_size, data_chunk_length, num_agents, rni_dims)]
                rni = torch.normal(0, 1, size=(mini_batch_size, self.data_chunk_length, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims + rni_dims)]
                obs_rni = torch.cat((obs, rni), dim=-1)
                # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims + rni_dims)] 
                obs_rni_batch = obs_rni.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims + self.rni_dims)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni_batch[i, :, :] if self.rni else obs_batch[i, :, :], edge_index=edge_index) 
                                            for i in range(mini_batch_size * self.data_chunk_length)]).to(self.device)
        else:
            # obtain edge index
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
            # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims(pre-rni))]   
            obs = check(obs).to(**self.tpdv)
            mini_batch_size = obs.shape[0]
            # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims)] 
            obs_batch = obs.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims)
            if self.rni:
                # zero mean std 1 gaussian noise 
                # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims + rni_dims)] 
                rni = torch.normal(0, 1, size=(mini_batch_size, self.data_chunk_length, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims + rni_dims)]
                obs_rni = torch.cat((obs, rni), dim=-1)
                # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims + rni_dims)] 
                obs_rni_batch = obs_rni.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims + self.rni_dims)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni_batch[i, :, :] if self.rni else obs_batch[i, :, :], edge_index=edge_index) 
                                            for i in range(mini_batch_size * self.data_chunk_length)]).to(self.device)
        # [shape: (mini_batch_size, data_chunk_length, num_agents, action_space_dim)]  
        action = check(action).to(**self.tpdv) 
        # [shape: (mini_batch_size, num_agents, somu_n_layers, somu_lstm_hidden_size)]
        somu_hidden_states_actor = check(somu_hidden_states_actor).to(**self.tpdv) 
        somu_cell_states_actor = check(somu_cell_states_actor).to(**self.tpdv)
        # [shape: (mini_batch_size, num_agents, scmu_n_layers, scmu_lstm_hidden_size)]
        scmu_hidden_states_actor = check(scmu_hidden_states_actor).to(**self.tpdv) 
        scmu_cell_states_actor = check(scmu_cell_states_actor).to(**self.tpdv)
        # [shape: (mini_batch_size, data_chunk_length, num_agents, 1)]  
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            # [shape: (mini_batch_size, data_chunk_length, num_agents, action_space_dim)]
            available_actions = check(available_actions).to(**self.tpdv) 
            # [shape: (mini_batch_size * data_chunk_length, num_agents, action_space_dim)]
            available_actions = available_actions.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, -1)
        if active_masks is not None:
            # [shape: (mini_batch_size, data_chunk_length, num_agents, 1)]
            active_masks = check(active_masks).to(**self.tpdv)
            # [shape: (mini_batch_size * data_chunk_length, num_agents, 1)]
            active_masks = active_masks.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, -1)
        # [shape: (mini_batch_size * data_chunk_length, num_agents, action_space_dim)] 
        action = action.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, -1)
        # store actions and actions_log_probs
        action_log_probs_list = []
        dist_entropy_list = []

        # obs_gnn.x [shape: (mini_batch_size * data_chunk_length * num_agents, obs_dims / (obs_dims + rni_dims))] -->
        # gnn_layers [shape: (mini_batch_size, data_chunk_length, num_agents, (n_gnn_layers + 1) * obs_dims \ 
        # (n_gnn_layers + 1) * (obs_dims + rni_dims))]
        gnn_output = self.gnn_layers(x=obs_gnn.x, edge_index=obs_gnn.edge_index)\
                         .reshape(mini_batch_size, self.data_chunk_length, self.num_agents, 
                                  (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) if self.rni else\
                                  (self.n_gnn_layers + 1) * self.obs_dims)

        # iterate over agents 
        for i in range(self.num_agents):
            # list to store somu and scmu outputs, hidden states and cell states over sequence of inputs of len, data_chunk_length
            somu_seq_output_list = []
            somu_seq_lstm_hidden_state_list = []
            somu_seq_lstm_cell_state_list = []
            scmu_seq_output_list = []
            scmu_seq_lstm_hidden_state_list = []
            scmu_seq_lstm_cell_state_list = []
            # initialise hidden and cell states for somu and scmu at start of sequence
            # (h_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
            somu_seq_lstm_hidden_state_list.append(somu_hidden_states_actor[:, i, :, :].transpose(0, 1).contiguous())
            somu_seq_lstm_cell_state_list.append(somu_cell_states_actor[:, i, :, :].transpose(0, 1).contiguous())
            scmu_seq_lstm_hidden_state_list.append(scmu_hidden_states_actor[:, i, :, :].transpose(0, 1).contiguous())
            scmu_seq_lstm_cell_state_list.append(scmu_cell_states_actor[:, i, :, :].transpose(0, 1).contiguous())

            # iterate over data_chunk_length
            for j in range(self.data_chunk_length):
                # obs[:, j, i, :].unsqueeze(dim=1) [shape: (mini_batch_size, sequence_length=1, obs_dims)],
                # masks[:, j, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (somu_n_layers, batch_size, 1)],
                # (h_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
                # -->
                # somu_output [shape: (mini_batch_size, sequence_length=1, somu_lstm_hidden_size)], 
                # (h_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], c_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
                somu_output, somu_hidden_cell_states_tup = self.somu_lstm_list[i](obs[:, j, i, :].unsqueeze(dim=1),
                                                                                  (somu_seq_lstm_hidden_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.somu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous(),
                                                                                   somu_seq_lstm_cell_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.somu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous()))
                # gnn_output[:, j, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, (n_gnn_layers + 1) * obs_dims)],
                # masks[:, j, i, :].repeat(1, self.scmu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (scmu_n_layers, batch_size, 1)],
                # (h_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], c_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)])
                # -->
                # scmu_output [shape: (mini_batch_size, sequence_length=1, scmu_lstm_hidden_size)], 
                # (h_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], c_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)])
                scmu_output, scmu_hidden_cell_states_tup = self.scmu_lstm_list[i](gnn_output[:, j, i, :].unsqueeze(dim=1),
                                                                                  (scmu_seq_lstm_hidden_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.scmu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous(),
                                                                                   scmu_seq_lstm_cell_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.scmu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous()))
                somu_seq_output_list.append(somu_output)
                scmu_seq_output_list.append(scmu_output)
                somu_seq_lstm_hidden_state_list.append(somu_hidden_cell_states_tup[0])
                somu_seq_lstm_cell_state_list.append(somu_hidden_cell_states_tup[1])
                scmu_seq_lstm_hidden_state_list.append(scmu_hidden_cell_states_tup[0])
                scmu_seq_lstm_cell_state_list.append(scmu_hidden_cell_states_tup[1])

            # [shape: (mini_batch_size * data_chunk_length, 1, somu_lstm_hidden_size / scmu_lstm_hidden_state)]
            somu_output = torch.stack(somu_seq_output_list, dim=1).reshape(mini_batch_size * self.data_chunk_length, 1, self.somu_lstm_hidden_size)
            scmu_output = torch.stack(scmu_seq_output_list, dim=1).reshape(mini_batch_size * self.data_chunk_length, 1, self.scmu_lstm_hidden_size)
            # [shape: (data_chunk_length, somu_n_layers / scmu_n_layers, mini_batch_size, somu_lstm_hidden_state / scmu_lstm_hidden_state)] --> 
            # [shape: (mini_batch_size * data_chunk_length, somu_n_layers / scmu_n_layers, somu_lstm_hidden_state / scmu_lstm_hidden_state)]
            somu_lstm_hidden_state = torch.stack(somu_seq_lstm_hidden_state_list[1:], dim=0).permute(2, 0, 1, 3).reshape(mini_batch_size * self.data_chunk_length, 
                                                                                                                         self.somu_n_layers, 
                                                                                                                         self.somu_lstm_hidden_size)
            scmu_lstm_hidden_state = torch.stack(scmu_seq_lstm_hidden_state_list[1:], dim=0).permute(2, 0, 1, 3).reshape(mini_batch_size * self.data_chunk_length, 
                                                                                                                         self.scmu_n_layers, 
                                                                                                                         self.scmu_lstm_hidden_size)
           
            # multihead attention [shape: (mini_batch_size * data_chunk_length, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
            somu_output = self.somu_multi_att_layer_list[i](somu_lstm_hidden_state, somu_output, somu_output)[0]
            scmu_output = self.somu_multi_att_layer_list[i](scmu_lstm_hidden_state, scmu_output, scmu_output)[0]

            # concatenate obs and outputs from gnn, somu and scmu 
            # [shape: (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * obs_dims + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size) / 
            # (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            output = torch.cat((gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims)\
                                if self.rni else (self.n_gnn_layers + 1) * self.obs_dims), 
                                somu_output.reshape(mini_batch_size * self.data_chunk_length, self.somu_n_layers * self.somu_lstm_hidden_size), 
                                scmu_output.reshape(mini_batch_size * self.data_chunk_length, self.scmu_n_layers * self.scmu_lstm_hidden_size)), dim=-1)
            # output [shape: (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * obs_dims + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size) /  
            # (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            # --> 
            # fc_layers [shape: (mini_batch_size * data_chunk_length, fc_output_dims)]
            output = self.fc_layers(output)
            # fc_layers --> act [shape: (mini_batch_size * data_chunk_length, action_space_dim)], [shape: () == scalar]
            action_log_probs, dist_entropy = self.act.evaluate_actions(output, 
                                                                       action[:, i, :], available_actions[:, i, :] if available_actions is not None else None, 
                                                                       active_masks[:, i, :] if self._use_policy_active_masks and active_masks is not None else None)
            # append action_log_probs and dist_entropy to respective lists
            action_log_probs_list.append(action_log_probs)
            dist_entropy_list.append(dist_entropy)

        # [shape: (mini_batch_size * data_chunk_length * num_agents, action_space_dim)] and [shape: () == scalar]
        return torch.stack(action_log_probs_list, dim=1).reshape(mini_batch_size * self.data_chunk_length * self.num_agents, -1), torch.stack(dist_entropy_list, dim=0).mean()


class GCMNetDNAGATv2Critic(nn.Module):
    """
    GCMNetDNAGATv2 Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO)
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """ 
        class constructor for attributes for the critic model 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()

        self.data_chunk_length = args.data_chunk_length
        self._use_orthogonal = args.use_orthogonal
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        self.num_agents = args.num_agents
        self.n_rollout_threads = args.n_rollout_threads
        self.n_gnn_layers = args.n_gnn_layers
        self.somu_n_layers = args.somu_n_layers
        self.scmu_n_layers = args.scmu_n_layers
        self.somu_lstm_hidden_size = args.somu_lstm_hidden_size
        self.scmu_lstm_hidden_size = args.scmu_lstm_hidden_size
        self.somu_multi_att_n_heads = args.somu_multi_att_n_heads
        self.scmu_multi_att_n_heads = args.scmu_multi_att_n_heads
        self.fc_output_dims = args.fc_output_dims
        self.n_fc_layers = args.n_fc_layers
        self.knn = args.knn
        self.k = args.k
        self.rni = args.rni
        self.rni_ratio = args.rni_ratio

        cent_obs_space = get_shape_from_obs_space(cent_obs_space)
        if isinstance(cent_obs_space, list):
            self.obs_dims = cent_obs_space[0]
        else:
            self.obs_dims = cent_obs_space

        self.rni_dims = round(self.obs_dims * self.rni_ratio)

        # model architecture for mappo GCMNetDNAGATv2 critic

        # gnn layers
        self.gnn_layers = DNAGATv2Layers(input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                                         block=DNAGATv2Block, 
                                         output_channels=[self.obs_dims + self.rni_dims if self.rni\
                                                          else self.obs_dims for i in range(self.n_gnn_layers)], 
                                         concat=False)

        # list of lstms for self observation memory unit (somu) for each agent
        # somu_lstm_input_size is the dimension of the observations
        self.somu_lstm_list = nn.ModuleList([nn.LSTM(input_size=self.obs_dims, hidden_size=self.somu_lstm_hidden_size, 
                                                     num_layers=self.somu_n_layers, batch_first=True).to(device) 
                                             for _ in range(self.num_agents)])

        # list of lstms for self communication memory unit (scmu) for each agent
        # somu_lstm_input_size are all layers of gnn
        self.scmu_lstm_list = nn.ModuleList([nn.LSTM(input_size=(self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims)\
                                                                if self.rni else (self.n_gnn_layers + 1) * self.obs_dims, 
                                                     hidden_size=self.scmu_lstm_hidden_size, 
                                                     num_layers=self.scmu_n_layers, batch_first=True).to(device) 
                                             for _ in range(self.num_agents)])

        # multi-head self attention layer for somu and scmu to selectively choose between the lstms outputs
        self.somu_multi_att_layer_list = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.somu_lstm_hidden_size, 
                                                                              num_heads=self.somu_multi_att_n_heads, 
                                                                              dropout=0, batch_first=True, device=device) 
                                                        for _ in range(self.num_agents)])
        self.scmu_multi_att_layer_list = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.scmu_lstm_hidden_size, 
                                                                              num_heads=self.scmu_multi_att_n_heads, 
                                                                              dropout=0, batch_first=True, device=device) 
                                                        for _ in range(self.num_agents)])

        # shared hidden fc layers for to generate actions for each agent
        # input channels are all layers of gnn (including initial observations) 
        # + concatenated outputs of somu_multi_att_layer and scmu_multi_att_layer
        # fc_output_dims is the list of sizes of output channels fc_block
        self.fc_layers = NNLayers(input_channels=(self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) + \
                                                 self.somu_n_layers * self.somu_lstm_hidden_size + \
                                                 self.scmu_n_layers * self.scmu_lstm_hidden_size if self.rni
                                                 else (self.n_gnn_layers + 1) * self.obs_dims + \
                                                 self.somu_n_layers * self.somu_lstm_hidden_size + \
                                                 self.scmu_n_layers * self.scmu_lstm_hidden_size, 
                                  block=MLPBlock, output_channels=[self.fc_output_dims for i in range(self.n_fc_layers)], 
                                  activation_func='relu', dropout_p=0, 
                                  weight_initialisation="orthogonal" if self._use_orthogonal else "default").to(device)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # final layer for value function using popart / mlp
        if self._use_popart:
            self.v_out = init_(PopArt(self.fc_output_dims, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.fc_output_dims, 1))
        
        self.to(device)

    def forward(self, cent_obs, somu_hidden_states_critic, somu_cell_states_critic, scmu_hidden_states_critic, 
                scmu_cell_states_critic, masks):
        """
        Compute value function
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param somu_hidden_states_critic: (np.ndarray / torch.Tensor) hidden states for somu network.
        :param somu_cell_states_critic: (np.ndarray / torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_critic: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_critic: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return somu_hidden_states_critic: (torch.Tensor) hidden states for somu network.
        :return somu_cell_states_critic: (torch.Tensor) cell states for somu network.
        :return scmu_hidden_states_critic: (torch.Tensor) hidden states for scmu network.
        :return scmu_cell_states_critic: (torch.Tensor) hidden states for scmu network.
        """
        if self.knn:
            # [shape: (batch_size, num_agents, obs_dims)]
            obs = check(cent_obs) 
            batch_size = obs.shape[0]
            # [shape: (batch_size * num_agents, obs_dims)]
            obs_temp = obs.reshape(batch_size * self.num_agents, self.obs_dims)
            batch = torch.tensor([i // self.num_agents for i in range(batch_size * self.num_agents)])
            edge_index = knn_graph(x=obs_temp, k=self.k, batch=batch, loop=True)
            obs = obs.to(**self.tpdv)  
            if self.rni:
                # zero mean std 1 gaussian noise
                # [shape: (batch_size, num_agents, rni_dims)] 
                rni = torch.normal(0, 1, size=(batch_size, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # [shape: (batch_size, num_agents, obs_dims + rni_dims)]
                obs_rni = torch.cat((obs, rni), dim=-1)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni[i, :, :] if self.rni else obs[i, :, :], edge_index=edge_index) 
                                            for i in range(batch_size)]).to(self.device)
        else:
            # obtain edge index
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
            # [shape: (batch_size, num_agents, obs_dims)]  
            obs = check(cent_obs).to(**self.tpdv)  
            batch_size = obs.shape[0]
            if self.rni:
                # zero mean std 1 gaussian noise 
                # [shape: (batch_size, num_agents, rni_dims)] 
                rni = torch.normal(0, 1, size=(batch_size, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # [shape: (batch_size, num_agents, obs_dims + rni_dims)]
                obs_rni = torch.cat((obs, rni), dim=-1)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni[i, :, :] if self.rni else obs[i, :, :], edge_index=edge_index) 
                                            for i in range(batch_size)]).to(self.device)
        # shape: (batch_size, num_agents, somu_n_layers, somu_lstm_hidden_size)
        somu_hidden_states_critic = check(somu_hidden_states_critic).to(**self.tpdv)
        somu_cell_states_critic = check(somu_cell_states_critic).to(**self.tpdv)
        # shape: (batch_size, num_agents, scmu_n_layers, scmu_lstm_hidden_size) 
        scmu_hidden_states_critic = check(scmu_hidden_states_critic).to(**self.tpdv)
        scmu_cell_states_critic = check(scmu_cell_states_critic).to(**self.tpdv) 
        # shape: (batch_size, num_agents, 1)
        masks = check(masks).to(**self.tpdv).reshape(batch_size, self.num_agents, -1) 
        # store somu and scmu hidden states and cell states and values
        somu_lstm_hidden_state_list = []
        somu_lstm_cell_state_list = []
        scmu_lstm_hidden_state_list = []
        scmu_lstm_cell_state_list = []
        values_list = []
       
        # obs_gnn.x [shape: (batch_size * num_agents, obs_dims / (obs_dims + rni_dims))] 
        # --> gnn_layers [shape: (batch_size, num_agents, (n_gnn_layers + 1) * obs_dims / (n_gnn_layers + 1) * (obs_dims + rni_dims))]
        gnn_output = self.gnn_layers(x=obs_gnn.x, edge_index=obs_gnn.edge_index)\
                         .reshape(batch_size, self.num_agents, (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) if self.rni else\
                                  (self.n_gnn_layers + 1) * self.obs_dims)
       
        # iterate over agents 
        for i in range(self.num_agents):
            # obs[:, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, obs_dims)],
            # masks[:, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (somu_n_layers, batch_size, 1)],
            # (h_0 [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)]) -->
            # somu_output [shape: (batch_size, sequence_length=1, somu_lstm_hidden_size)], 
            # (h_n [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)], c_n [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)])
            somu_output, somu_hidden_cell_states_tup = self.somu_lstm_list[i](obs[:, i, :].unsqueeze(dim=1), 
                                                                              (somu_hidden_states_critic.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.somu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous(), 
                                                                               somu_cell_states_critic.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.somu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous()))
            # gnn_output[:, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, (n_gnn_layers + 1) * obs_dims)],
            # masks[:, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (scmu_n_layers, batch_size, 1)], 
            # (h_0 [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)], c_0 [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)]) -->
            # scmu_output [shape: (batch_size, sequence_length=1, scmu_lstm_hidden_state)], 
            # (h_n [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)], c_n [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)])
            scmu_output, scmu_hidden_cell_states_tup = self.scmu_lstm_list[i](gnn_output[:, i, :].unsqueeze(dim=1), 
                                                                              (scmu_hidden_states_critic.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.scmu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous(), 
                                                                               scmu_cell_states_critic.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.scmu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous()))
            # (h_n [shape: (batch_size, somu_n_layers / scmu_n_layers, scmu_lstm_hidden_state)], 
            # c_n [shape: (batch_size, somu_n_layers / scmu_n_layers, scmu_lstm_hidden_state)])
            somu_lstm_hidden_state_list.append(somu_hidden_cell_states_tup[0].transpose(0, 1))
            somu_lstm_cell_state_list.append(somu_hidden_cell_states_tup[1].transpose(0, 1))
            scmu_lstm_hidden_state_list.append(scmu_hidden_cell_states_tup[0].transpose(0, 1))
            scmu_lstm_cell_state_list.append(scmu_hidden_cell_states_tup[1].transpose(0, 1))
            # somu_output / scmu_output [shape: (batch_size, sequence_length=1, somu_lstm_hidden_size / scmu_lstm_hidden_state)],
            # somu_hidden_state / scmu_hidden_state [shape: (batch_size, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)] 
            # --> multihead attention [shape: (batch_size, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
            somu_output = self.somu_multi_att_layer_list[i](somu_lstm_hidden_state_list[i], somu_output, somu_output)[0]
            scmu_output = self.somu_multi_att_layer_list[i](scmu_lstm_hidden_state_list[i], scmu_output, scmu_output)[0]

            # concatenate obs and outputs from gnn, somu and scmu 
            # [shape: (batch_size, (n_gnn_layers + 1) * obs_dims + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size) / 
            # (batch_size, (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            output = torch.cat((gnn_output[:, i, :], somu_output.reshape(batch_size, -1), scmu_output.reshape(batch_size, -1)), dim=-1)
            # output [shape: (batch_size, (n_gnn_layers + 1) * self.obs_dims + somu_lstm_hidden_size + scmu_lstm_hidden_size) / 
            # (batch_size, (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            # -->   
            # fc_layers [shape: (batch_size, fc_output_dims)]
            output = self.fc_layers(output)
            # output --> v_out [shape: (batch_size, 1)]
            values = self.v_out(output)
            values_list.append(values)
       
        # [shape: (batch_size, num_agents, 1)]
        # [shape: (batch_size, num_agents, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
        return torch.stack(values_list, dim=1), torch.stack(somu_lstm_hidden_state_list, dim=1), torch.stack(somu_lstm_cell_state_list, dim=1), \
               torch.stack(scmu_lstm_hidden_state_list, dim=1), torch.stack(scmu_lstm_cell_state_list, dim=1)

    def evaluate_actions(self, cent_obs, somu_hidden_states_critic, somu_cell_states_critic, scmu_hidden_states_critic, 
                         scmu_cell_states_critic, masks):
        """
        Compute value function
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param somu_hidden_states_critic: (np.ndarray / torch.Tensor) hidden states for somu network.
        :param somu_cell_states_critic: (np.ndarray / torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_critic: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_critic: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        """
        if self.knn:
            # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims]   
            obs = check(cent_obs)
            mini_batch_size = obs.shape[0]
            # [shape: (mini_batch_size * data_chunk_length * num_agents, obs_dims)] 
            obs_temp = obs.reshape(mini_batch_size * self.data_chunk_length * self.num_agents, self.obs_dims)
            batch = torch.tensor([i // self.num_agents for i in range(mini_batch_size * self.data_chunk_length * self.num_agents)])
            edge_index = knn_graph(x=obs_temp, k=self.k, batch=batch, loop=True)
            obs = obs.to(**self.tpdv)
            # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims)] 
            obs_batch = obs.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims)
            if self.rni:
                # zero mean std 1 gaussian noise 
                # [shape: (mini_batch_size, data_chunk_length, num_agents, rni_dims)] 
                rni = torch.normal(0, 1, size=(mini_batch_size, self.data_chunk_length, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims + rni_dims)]
                obs_rni = torch.cat((obs, rni), dim=-1)
                # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims + rni_dims)] 
                obs_rni_batch = obs_rni.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims + self.rni_dims)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni_batch[i, :, :] if self.rni else obs_batch[i, :, :], edge_index=edge_index) 
                                            for i in range(mini_batch_size * self.data_chunk_length)]).to(self.device)
        else:
            # obtain edge index
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
            # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims)]   
            obs = check(cent_obs).to(**self.tpdv)
            mini_batch_size = obs.shape[0]
            # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims)] 
            obs_batch = obs.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims)
            if self.rni:
                # zero mean std 1 gaussian noise 
                # shape: (mini_batch_size, data_chunk_length, num_agents, rni_dims) 
                rni = torch.normal(0, 1, size=(mini_batch_size, self.data_chunk_length, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims + rni_dims)
                obs_rni = torch.cat((obs, rni), dim=-1)
                # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims + rni_dims)] 
                obs_rni_batch = obs_rni.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims + self.rni_dims)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni_batch[i, :, :] if self.rni else obs_batch[i, :, :], edge_index=edge_index) 
                                            for i in range(mini_batch_size * self.data_chunk_length)]).to(self.device)
        # [shape: (mini_batch_size, num_agents, somu_n_layers, somu_lstm_hidden_size)]
        somu_hidden_states_critic = check(somu_hidden_states_critic).to(**self.tpdv) 
        somu_cell_states_critic = check(somu_cell_states_critic).to(**self.tpdv)
        # [shape: (mini_batch_size, num_agents, scmu_n_layers, scmu_lstm_hidden_size)]
        scmu_hidden_states_critic = check(scmu_hidden_states_critic).to(**self.tpdv) 
        scmu_cell_states_critic = check(scmu_cell_states_critic).to(**self.tpdv)
        # [shape: (mini_batch_size, data_chunk_length, num_agents, 1)]  
        masks = check(masks).to(**self.tpdv)
        # store values
        values_list = []

        # obs_gnn.x [shape: (mini_batch_size * data_chunk_length * num_agents, obs_dims / (obs_dims + rni_dims))] -->
        # gnn_layers [shape: (mini_batch_size, data_chunk_length, num_agents, (n_gnn_layers + 1) * obs_dims \ 
        # (n_gnn_layers + 1) * (obs_dims + rni_dims))]
        gnn_output = self.gnn_layers(x=obs_gnn.x, edge_index=obs_gnn.edge_index)\
                         .reshape(mini_batch_size, self.data_chunk_length, self.num_agents, 
                                  (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) if self.rni else\
                                  (self.n_gnn_layers + 1) * self.obs_dims)

        # iterate over agents 
        for i in range(self.num_agents):
            # list to store somu and scmu outputs, hidden states and cell states over sequence of inputs of len, data_chunk_length
            somu_seq_output_list = []
            somu_seq_lstm_hidden_state_list = []
            somu_seq_lstm_cell_state_list = []
            scmu_seq_output_list = []
            scmu_seq_lstm_hidden_state_list = []
            scmu_seq_lstm_cell_state_list = []
            # initialise hidden and cell states for somu and scmu at start of sequence
            # (h_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
            somu_seq_lstm_hidden_state_list.append(somu_hidden_states_critic[:, i, :, :].transpose(0, 1).contiguous())
            somu_seq_lstm_cell_state_list.append(somu_cell_states_critic[:, i, :, :].transpose(0, 1).contiguous())
            scmu_seq_lstm_hidden_state_list.append(scmu_hidden_states_critic[:, i, :, :].transpose(0, 1).contiguous())
            scmu_seq_lstm_cell_state_list.append(scmu_cell_states_critic[:, i, :, :].transpose(0, 1).contiguous())

            # iterate over data_chunk_length
            for j in range(self.data_chunk_length):
                # obs[:, j, i, :].unsqueeze(dim=1) [shape: (mini_batch_size, sequence_length=1, obs_dims)],
                # masks[:, j, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (somu_n_layers, batch_size, 1)],
                # (h_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
                # -->
                # somu_output [shape: (mini_batch_size, sequence_length=1, somu_lstm_hidden_size)], 
                # (h_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], c_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
                somu_output, somu_hidden_cell_states_tup = self.somu_lstm_list[i](obs[:, j, i, :].unsqueeze(dim=1),
                                                                                  (somu_seq_lstm_hidden_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.somu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous(),
                                                                                   somu_seq_lstm_cell_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.somu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous()))
                # gnn_output[:, j, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, (n_gnn_layers + 1) * obs_dims)],
                # masks[:, j, i, :].repeat(1, self.scmu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (scmu_n_layers, batch_size, 1)],
                # (h_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], c_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)])
                # -->
                # scmu_output [shape: (mini_batch_size, sequence_length=1, scmu_lstm_hidden_size)], 
                # (h_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], c_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)])
                scmu_output, scmu_hidden_cell_states_tup = self.scmu_lstm_list[i](gnn_output[:, j, i, :].unsqueeze(dim=1),
                                                                                  (scmu_seq_lstm_hidden_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.scmu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous(),
                                                                                   scmu_seq_lstm_cell_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.scmu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous()))
                somu_seq_output_list.append(somu_output)
                scmu_seq_output_list.append(scmu_output)
                somu_seq_lstm_hidden_state_list.append(somu_hidden_cell_states_tup[0])
                somu_seq_lstm_cell_state_list.append(somu_hidden_cell_states_tup[1])
                scmu_seq_lstm_hidden_state_list.append(scmu_hidden_cell_states_tup[0])
                scmu_seq_lstm_cell_state_list.append(scmu_hidden_cell_states_tup[1])

            # [shape: (mini_batch_size * data_chunk_length, 1, somu_lstm_hidden_size / scmu_lstm_hidden_state)]
            somu_output = torch.stack(somu_seq_output_list, dim=1).reshape(mini_batch_size * self.data_chunk_length, 1, self.somu_lstm_hidden_size)
            scmu_output = torch.stack(scmu_seq_output_list, dim=1).reshape(mini_batch_size * self.data_chunk_length, 1, self.scmu_lstm_hidden_size)
            # [shape: (data_chunk_length, somu_n_layers / scmu_n_layers, mini_batch_size, somu_lstm_hidden_state / scmu_lstm_hidden_state)] --> 
            # [shape: (mini_batch_size * data_chunk_length, somu_n_layers / scmu_n_layers, somu_lstm_hidden_state / scmu_lstm_hidden_state)]
            somu_lstm_hidden_state = torch.stack(somu_seq_lstm_hidden_state_list[1:], dim=0).permute(2, 0, 1, 3).reshape(mini_batch_size * self.data_chunk_length, 
                                                                                                                         self.somu_n_layers, 
                                                                                                                         self.somu_lstm_hidden_size)
            scmu_lstm_hidden_state = torch.stack(scmu_seq_lstm_hidden_state_list[1:], dim=0).permute(2, 0, 1, 3).reshape(mini_batch_size * self.data_chunk_length, 
                                                                                                                         self.scmu_n_layers, 
                                                                                                                         self.scmu_lstm_hidden_size)
           
            # multihead attention [shape: (mini_batch_size * data_chunk_length, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
            somu_output = self.somu_multi_att_layer_list[i](somu_lstm_hidden_state, somu_output, somu_output)[0]
            scmu_output = self.somu_multi_att_layer_list[i](scmu_lstm_hidden_state, scmu_output, scmu_output)[0]

            # concatenate obs and outputs from gnn, somu and scmu 
            # [shape: (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * obs_dims + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size) / 
            # (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            output = torch.cat((gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims)\
                                if self.rni else (self.n_gnn_layers + 1) * self.obs_dims), 
                                somu_output.reshape(mini_batch_size * self.data_chunk_length, self.somu_n_layers * self.somu_lstm_hidden_size), 
                                scmu_output.reshape(mini_batch_size * self.data_chunk_length, self.scmu_n_layers * self.scmu_lstm_hidden_size)), dim=-1)
            # output [shape: (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * obs_dims + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size) /  
            # (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            # --> 
            # fc_layers [shape: (mini_batch_size * data_chunk_length, fc_output_dims)]
            output = self.fc_layers(output)
            # output --> v_out [shape: (mini_batch_size * data_chunk_length, 1)]
            values = self.v_out(output)
            values_list.append(values)

        # [shape: (mini_batch_size * data_chunk_length * num_agents, 1)]
        return torch.stack(values_list, dim=1).reshape(-1, 1)


class GCMNetGINActor(nn.Module):
    """
    GCMNetGIN Actor network class for MAPPO. Outputs actions given observations.
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

        self.data_chunk_length = args.data_chunk_length
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device

        self.num_agents = args.num_agents
        self.n_rollout_threads = args.n_rollout_threads
        self.n_gnn_layers = args.n_gnn_layers
        self.somu_n_layers = args.somu_n_layers
        self.scmu_n_layers = args.scmu_n_layers
        self.somu_lstm_hidden_size = args.somu_lstm_hidden_size
        self.scmu_lstm_hidden_size = args.scmu_lstm_hidden_size
        self.somu_multi_att_n_heads = args.somu_multi_att_n_heads
        self.scmu_multi_att_n_heads = args.scmu_multi_att_n_heads
        self.fc_output_dims = args.fc_output_dims
        self.n_fc_layers = args.n_fc_layers
        self.knn = args.knn
        self.k = args.k
        self.rni = args.rni
        self.rni_ratio = args.rni_ratio
        self.n_gin_fc_layers = args.n_gin_fc_layers

        obs_shape = get_shape_from_obs_space(obs_space)
        if isinstance(obs_shape, list):
            self.obs_dims = obs_shape[0]
        else:
            self.obs_dims = obs_shape

        self.rni_dims = round(self.obs_dims * self.rni_ratio)

        # model architecture for mappo GCMNetGIN actor

        # gnn layers
        self.gnn_layers = GINLayers(input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                                    block=GINBlock, 
                                    output_channels=[self.obs_dims + self.rni_dims if self.rni\
                                                     else self.obs_dims for i in range(self.n_gnn_layers)],
                                    n_gin_fc_layers=self.n_gin_fc_layers)

        # list of lstms for self observation memory unit (somu) for each agent
        # somu_lstm_input_size is the dimension of the observations
        self.somu_lstm_list = nn.ModuleList([nn.LSTM(input_size=self.obs_dims, hidden_size=self.somu_lstm_hidden_size, 
                                                     num_layers=self.somu_n_layers, batch_first=True).to(device) 
                                             for _ in range(self.num_agents)])

        # list of lstms for self communication memory unit (scmu) for each agent
        # somu_lstm_input_size are all layers of gnn
        self.scmu_lstm_list = nn.ModuleList([nn.LSTM(input_size=(self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims)\
                                                                if self.rni else (self.n_gnn_layers + 1) * self.obs_dims, 
                                                     hidden_size=self.scmu_lstm_hidden_size, 
                                                     num_layers=self.scmu_n_layers, batch_first=True).to(device) 
                                             for _ in range(self.num_agents)])

        # multi-head self attention layer for somu and scmu to selectively choose between the lstms outputs
        self.somu_multi_att_layer_list = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.somu_lstm_hidden_size, 
                                                                              num_heads=self.somu_multi_att_n_heads, 
                                                                              dropout=0, batch_first=True, device=device) 
                                                        for _ in range(self.num_agents)])
        self.scmu_multi_att_layer_list = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.scmu_lstm_hidden_size, 
                                                                              num_heads=self.scmu_multi_att_n_heads, 
                                                                              dropout=0, batch_first=True, device=device) 
                                                        for _ in range(self.num_agents)])

        # shared hidden fc layers for to generate actions for each agent
        # input channels are all layers of gnn (including initial observations) 
        # + concatenated outputs of somu_multi_att_layer and scmu_multi_att_layer
        # fc_output_dims is the list of sizes of output channels fc_block
        self.fc_layers = NNLayers(input_channels=(self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) + \
                                                 self.somu_n_layers * self.somu_lstm_hidden_size + \
                                                 self.scmu_n_layers * self.scmu_lstm_hidden_size if self.rni
                                                 else (self.n_gnn_layers + 1) * self.obs_dims + \
                                                 self.somu_n_layers * self.somu_lstm_hidden_size + \
                                                 self.scmu_n_layers * self.scmu_lstm_hidden_size, 
                                  block=MLPBlock, output_channels=[self.fc_output_dims for i in range(self.n_fc_layers)], 
                                  activation_func='relu', dropout_p=0, 
                                  weight_initialisation="orthogonal" if self._use_orthogonal else "default").to(device)

        # shared final action layer for each agent
        self.act = SingleACTLayer(action_space, self.fc_output_dims, self._use_orthogonal, self._gain).to(device)
        
        self.to(device)
        
    def forward(self, obs, somu_hidden_states_actor, somu_cell_states_actor, scmu_hidden_states_actor, scmu_cell_states_actor, 
                masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param somu_hidden_states_actor: (np.ndarray / torch.Tensor) hidden states for somu network.
        :param somu_cell_states_actor: (np.ndarray / torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_actor: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_actor: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return somu_hidden_states_actor: (np.ndarray / torch.Tensor) hidden states for somu network.
        :return somu_cell_states_actor: (np.ndarray / torch.Tensor) cell states for somu network.
        :return scmu_hidden_states_actor: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :return scmu_cell_states_actor: (np.ndarray / torch.Tensor) hidden states for scmu network.
        """
        if self.knn:
            # [shape: (batch_size, num_agents, obs_dims)]
            obs = check(obs) 
            batch_size = obs.shape[0]
            # [shape: (batch_size * num_agents, obs_dims)]
            obs_temp = obs.reshape(batch_size * self.num_agents, self.obs_dims)
            batch = torch.tensor([i // self.num_agents for i in range(batch_size * self.num_agents)])
            edge_index = knn_graph(x=obs_temp, k=self.k, batch=batch, loop=True)
            obs = obs.to(**self.tpdv)  
            if self.rni:
                # zero mean std 1 gaussian noise
                # [shape: (batch_size, num_agents, rni_dims)] 
                rni = torch.normal(0, 1, size=(batch_size, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # [shape: (batch_size, num_agents, obs_dims + rni_dims)]
                obs_rni = torch.cat((obs, rni), dim=-1)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni[i, :, :] if self.rni else obs[i, :, :], edge_index=edge_index) 
                                            for i in range(batch_size)]).to(self.device)
        else:
            # obtain edge index
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
            # [shape: (batch_size, num_agents, obs_dims)]  
            obs = check(obs).to(**self.tpdv)  
            batch_size = obs.shape[0]
            if self.rni:
                # zero mean std 1 gaussian noise 
                # [shape: (batch_size, num_agents, rni_dims)] 
                rni = torch.normal(0, 1, size=(batch_size, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # [shape: (batch_size, num_agents, obs_dims + rni_dims)]
                obs_rni = torch.cat((obs, rni), dim=-1)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni[i, :, :] if self.rni else obs[i, :, :], edge_index=edge_index) 
                                            for i in range(batch_size)]).to(self.device)
        # [shape: (batch_size, num_agents, somu_n_layers, somu_lstm_hidden_size)]
        somu_hidden_states_actor = check(somu_hidden_states_actor).to(**self.tpdv)
        somu_cell_states_actor = check(somu_cell_states_actor).to(**self.tpdv)
        # [shape: (batch_size, num_agents, scmu_n_layers, scmu_lstm_hidden_size)] 
        scmu_hidden_states_actor = check(scmu_hidden_states_actor).to(**self.tpdv)
        scmu_cell_states_actor = check(scmu_cell_states_actor).to(**self.tpdv) 
        # [shape: (batch_size, num_agents, 1)]
        masks = check(masks).to(**self.tpdv).reshape(batch_size, self.num_agents, -1) 
        if available_actions is not None:
            # [shape: (batch_size, num_agents, action_space_dim)]
            available_actions = check(available_actions).to(**self.tpdv) 
        # store somu and scmu hidden states and cell states, actions and action_log_probs
        somu_lstm_hidden_state_list = []
        somu_lstm_cell_state_list = []
        scmu_lstm_hidden_state_list = []
        scmu_lstm_cell_state_list = []
        actions_list = []
        action_log_probs_list = []
       
        # obs_gnn.x [shape: (batch_size * num_agents, obs_dims / (obs_dims + rni_dims))] 
        # --> gnn_layers [shape: (batch_size, num_agents, (n_gnn_layers + 1) * obs_dims / (n_gnn_layers + 1) * (obs_dims + rni_dims))]
        gnn_output = self.gnn_layers(x=obs_gnn.x, edge_index=obs_gnn.edge_index)\
                         .reshape(batch_size, self.num_agents, (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) if self.rni else\
                                  (self.n_gnn_layers + 1) * self.obs_dims)
        # sum embeddings across nodes at each iteration that are concatenated together
        # gnn_output_sum [shape: (batch_size, (n_gnn_layers + 1) * obs_dims / (n_gnn_layers + 1) * (obs_dims + rni_dims))]
        gnn_output_sum = gnn_output.sum(dim=1)
       
        # iterate over agents 
        for i in range(self.num_agents):
            # obs[:, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, obs_dims)],
            # masks[:, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (somu_n_layers, batch_size, 1)],
            # (h_0 [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)]) -->
            # somu_output [shape: (batch_size, sequence_length=1, somu_lstm_hidden_size)], 
            # (h_n [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)], c_n [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)])
            somu_output, somu_hidden_cell_states_tup = self.somu_lstm_list[i](obs[:, i, :].unsqueeze(dim=1), 
                                                                              (somu_hidden_states_actor.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.somu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous(), 
                                                                               somu_cell_states_actor.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.somu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous()))
            # gnn_output[:, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, (n_gnn_layers + 1) * obs_dims)],
            # masks[:, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (scmu_n_layers, batch_size, 1)], 
            # (h_0 [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)], c_0 [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)]) -->
            # scmu_output [shape: (batch_size, sequence_length=1, scmu_lstm_hidden_state)], 
            # (h_n [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)], c_n [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)])
            scmu_output, scmu_hidden_cell_states_tup = self.scmu_lstm_list[i](gnn_output[:, i, :].unsqueeze(dim=1), 
                                                                              (scmu_hidden_states_actor.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.scmu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous(), 
                                                                               scmu_cell_states_actor.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.scmu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous()))
            # (h_n [shape: (batch_size, somu_n_layers / scmu_n_layers, scmu_lstm_hidden_state)], 
            # c_n [shape: (batch_size, somu_n_layers / scmu_n_layers, scmu_lstm_hidden_state)])
            somu_lstm_hidden_state_list.append(somu_hidden_cell_states_tup[0].transpose(0, 1))
            somu_lstm_cell_state_list.append(somu_hidden_cell_states_tup[1].transpose(0, 1))
            scmu_lstm_hidden_state_list.append(scmu_hidden_cell_states_tup[0].transpose(0, 1))
            scmu_lstm_cell_state_list.append(scmu_hidden_cell_states_tup[1].transpose(0, 1))
            # somu_output / scmu_output [shape: (batch_size, sequence_length=1, somu_lstm_hidden_size / scmu_lstm_hidden_state)],
            # somu_hidden_state / scmu_hidden_state [shape: (batch_size, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)] 
            # --> multihead attention [shape: (batch_size, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
            somu_output = self.somu_multi_att_layer_list[i](somu_lstm_hidden_state_list[i], somu_output, somu_output)[0]
            scmu_output = self.somu_multi_att_layer_list[i](scmu_lstm_hidden_state_list[i], scmu_output, scmu_output)[0]

            # concatenate obs and outputs from gnn, somu and scmu 
            # [shape: (batch_size, (n_gnn_layers + 1) * obs_dims + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size) / 
            # (batch_size, (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            output = torch.cat((gnn_output_sum, somu_output.reshape(batch_size, -1), scmu_output.reshape(batch_size, -1)), dim=-1)
            # output [shape: (batch_size, (n_gnn_layers + 1) * self.obs_dims + somu_lstm_hidden_size + scmu_lstm_hidden_size) / 
            # (batch_size, (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            # -->   
            # fc_layers [shape: (batch_size, fc_output_dims)]
            output = self.fc_layers(output)
            # fc_layers --> act [shape: (batch_size, action_space_dim)]
            actions, action_log_probs = self.act(output, available_actions[:, i, :] if available_actions is not None else None, deterministic)
            # append actions and action_log_probs to respective lists
            actions_list.append(actions)
            action_log_probs_list.append(action_log_probs)
       
        # [shape: (batch_size, num_agents, action_space_dim)]
        # [shape: (batch_size, num_agents, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
        return torch.stack(actions_list, dim=1), torch.stack(action_log_probs_list, dim=1), \
               torch.stack(somu_lstm_hidden_state_list, dim=1), torch.stack(somu_lstm_cell_state_list, dim=1), \
               torch.stack(scmu_lstm_hidden_state_list, dim=1), torch.stack(scmu_lstm_cell_state_list, dim=1)

    def evaluate_actions(self, obs, somu_hidden_states_actor, somu_cell_states_actor, scmu_hidden_states_actor, scmu_cell_states_actor, 
                         action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param somu_hidden_states_actor: (torch.Tensor) hidden states for somu network.
        :param somu_cell_states_actor: (torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_actor: (torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_actor: (torch.Tensor) hidden states for scmu network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.knn:
            # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims]   
            obs = check(obs)
            mini_batch_size = obs.shape[0]
            # [shape: (mini_batch_size * data_chunk_length * num_agents, obs_dims)] 
            obs_temp = obs.reshape(mini_batch_size * self.data_chunk_length * self.num_agents, self.obs_dims)
            batch = torch.tensor([i // self.num_agents for i in range(mini_batch_size * self.data_chunk_length * self.num_agents)])
            edge_index = knn_graph(x=obs_temp, k=self.k, batch=batch, loop=True)
            obs = obs.to(**self.tpdv)
            # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims)] 
            obs_batch = obs.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims)
            if self.rni:
                # zero mean std 1 gaussian noise 
                # [shape: (mini_batch_size, data_chunk_length, num_agents, rni_dims)] 
                rni = torch.normal(0, 1, size=(mini_batch_size, self.data_chunk_length, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims + rni_dims)]
                obs_rni = torch.cat((obs, rni), dim=-1)
                # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims + rni_dims)] 
                obs_rni_batch = obs_rni.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims + self.rni_dims)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni_batch[i, :, :] if self.rni else obs_batch[i, :, :], edge_index=edge_index) 
                                            for i in range(mini_batch_size * self.data_chunk_length)]).to(self.device)
        else:
            # obtain edge index
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
            # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims)]   
            obs = check(obs).to(**self.tpdv)
            mini_batch_size = obs.shape[0]
            # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims)] 
            obs_batch = obs.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims)
            if self.rni:
                # zero mean std 1 gaussian noise 
                # shape: (mini_batch_size, data_chunk_length, num_agents, rni_dims) 
                rni = torch.normal(0, 1, size=(mini_batch_size, self.data_chunk_length, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims + rni_dims)
                obs_rni = torch.cat((obs, rni), dim=-1)
                # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims + rni_dims)] 
                obs_rni_batch = obs_rni.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims + self.rni_dims)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni_batch[i, :, :] if self.rni else obs_batch[i, :, :], edge_index=edge_index) 
                                            for i in range(mini_batch_size * self.data_chunk_length)]).to(self.device)
        # [shape: (mini_batch_size, data_chunk_length, num_agents, action_space_dim)]  
        action = check(action).to(**self.tpdv) 
        # [shape: (mini_batch_size, num_agents, somu_n_layers, somu_lstm_hidden_size)]
        somu_hidden_states_actor = check(somu_hidden_states_actor).to(**self.tpdv) 
        somu_cell_states_actor = check(somu_cell_states_actor).to(**self.tpdv)
        # [shape: (mini_batch_size, num_agents, scmu_n_layers, scmu_lstm_hidden_size)]
        scmu_hidden_states_actor = check(scmu_hidden_states_actor).to(**self.tpdv) 
        scmu_cell_states_actor = check(scmu_cell_states_actor).to(**self.tpdv)
        # [shape: (mini_batch_size, data_chunk_length, num_agents, 1)]  
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            # [shape: (mini_batch_size, data_chunk_length, num_agents, action_space_dim)]
            available_actions = check(available_actions).to(**self.tpdv) 
            # [shape: (mini_batch_size * data_chunk_length, num_agents, action_space_dim)]
            available_actions = available_actions.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, -1)
        if active_masks is not None:
            # [shape: (mini_batch_size, data_chunk_length, num_agents, 1)]
            active_masks = check(active_masks).to(**self.tpdv)
            # [shape: (mini_batch_size * data_chunk_length, num_agents, 1)]
            active_masks = active_masks.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, -1)
        # [shape: (mini_batch_size * data_chunk_length, num_agents, action_space_dim)] 
        action = action.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, -1)
        # store actions and actions_log_probs
        action_log_probs_list = []
        dist_entropy_list = []

        # obs_gnn.x [shape: (mini_batch_size * data_chunk_length * num_agents, obs_dims)] -->
        # gnn_layers [shape: (mini_batch_size, data_chunk_length, num_agents, (n_gnn_layers + 1) * obs_dims \ 
        # (n_gnn_layers + 1) * (obs_dims + rni_dims))]
        gnn_output = self.gnn_layers(x=obs_gnn.x, edge_index=obs_gnn.edge_index)\
                         .reshape(mini_batch_size, self.data_chunk_length, self.num_agents, 
                                  (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) if self.rni else\
                                  (self.n_gnn_layers + 1) * self.obs_dims)
        # sum embeddings across nodes at each iteration that are concatenated together
        # gnn_output_sum [shape: (mini_batch_size * data_chunk_length, (n_gnn_layers + 1) * obs_dims / (n_gnn_layers + 1) * (obs_dims + rni_dims))]
        gnn_output_sum = gnn_output.sum(dim=2)\
                                   .reshape(mini_batch_size * self.data_chunk_length, (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims)\
                                            if self.rni else (self.n_gnn_layers + 1) * self.obs_dims)

        # iterate over agents 
        for i in range(self.num_agents):
            # list to store somu and scmu outputs, hidden states and cell states over sequence of inputs of len, data_chunk_length
            somu_seq_output_list = []
            somu_seq_lstm_hidden_state_list = []
            somu_seq_lstm_cell_state_list = []
            scmu_seq_output_list = []
            scmu_seq_lstm_hidden_state_list = []
            scmu_seq_lstm_cell_state_list = []
            # initialise hidden and cell states for somu and scmu at start of sequence
            # (h_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
            somu_seq_lstm_hidden_state_list.append(somu_hidden_states_actor[:, i, :, :].transpose(0, 1).contiguous())
            somu_seq_lstm_cell_state_list.append(somu_cell_states_actor[:, i, :, :].transpose(0, 1).contiguous())
            scmu_seq_lstm_hidden_state_list.append(scmu_hidden_states_actor[:, i, :, :].transpose(0, 1).contiguous())
            scmu_seq_lstm_cell_state_list.append(scmu_cell_states_actor[:, i, :, :].transpose(0, 1).contiguous())

            # iterate over data_chunk_length
            for j in range(self.data_chunk_length):
                # obs[:, j, i, :].unsqueeze(dim=1) [shape: (mini_batch_size, sequence_length=1, obs_dims)],
                # masks[:, j, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (somu_n_layers, batch_size, 1)],
                # (h_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
                # -->
                # somu_output [shape: (mini_batch_size, sequence_length=1, somu_lstm_hidden_size)], 
                # (h_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], c_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
                somu_output, somu_hidden_cell_states_tup = self.somu_lstm_list[i](obs[:, j, i, :].unsqueeze(dim=1),
                                                                                  (somu_seq_lstm_hidden_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.somu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous(),
                                                                                   somu_seq_lstm_cell_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.somu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous()))
                # gnn_output[:, j, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, (n_gnn_layers + 1) * obs_dims)],
                # masks[:, j, i, :].repeat(1, self.scmu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (scmu_n_layers, batch_size, 1)],
                # (h_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], c_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)])
                # -->
                # scmu_output [shape: (mini_batch_size, sequence_length=1, scmu_lstm_hidden_size)], 
                # (h_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], c_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)])
                scmu_output, scmu_hidden_cell_states_tup = self.scmu_lstm_list[i](gnn_output[:, j, i, :].unsqueeze(dim=1),
                                                                                  (scmu_seq_lstm_hidden_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.scmu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous(),
                                                                                   scmu_seq_lstm_cell_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.scmu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous()))
                somu_seq_output_list.append(somu_output)
                scmu_seq_output_list.append(scmu_output)
                somu_seq_lstm_hidden_state_list.append(somu_hidden_cell_states_tup[0])
                somu_seq_lstm_cell_state_list.append(somu_hidden_cell_states_tup[1])
                scmu_seq_lstm_hidden_state_list.append(scmu_hidden_cell_states_tup[0])
                scmu_seq_lstm_cell_state_list.append(scmu_hidden_cell_states_tup[1])

            # [shape: (mini_batch_size * data_chunk_length, 1, somu_lstm_hidden_size / scmu_lstm_hidden_state)]
            somu_output = torch.stack(somu_seq_output_list, dim=1).reshape(mini_batch_size * self.data_chunk_length, 1, self.somu_lstm_hidden_size)
            scmu_output = torch.stack(scmu_seq_output_list, dim=1).reshape(mini_batch_size * self.data_chunk_length, 1, self.scmu_lstm_hidden_size)
            # [shape: (data_chunk_length, somu_n_layers / scmu_n_layers, mini_batch_size, somu_lstm_hidden_state / scmu_lstm_hidden_state)] --> 
            # [shape: (mini_batch_size * data_chunk_length, somu_n_layers / scmu_n_layers, somu_lstm_hidden_state / scmu_lstm_hidden_state)]
            somu_lstm_hidden_state = torch.stack(somu_seq_lstm_hidden_state_list[1:], dim=0).permute(2, 0, 1, 3).reshape(mini_batch_size * self.data_chunk_length, 
                                                                                                                         self.somu_n_layers, 
                                                                                                                         self.somu_lstm_hidden_size)
            scmu_lstm_hidden_state = torch.stack(scmu_seq_lstm_hidden_state_list[1:], dim=0).permute(2, 0, 1, 3).reshape(mini_batch_size * self.data_chunk_length, 
                                                                                                                         self.scmu_n_layers, 
                                                                                                                         self.scmu_lstm_hidden_size)
           
            # multihead attention [shape: (mini_batch_size * data_chunk_length, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
            somu_output = self.somu_multi_att_layer_list[i](somu_lstm_hidden_state, somu_output, somu_output)[0]
            scmu_output = self.somu_multi_att_layer_list[i](scmu_lstm_hidden_state, scmu_output, scmu_output)[0]

            # concatenate obs and outputs from gnn, somu and scmu 
            # [shape: (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * obs_dims + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size) / 
            # (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            output = torch.cat((gnn_output_sum, 
                                somu_output.reshape(mini_batch_size * self.data_chunk_length, self.somu_n_layers * self.somu_lstm_hidden_size), 
                                scmu_output.reshape(mini_batch_size * self.data_chunk_length, self.scmu_n_layers * self.scmu_lstm_hidden_size)), dim=-1)
            # output [shape: (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * obs_dims + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size) /  
            # (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            # --> 
            # fc_layers [shape: (mini_batch_size * data_chunk_length, fc_output_dims)]
            output = self.fc_layers(output)
            # fc_layers --> act [shape: (mini_batch_size * data_chunk_length, action_space_dim)], [shape: () == scalar]
            action_log_probs, dist_entropy = self.act.evaluate_actions(output, 
                                                                       action[:, i, :], available_actions[:, i, :] if available_actions is not None else None, 
                                                                       active_masks[:, i, :] if self._use_policy_active_masks and active_masks is not None else None)
            # append action_log_probs and dist_entropy to respective lists
            action_log_probs_list.append(action_log_probs)
            dist_entropy_list.append(dist_entropy)

        # [shape: (mini_batch_size * data_chunk_length * num_agents, action_space_dim)] and [shape: () == scalar]
        return torch.stack(action_log_probs_list, dim=1).reshape(mini_batch_size * self.data_chunk_length * self.num_agents, -1), torch.stack(dist_entropy_list, dim=0).mean()


class GCMNetGINCritic(nn.Module):
    """
    GCMNetGIN Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO)
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """ 
        class constructor for attributes for the critic model 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()

        self.data_chunk_length = args.data_chunk_length
        self._use_orthogonal = args.use_orthogonal
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        self.num_agents = args.num_agents
        self.n_rollout_threads = args.n_rollout_threads
        self.n_gnn_layers = args.n_gnn_layers
        self.somu_n_layers = args.somu_n_layers
        self.scmu_n_layers = args.scmu_n_layers
        self.somu_lstm_hidden_size = args.somu_lstm_hidden_size
        self.scmu_lstm_hidden_size = args.scmu_lstm_hidden_size
        self.somu_multi_att_n_heads = args.somu_multi_att_n_heads
        self.scmu_multi_att_n_heads = args.scmu_multi_att_n_heads
        self.fc_output_dims = args.fc_output_dims
        self.n_fc_layers = args.n_fc_layers
        self.knn = args.knn
        self.k = args.k
        self.rni = args.rni
        self.rni_ratio = args.rni_ratio
        self.n_gin_fc_layers = args.n_gin_fc_layers

        cent_obs_space = get_shape_from_obs_space(cent_obs_space)
        if isinstance(cent_obs_space, list):
            self.obs_dims = cent_obs_space[0]
        else:
            self.obs_dims = cent_obs_space

        self.rni_dims = round(self.obs_dims * self.rni_ratio)

        # model architecture for mappo GCMNetDNAGATv2 critic

        # gnn layers
        self.gnn_layers = GINLayers(input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                                    block=GINBlock, 
                                    output_channels=[self.obs_dims + self.rni_dims if self.rni\
                                                    else self.obs_dims for i in range(self.n_gnn_layers)],
                                    n_gin_fc_layers=self.n_gin_fc_layers)

        # list of lstms for self observation memory unit (somu) for each agent
        # somu_lstm_input_size is the dimension of the observations
        self.somu_lstm_list = nn.ModuleList([nn.LSTM(input_size=self.obs_dims, hidden_size=self.somu_lstm_hidden_size, 
                                                     num_layers=self.somu_n_layers, batch_first=True).to(device) 
                                             for _ in range(self.num_agents)])

        # list of lstms for self communication memory unit (scmu) for each agent
        # somu_lstm_input_size are all layers of gnn
        self.scmu_lstm_list = nn.ModuleList([nn.LSTM(input_size=(self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims)\
                                                                if self.rni else (self.n_gnn_layers + 1) * self.obs_dims, 
                                                     hidden_size=self.scmu_lstm_hidden_size, 
                                                     num_layers=self.scmu_n_layers, batch_first=True).to(device) 
                                             for _ in range(self.num_agents)])

        # multi-head self attention layer for somu and scmu to selectively choose between the lstms outputs
        self.somu_multi_att_layer_list = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.somu_lstm_hidden_size, 
                                                                              num_heads=self.somu_multi_att_n_heads, 
                                                                              dropout=0, batch_first=True, device=device) 
                                                        for _ in range(self.num_agents)])
        self.scmu_multi_att_layer_list = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.scmu_lstm_hidden_size, 
                                                                              num_heads=self.scmu_multi_att_n_heads, 
                                                                              dropout=0, batch_first=True, device=device) 
                                                        for _ in range(self.num_agents)])

        # shared hidden fc layers for to generate actions for each agent
        # input channels are all layers of gnn (including initial observations) 
        # + concatenated outputs of somu_multi_att_layer and scmu_multi_att_layer
        # fc_output_dims is the list of sizes of output channels fc_block
        self.fc_layers = NNLayers(input_channels=(self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) + \
                                                 self.somu_n_layers * self.somu_lstm_hidden_size + \
                                                 self.scmu_n_layers * self.scmu_lstm_hidden_size if self.rni
                                                 else (self.n_gnn_layers + 1) * self.obs_dims + \
                                                 self.somu_n_layers * self.somu_lstm_hidden_size + \
                                                 self.scmu_n_layers * self.scmu_lstm_hidden_size, 
                                  block=MLPBlock, output_channels=[self.fc_output_dims for i in range(self.n_fc_layers)], 
                                  activation_func='relu', dropout_p=0, 
                                  weight_initialisation="orthogonal" if self._use_orthogonal else "default").to(device)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # final layer for value function using popart / mlp
        if self._use_popart:
            self.v_out = init_(PopArt(self.fc_output_dims, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.fc_output_dims, 1))
        
        self.to(device)

    def forward(self, cent_obs, somu_hidden_states_critic, somu_cell_states_critic, scmu_hidden_states_critic, 
                scmu_cell_states_critic, masks):
        """
        Compute value function
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param somu_hidden_states_critic: (np.ndarray / torch.Tensor) hidden states for somu network.
        :param somu_cell_states_critic: (np.ndarray / torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_critic: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_critic: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return somu_hidden_states_critic: (torch.Tensor) hidden states for somu network.
        :return somu_cell_states_critic: (torch.Tensor) cell states for somu network.
        :return scmu_hidden_states_critic: (torch.Tensor) hidden states for scmu network.
        :return scmu_cell_states_critic: (torch.Tensor) hidden states for scmu network.
        """
        if self.knn:
            # [shape: (batch_size, num_agents, obs_dims)]
            obs = check(cent_obs) 
            batch_size = obs.shape[0]
            # [shape: (batch_size * num_agents, obs_dims)]
            obs_temp = obs.reshape(batch_size * self.num_agents, self.obs_dims)
            batch = torch.tensor([i // self.num_agents for i in range(batch_size * self.num_agents)])
            edge_index = knn_graph(x=obs_temp, k=self.k, batch=batch, loop=True)
            obs = obs.to(**self.tpdv)  
            if self.rni:
                # zero mean std 1 gaussian noise
                # [shape: (batch_size, num_agents, rni_dims)] 
                rni = torch.normal(0, 1, size=(batch_size, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # [shape: (batch_size, num_agents, obs_dims + rni_dims)]
                obs_rni = torch.cat((obs, rni), dim=-1)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni[i, :, :] if self.rni else obs[i, :, :], edge_index=edge_index) 
                                            for i in range(batch_size)]).to(self.device)
        else:
            # obtain edge index
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
            # [shape: (batch_size, num_agents, obs_dims)]  
            obs = check(cent_obs).to(**self.tpdv)  
            batch_size = obs.shape[0]
            if self.rni:
                # zero mean std 1 gaussian noise 
                # [shape: (batch_size, num_agents, rni_dims)] 
                rni = torch.normal(0, 1, size=(batch_size, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # [shape: (batch_size, num_agents, obs_dims + rni_dims)]
                obs_rni = torch.cat((obs, rni), dim=-1)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni[i, :, :] if self.rni else obs[i, :, :], edge_index=edge_index) 
                                            for i in range(batch_size)]).to(self.device)
        # shape: (batch_size, num_agents, somu_n_layers, somu_lstm_hidden_size)
        somu_hidden_states_critic = check(somu_hidden_states_critic).to(**self.tpdv)
        somu_cell_states_critic = check(somu_cell_states_critic).to(**self.tpdv)
        # shape: (batch_size, num_agents, scmu_n_layers, scmu_lstm_hidden_size) 
        scmu_hidden_states_critic = check(scmu_hidden_states_critic).to(**self.tpdv)
        scmu_cell_states_critic = check(scmu_cell_states_critic).to(**self.tpdv) 
        # shape: (batch_size, num_agents, 1)
        masks = check(masks).to(**self.tpdv).reshape(batch_size, self.num_agents, -1) 
        # store somu and scmu hidden states and cell states and values
        somu_lstm_hidden_state_list = []
        somu_lstm_cell_state_list = []
        scmu_lstm_hidden_state_list = []
        scmu_lstm_cell_state_list = []
        values_list = []
       
        # obs_gnn.x [shape: (batch_size * num_agents, obs_dims / (obs_dims + rni_dims))] 
        # --> gnn_layers [shape: (batch_size, num_agents, (n_gnn_layers + 1) * obs_dims / (n_gnn_layers + 1) * (obs_dims + rni_dims))]
        gnn_output = self.gnn_layers(x=obs_gnn.x, edge_index=obs_gnn.edge_index)\
                         .reshape(batch_size, self.num_agents, (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) if self.rni else\
                                  (self.n_gnn_layers + 1) * self.obs_dims)
        # sum embeddings across nodes at each iteration that are concatenated together
        # gnn_output_sum [shape: (batch_size, (n_gnn_layers + 1) * obs_dims / (n_gnn_layers + 1) * (obs_dims + rni_dims))]
        gnn_output_sum = gnn_output.sum(dim=1)
       
        # iterate over agents 
        for i in range(self.num_agents):
            # obs[:, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, obs_dims)],
            # masks[:, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (somu_n_layers, batch_size, 1)],
            # (h_0 [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)]) -->
            # somu_output [shape: (batch_size, sequence_length=1, somu_lstm_hidden_size)], 
            # (h_n [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)], c_n [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)])
            somu_output, somu_hidden_cell_states_tup = self.somu_lstm_list[i](obs[:, i, :].unsqueeze(dim=1), 
                                                                              (somu_hidden_states_critic.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.somu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous(), 
                                                                               somu_cell_states_critic.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.somu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous()))
            # gnn_output[:, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, (n_gnn_layers + 1) * obs_dims)],
            # masks[:, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (scmu_n_layers, batch_size, 1)], 
            # (h_0 [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)], c_0 [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)]) -->
            # scmu_output [shape: (batch_size, sequence_length=1, scmu_lstm_hidden_state)], 
            # (h_n [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)], c_n [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)])
            scmu_output, scmu_hidden_cell_states_tup = self.scmu_lstm_list[i](gnn_output[:, i, :].unsqueeze(dim=1), 
                                                                              (scmu_hidden_states_critic.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.scmu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous(), 
                                                                               scmu_cell_states_critic.transpose(0, 2)[:, i, :, :].contiguous() 
                                                                               * masks[:, i, :].repeat(1, self.scmu_n_layers)\
                                                                                               .transpose(0, 1)\
                                                                                               .unsqueeze(-1)\
                                                                                               .contiguous()))
            # (h_n [shape: (batch_size, somu_n_layers / scmu_n_layers, scmu_lstm_hidden_state)], 
            # c_n [shape: (batch_size, somu_n_layers / scmu_n_layers, scmu_lstm_hidden_state)])
            somu_lstm_hidden_state_list.append(somu_hidden_cell_states_tup[0].transpose(0, 1))
            somu_lstm_cell_state_list.append(somu_hidden_cell_states_tup[1].transpose(0, 1))
            scmu_lstm_hidden_state_list.append(scmu_hidden_cell_states_tup[0].transpose(0, 1))
            scmu_lstm_cell_state_list.append(scmu_hidden_cell_states_tup[1].transpose(0, 1))
            # somu_output / scmu_output [shape: (batch_size, sequence_length=1, somu_lstm_hidden_size / scmu_lstm_hidden_state)],
            # somu_hidden_state / scmu_hidden_state [shape: (batch_size, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)] 
            # --> multihead attention [shape: (batch_size, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
            somu_output = self.somu_multi_att_layer_list[i](somu_lstm_hidden_state_list[i], somu_output, somu_output)[0]
            scmu_output = self.somu_multi_att_layer_list[i](scmu_lstm_hidden_state_list[i], scmu_output, scmu_output)[0]

            # concatenate obs and outputs from gnn, somu and scmu 
            # [shape: (batch_size, (n_gnn_layers + 1) * obs_dims + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size) / 
            # (batch_size, (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            output = torch.cat((gnn_output_sum, somu_output.reshape(batch_size, -1), scmu_output.reshape(batch_size, -1)), dim=-1)
            # output [shape: (batch_size, (n_gnn_layers + 1) * self.obs_dims + somu_lstm_hidden_size + scmu_lstm_hidden_size) / 
            # (batch_size, (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            # -->   
            # fc_layers [shape: (batch_size, fc_output_dims)]
            output = self.fc_layers(output)
            # output --> v_out [shape: (batch_size, 1)]
            values = self.v_out(output)
            values_list.append(values)
       
        # [shape: (batch_size, num_agents, 1)]
        # [shape: (batch_size, num_agents, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
        return torch.stack(values_list, dim=1), torch.stack(somu_lstm_hidden_state_list, dim=1), torch.stack(somu_lstm_cell_state_list, dim=1), \
               torch.stack(scmu_lstm_hidden_state_list, dim=1), torch.stack(scmu_lstm_cell_state_list, dim=1)

    def evaluate_actions(self, cent_obs, somu_hidden_states_critic, somu_cell_states_critic, scmu_hidden_states_critic, 
                         scmu_cell_states_critic, masks):
        """
        Compute value function
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param somu_hidden_states_critic: (np.ndarray / torch.Tensor) hidden states for somu network.
        :param somu_cell_states_critic: (np.ndarray / torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_critic: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_critic: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        """
        if self.knn:
            # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims]   
            obs = check(cent_obs)
            mini_batch_size = obs.shape[0]
            # [shape: (mini_batch_size * data_chunk_length * num_agents, obs_dims)] 
            obs_temp = obs.reshape(mini_batch_size * self.data_chunk_length * self.num_agents, self.obs_dims)
            batch = torch.tensor([i // self.num_agents for i in range(mini_batch_size * self.data_chunk_length * self.num_agents)])
            edge_index = knn_graph(x=obs_temp, k=self.k, batch=batch, loop=True)
            obs = obs.to(**self.tpdv)
            # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims)] 
            obs_batch = obs.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims)
            if self.rni:
                # zero mean std 1 gaussian noise 
                # [shape: (mini_batch_size, data_chunk_length, num_agents, rni_dims)] 
                rni = torch.normal(0, 1, size=(mini_batch_size, self.data_chunk_length, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims + rni_dims)]
                obs_rni = torch.cat((obs, rni), dim=-1)
                # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims + rni_dims)] 
                obs_rni_batch = obs_rni.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims + self.rni_dims)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni_batch[i, :, :] if self.rni else obs_batch[i, :, :], edge_index=edge_index) 
                                            for i in range(mini_batch_size * self.data_chunk_length)]).to(self.device)
        else:
            # obtain edge index
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
            # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims)]   
            obs = check(cent_obs).to(**self.tpdv)
            mini_batch_size = obs.shape[0]
            # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims)] 
            obs_batch = obs.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims)
            if self.rni:
                # zero mean std 1 gaussian noise 
                # shape: (mini_batch_size, data_chunk_length, num_agents, rni_dims) 
                rni = torch.normal(0, 1, size=(mini_batch_size, self.data_chunk_length, self.num_agents, self.rni_dims)).to(**self.tpdv)  
                # shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims + rni_dims)
                obs_rni = torch.cat((obs, rni), dim=-1)
                # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims + rni_dims)] 
                obs_rni_batch = obs_rni.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims + self.rni_dims)
            obs_gnn = Batch.from_data_list([Data(x=obs_rni_batch[i, :, :] if self.rni else obs_batch[i, :, :], edge_index=edge_index) 
                                            for i in range(mini_batch_size * self.data_chunk_length)]).to(self.device)
        # [shape: (mini_batch_size, num_agents, somu_n_layers, somu_lstm_hidden_size)]
        somu_hidden_states_critic = check(somu_hidden_states_critic).to(**self.tpdv) 
        somu_cell_states_critic = check(somu_cell_states_critic).to(**self.tpdv)
        # [shape: (mini_batch_size, num_agents, scmu_n_layers, scmu_lstm_hidden_size)]
        scmu_hidden_states_critic = check(scmu_hidden_states_critic).to(**self.tpdv) 
        scmu_cell_states_critic = check(scmu_cell_states_critic).to(**self.tpdv)
        # [shape: (mini_batch_size, data_chunk_length, num_agents, 1)]  
        masks = check(masks).to(**self.tpdv)
        # store values
        values_list = []

        # obs_gnn.x [shape: (mini_batch_size * data_chunk_length * num_agents, obs_dims)] -->
        # gnn_layers [shape: (mini_batch_size, data_chunk_length, num_agents, (n_gnn_layers + 1) * obs_dims \ 
        # (n_gnn_layers + 1) * (obs_dims + rni_dims))]
        gnn_output = self.gnn_layers(x=obs_gnn.x, edge_index=obs_gnn.edge_index)\
                         .reshape(mini_batch_size, self.data_chunk_length, self.num_agents, 
                                  (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) if self.rni else\
                                  (self.n_gnn_layers + 1) * self.obs_dims)
        # sum embeddings across nodes at each iteration that are concatenated together
        # gnn_output_sum [shape: (mini_batch_size * data_chunk_length, (n_gnn_layers + 1) * obs_dims / (n_gnn_layers + 1) * (obs_dims + rni_dims))]
        gnn_output_sum = gnn_output.sum(dim=2)\
                                   .reshape(mini_batch_size * self.data_chunk_length, (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims)\
                                            if self.rni else (self.n_gnn_layers + 1) * self.obs_dims)

        # iterate over agents 
        for i in range(self.num_agents):
            # list to store somu and scmu outputs, hidden states and cell states over sequence of inputs of len, data_chunk_length
            somu_seq_output_list = []
            somu_seq_lstm_hidden_state_list = []
            somu_seq_lstm_cell_state_list = []
            scmu_seq_output_list = []
            scmu_seq_lstm_hidden_state_list = []
            scmu_seq_lstm_cell_state_list = []
            # initialise hidden and cell states for somu and scmu at start of sequence
            # (h_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
            somu_seq_lstm_hidden_state_list.append(somu_hidden_states_critic[:, i, :, :].transpose(0, 1).contiguous())
            somu_seq_lstm_cell_state_list.append(somu_cell_states_critic[:, i, :, :].transpose(0, 1).contiguous())
            scmu_seq_lstm_hidden_state_list.append(scmu_hidden_states_critic[:, i, :, :].transpose(0, 1).contiguous())
            scmu_seq_lstm_cell_state_list.append(scmu_cell_states_critic[:, i, :, :].transpose(0, 1).contiguous())

            # iterate over data_chunk_length
            for j in range(self.data_chunk_length):
                # obs[:, j, i, :].unsqueeze(dim=1) [shape: (mini_batch_size, sequence_length=1, obs_dims)],
                # masks[:, j, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (somu_n_layers, batch_size, 1)],
                # (h_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], c_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
                # -->
                # somu_output [shape: (mini_batch_size, sequence_length=1, somu_lstm_hidden_size)], 
                # (h_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], c_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
                somu_output, somu_hidden_cell_states_tup = self.somu_lstm_list[i](obs[:, j, i, :].unsqueeze(dim=1),
                                                                                  (somu_seq_lstm_hidden_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.somu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous(),
                                                                                   somu_seq_lstm_cell_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.somu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous()))
                # gnn_output[:, j, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, (n_gnn_layers + 1) * obs_dims)],
                # masks[:, j, i, :].repeat(1, self.scmu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() [shape: (scmu_n_layers, batch_size, 1)],
                # (h_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], c_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)])
                # -->
                # scmu_output [shape: (mini_batch_size, sequence_length=1, scmu_lstm_hidden_size)], 
                # (h_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], c_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)])
                scmu_output, scmu_hidden_cell_states_tup = self.scmu_lstm_list[i](gnn_output[:, j, i, :].unsqueeze(dim=1),
                                                                                  (scmu_seq_lstm_hidden_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.scmu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous(),
                                                                                   scmu_seq_lstm_cell_state_list[-1] 
                                                                                   * masks[:, j, i, :].repeat(1, self.scmu_n_layers)\
                                                                                                      .transpose(0, 1)\
                                                                                                      .unsqueeze(-1)\
                                                                                                      .contiguous()))
                somu_seq_output_list.append(somu_output)
                scmu_seq_output_list.append(scmu_output)
                somu_seq_lstm_hidden_state_list.append(somu_hidden_cell_states_tup[0])
                somu_seq_lstm_cell_state_list.append(somu_hidden_cell_states_tup[1])
                scmu_seq_lstm_hidden_state_list.append(scmu_hidden_cell_states_tup[0])
                scmu_seq_lstm_cell_state_list.append(scmu_hidden_cell_states_tup[1])

            # [shape: (mini_batch_size * data_chunk_length, 1, somu_lstm_hidden_size / scmu_lstm_hidden_state)]
            somu_output = torch.stack(somu_seq_output_list, dim=1).reshape(mini_batch_size * self.data_chunk_length, 1, self.somu_lstm_hidden_size)
            scmu_output = torch.stack(scmu_seq_output_list, dim=1).reshape(mini_batch_size * self.data_chunk_length, 1, self.scmu_lstm_hidden_size)
            # [shape: (data_chunk_length, somu_n_layers / scmu_n_layers, mini_batch_size, somu_lstm_hidden_state / scmu_lstm_hidden_state)] --> 
            # [shape: (mini_batch_size * data_chunk_length, somu_n_layers / scmu_n_layers, somu_lstm_hidden_state / scmu_lstm_hidden_state)]
            somu_lstm_hidden_state = torch.stack(somu_seq_lstm_hidden_state_list[1:], dim=0).permute(2, 0, 1, 3).reshape(mini_batch_size * self.data_chunk_length, 
                                                                                                                         self.somu_n_layers, 
                                                                                                                         self.somu_lstm_hidden_size)
            scmu_lstm_hidden_state = torch.stack(scmu_seq_lstm_hidden_state_list[1:], dim=0).permute(2, 0, 1, 3).reshape(mini_batch_size * self.data_chunk_length, 
                                                                                                                         self.scmu_n_layers, 
                                                                                                                         self.scmu_lstm_hidden_size)
           
            # multihead attention [shape: (mini_batch_size * data_chunk_length, somu_n_layers / scmu_n_layers, somu_lstm_hidden_size / scmu_lstm_hidden_size)]
            somu_output = self.somu_multi_att_layer_list[i](somu_lstm_hidden_state, somu_output, somu_output)[0]
            scmu_output = self.somu_multi_att_layer_list[i](scmu_lstm_hidden_state, scmu_output, scmu_output)[0]

            # concatenate obs and outputs from gnn, somu and scmu 
            # [shape: (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * obs_dims + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size) / 
            # (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            output = torch.cat((gnn_output_sum, 
                                somu_output.reshape(mini_batch_size * self.data_chunk_length, self.somu_n_layers * self.somu_lstm_hidden_size), 
                                scmu_output.reshape(mini_batch_size * self.data_chunk_length, self.scmu_n_layers * self.scmu_lstm_hidden_size)), dim=-1)
            # output [shape: (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * obs_dims + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size) /  
            # (mini_batch_size * data_chunk_length, 
            # (n_gnn_layers + 1) * (obs_dims + rni_dims) + somu_n_layers * somu_lstm_hidden_size + scmu_n_layers * scmu_lstm_hidden_size)]
            # --> 
            # fc_layers [shape: (mini_batch_size * data_chunk_length, fc_output_dims)]
            output = self.fc_layers(output)
            # output --> v_out [shape: (mini_batch_size * data_chunk_length, 1)]
            values = self.v_out(output)
            values_list.append(values)

        # [shape: (mini_batch_size * data_chunk_length * num_agents, 1)]
        return torch.stack(values_list, dim=1).reshape(-1, 1)