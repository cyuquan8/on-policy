import torch
import torch.nn as nn

from onpolicy.algorithms.utils.nn import (
    DNAGATv2Block,
    GAINBlock,
    GATBlock,
    GATv2Block,
    GCNBlock,
    GINBlock, 
    GNNConcatAllLayers,
    GNNDNALayers, 
    MLPBlock, 
    NNLayers
)
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.single_act import SingleACTLayer
from onpolicy.algorithms.utils.util import init, check, complete_graph_edge_index
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph

class GCMNetActor(nn.Module):
    """
    GCMNet Actor network class for MAPPO. Outputs actions given observations.
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

        self.gnn_architecture = args.gcmnet_gnn_architecture
        self.gnn_output_dims = args.gcmnet_gnn_output_dims
        self.gnn_att_heads = args.gcmnet_gnn_att_heads
        self.gnn_dna_gatv2_multi_att_heads = args.gcmnet_gnn_dna_gatv2_multi_att_heads
        self.gnn_att_concat = args.gcmnet_gnn_att_concat
        self.gnn_cpa_model = args.gcmnet_gnn_cpa_model
        self.n_gnn_layers = args.gcmnet_n_gnn_layers
        self.n_gnn_fc_layers = args.gcmnet_n_gnn_fc_layers
        self.gnn_train_eps = args.gcmnet_gnn_train_eps
        self.gnn_norm = args.gcmnet_gnn_norm

        self.somu_actor = args.gcmnet_somu_actor
        self.scmu_actor = args.gcmnet_scmu_actor
        self.somu_critic = args.gcmnet_somu_critic
        self.scmu_critic = args.gcmnet_scmu_critic
        self.somu_lstm_actor = args.gcmnet_somu_lstm_actor
        self.scmu_lstm_actor = args.gcmnet_scmu_lstm_actor
        self.somu_att_actor = args.gcmnet_somu_att_actor
        self.scmu_att_actor = args.gcmnet_scmu_att_actor
        self.somu_n_layers = args.gcmnet_somu_n_layers
        self.scmu_n_layers = args.gcmnet_scmu_n_layers
        self.somu_lstm_hidden_size = args.gcmnet_somu_lstm_hidden_size
        self.scmu_lstm_hidden_size = args.gcmnet_scmu_lstm_hidden_size
        self.somu_multi_att_n_heads = args.gcmnet_somu_multi_att_n_heads
        self.scmu_multi_att_n_heads = args.gcmnet_scmu_multi_att_n_heads

        assert not (self.somu_actor == False and self.somu_lstm_actor == True)
        assert not (self.scmu_actor == False and self.scmu_lstm_actor == True)
        assert not (self.somu_actor == False and self.somu_att_actor == True)
        assert not (self.scmu_actor == False and self.scmu_att_actor == True)
        assert not (self.somu_actor == True and self.somu_lstm_actor == False and self.somu_att_actor == False)
        assert not (self.scmu_actor == True and self.scmu_lstm_actor == False and self.scmu_att_actor == False)

        self.fc_output_dims = args.gcmnet_fc_output_dims
        self.n_fc_layers = args.gcmnet_n_fc_layers

        self.knn = args.gcmnet_knn
        self.k = args.gcmnet_k

        self.rni = args.gcmnet_rni
        self.rni_ratio = args.gcmnet_rni_ratio

        self.dynamics = args.gcmnet_dynamics
        self.dynamics_fc_output_dims = args.gcmnet_dynamics_fc_output_dims
        self.dynamics_n_fc_layers = args.gcmnet_dynamics_n_fc_layers

        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            raise NotImplementedError("CNN-based observations not implemented for GCMNet")
        if isinstance(obs_shape, (list, tuple)):
            self.obs_dims = obs_shape[0]
        else:
            self.obs_dims = obs_shape
        self.act_dims = get_shape_from_act_space(action_space)

        self.rni_dims = round(self.obs_dims * self.rni_ratio)

        # model architecture for mappo GCMNetActor

        # gnn layers
        if self.gnn_architecture == 'dna_gatv2':
            self.gnn_layers = GNNDNALayers(
                input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                block=DNAGATv2Block, 
                output_channels=[self.obs_dims + self.rni_dims if self.rni else self.obs_dims \
                                 for _ in range(self.n_gnn_layers)],
                att_heads=self.gnn_att_heads,
                mul_att_heads=self.gnn_dna_gatv2_multi_att_heads,
                gnn_cpa_model=self.gnn_cpa_model,
                norm_type=self.gnn_norm 
            )
            # calculate relevant input dimensions
            self.scmu_input_dims = (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) \
                                   if self.rni else (self.n_gnn_layers + 1) * self.obs_dims
        elif self.gnn_architecture == 'gcn':
            self.gnn_layers = GNNConcatAllLayers(
                input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                block=GCNBlock, 
                output_channels=[self.gnn_output_dims for _ in range(self.n_gnn_layers)], 
                n_gnn_fc_layers=self.n_gnn_fc_layers,
                norm_type=self.gnn_norm
            )
            # calculate relevant input dimensions
            self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims + self.obs_dims + self.rni_dims \
                                   if self.rni else self.n_gnn_layers * self.gnn_output_dims + self.obs_dims                         
        elif self.gnn_architecture == 'gat':
            self.gnn_layers = GNNConcatAllLayers(
                input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                block=GATBlock, 
                output_channels=[self.gnn_output_dims for _ in range(self.n_gnn_layers)], 
                heads=self.gnn_att_heads,
                concat=self.gnn_att_concat,
                gnn_cpa_model=self.gnn_cpa_model,
                norm_type=self.gnn_norm
            )
            # calculate relevant input dimensions
            if self.rni:
                self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims * self.gnn_att_heads + self.obs_dims + \
                                       self.rni_dims if self.gnn_att_concat else \
                                       self.n_gnn_layers * self.gnn_output_dims + self.obs_dims + self.rni_dims
            else: 
                self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims * self.gnn_att_heads + self.obs_dims \
                                       if self.gnn_att_concat else \
                                       self.n_gnn_layers * self.gnn_output_dims + self.obs_dims
        elif self.gnn_architecture == 'gatv2':
            self.gnn_layers = GNNConcatAllLayers(
                input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                block=GATv2Block, 
                output_channels=[self.gnn_output_dims for _ in range(self.n_gnn_layers)], 
                heads=self.gnn_att_heads,
                concat=self.gnn_att_concat,
                gnn_cpa_model=self.gnn_cpa_model,
                norm_type=self.gnn_norm
            )
            # calculate relevant input dimensions
            if self.rni:
                self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims * self.gnn_att_heads + self.obs_dims + \
                                       self.rni_dims if self.gnn_att_concat else \
                                       self.n_gnn_layers * self.gnn_output_dims + self.obs_dims + self.rni_dims
            else: 
                self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims * self.gnn_att_heads + self.obs_dims \
                                       if self.gnn_att_concat else \
                                       self.n_gnn_layers * self.gnn_output_dims + self.obs_dims
        elif self.gnn_architecture == 'gin':
            self.gnn_layers = GNNConcatAllLayers(
                input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                block=GINBlock, 
                output_channels=[self.gnn_output_dims for _ in range(self.n_gnn_layers)], 
                n_gnn_fc_layers=self.n_gnn_fc_layers,
                train_eps=self.gnn_train_eps,
                norm_type=self.gnn_norm
            )
            # calculate relevant input dimensions
            self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims + self.obs_dims + self.rni_dims \
                                   if self.rni else self.n_gnn_layers * self.gnn_output_dims + self.obs_dims
        elif self.gnn_architecture == 'gain':
            self.gnn_layers = GNNConcatAllLayers(
                input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                block=GAINBlock, 
                output_channels=[self.gnn_output_dims for _ in range(self.n_gnn_layers)],
                n_gnn_fc_layers=self.n_gnn_fc_layers,
                heads=self.gnn_att_heads,
                concat=self.gnn_att_concat,
                train_eps=self.gnn_train_eps,
                norm_type=self.gnn_norm
            )
            # calculate relevant input dimensions
            self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims + self.obs_dims + self.rni_dims \
                                   if self.rni else self.n_gnn_layers * self.gnn_output_dims + self.obs_dims

        if self.somu_actor:
            # list of lstms for self observation memory unit (somu) for each agent
            # somu_lstm_input_size is the dimension of the observations
            self.somu_lstm_list = nn.ModuleList([
                nn.LSTM(
                    input_size=self.obs_dims, 
                    hidden_size=self.somu_lstm_hidden_size, 
                    num_layers=self.somu_n_layers, 
                    batch_first=True,
                    device=device
                ) for _ in range(self.num_agents)
            ])
            if self.somu_att_actor:
                # multi-head self attention layer for somu to selectively choose between the lstms outputs
                self.somu_multi_att_layer_list = nn.ModuleList([
                    nn.MultiheadAttention(
                        embed_dim=self.somu_lstm_hidden_size, 
                        num_heads=self.somu_multi_att_n_heads, 
                        dropout=0, 
                        batch_first=True, 
                        device=device
                    ) for _ in range(self.num_agents)
                ])

        if self.scmu_actor:
            # list of lstms for self communication memory unit (scmu) for each agent
            # somu_lstm_input_size are all layers of gnn
            self.scmu_lstm_list = nn.ModuleList([
                nn.LSTM(
                    input_size=self.scmu_input_dims, 
                    hidden_size=self.scmu_lstm_hidden_size, 
                    num_layers=self.scmu_n_layers, 
                    batch_first=True,
                    device=device
                ) for _ in range(self.num_agents)
            ])
            if self.scmu_att_actor:
                # multi-head self attention layer for scmu to selectively choose between the lstms outputs
                self.scmu_multi_att_layer_list = nn.ModuleList([
                    nn.MultiheadAttention(
                        embed_dim=self.scmu_lstm_hidden_size, 
                        num_heads=self.scmu_multi_att_n_heads, 
                        dropout=0, 
                        batch_first=True, 
                        device=device
                    ) for _ in range(self.num_agents)
                ])

        # calculate input dimensions for fc layers
        if self.somu_actor == True and self.scmu_actor == True:
            if self.somu_lstm_actor == True and self.somu_att_actor == True and self.scmu_lstm_actor == True \
                and self.scmu_att_actor == True:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_lstm, somu_multi_att_layer, scmu_lstm and scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers + 1) * self.somu_lstm_hidden_size \
                                     + (2 * self.scmu_n_layers + 1) * self.scmu_lstm_hidden_size
            elif self.somu_lstm_actor == False and self.somu_att_actor == True and self.scmu_lstm_actor == True \
                and self.scmu_att_actor == True:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_multi_att_layer, scmu_lstm and scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers) * self.somu_lstm_hidden_size \
                                     + (2 * self.scmu_n_layers + 1) * self.scmu_lstm_hidden_size
            elif self.somu_lstm_actor == True and self.somu_att_actor == False and self.scmu_lstm_actor == True \
                and self.scmu_att_actor == True:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_lstm, scmu_lstm and scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + self.somu_lstm_hidden_size \
                                     + (2 * self.scmu_n_layers + 1) * self.scmu_lstm_hidden_size
            elif self.somu_lstm_actor == True and self.somu_att_actor == True and self.scmu_lstm_actor == False \
                and self.scmu_att_actor == True:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_lstm, somu_multi_att_layer and scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers + 1) * self.somu_lstm_hidden_size \
                                     + (2 * self.scmu_n_layers) * self.scmu_lstm_hidden_size
            elif self.somu_lstm_actor == True and self.somu_att_actor == True and self.scmu_lstm_actor == True \
                and self.scmu_att_actor == False:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_lstm, somu_multi_att_layer and scmu_lstm
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers + 1) * self.somu_lstm_hidden_size \
                                     + self.scmu_lstm_hidden_size
            elif self.somu_lstm_actor == False and self.somu_att_actor == True and self.scmu_lstm_actor == False \
                and self.scmu_att_actor == True:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_multi_att_layer and scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers) * self.somu_lstm_hidden_size \
                                     + (2 * self.scmu_n_layers) * self.scmu_lstm_hidden_size
            elif self.somu_lstm_actor == True and self.somu_att_actor == False and self.scmu_lstm_actor == True \
                and self.scmu_att_actor == False:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_lstm and scmu_lstm
                self.fc_input_dims = self.scmu_input_dims + self.somu_lstm_hidden_size + self.scmu_lstm_hidden_size
        elif self.somu_actor == True and self.scmu_actor == False:
            if self.somu_lstm_actor == True and self.somu_att_actor == True:
                # all concatenated layers of gnn (including initial observations) + 
                # concatenated outputs of somu_lstm and somu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers + 1) * self.somu_lstm_hidden_size
            elif self.somu_lstm_actor == False and self.somu_att_actor == True:
                # all concatenated layers of gnn (including initial observations) + somu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers) * self.somu_lstm_hidden_size
            elif self.somu_lstm_actor == True and self.somu_att_actor == False:
                # all concatenated layers of gnn (including initial observations) + somu_lstm
                self.fc_input_dims = self.scmu_input_dims + self.somu_lstm_hidden_size
        elif self.somu_actor == False and self.scmu_actor == True:
            if self.scmu_lstm_actor == True and self.scmu_att_actor == True:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of scmu_lstm and scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.scmu_n_layers + 1) * self.scmu_lstm_hidden_size
            elif self.scmu_lstm_actor == False and self.scmu_att_actor == True:
                # all concatenated layers of gnn (including initial observations) + scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.scmu_n_layers) * self.scmu_lstm_hidden_size
            elif self.scmu_lstm_actor == True and self.scmu_att_actor == False:
                # all concatenated layers of gnn (including initial observations) + scmu_lstm
                self.fc_input_dims = self.scmu_input_dims + self.scmu_lstm_hidden_size
        else:
            # all concatenated layers of gnn (including initial observations)
            self.fc_input_dims = self.scmu_input_dims

        # shared hidden fc layers for to generate actions for each agent
        # fc_output_dims is the list of sizes of output channels fc_block
        self.fc_layers = NNLayers(
            input_channels=self.fc_input_dims, 
            block=MLPBlock, 
            output_channels=[self.fc_output_dims for i in range(self.n_fc_layers)],
            norm_type='none', 
            activation_func='relu', 
            dropout_p=0, 
            weight_initialisation="orthogonal" if self._use_orthogonal else "default"
        )

        # dynamics models
        if self.dynamics:
            self.dynamics_list = nn.ModuleList([
                NNLayers(
                    input_channels=self.fc_input_dims + self.act_dims, 
                    block=MLPBlock, 
                    output_channels=[self.obs_dims if i + 1 == self.dynamics_n_fc_layers else \
                                     self.dynamics_fc_output_dims for i in range(self.dynamics_n_fc_layers)],
                    norm_type='none', 
                    activation_func='relu', 
                    dropout_p=0, 
                    weight_initialisation="orthogonal" if self._use_orthogonal else "default"
                ) for _ in range(self.num_agents)
            ])

        # shared final action layer for each agent
        self.act = SingleACTLayer(action_space, self.fc_output_dims, self._use_orthogonal, self._gain)
        
        self.to(device)
        
    def forward(
            self, 
            obs, 
            masks, 
            available_actions=None, 
            somu_hidden_states_actor=None, 
            somu_cell_states_actor=None, 
            scmu_hidden_states_actor=None, 
            scmu_cell_states_actor=None, 
            deterministic=False
        ):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be initialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param somu_hidden_states_actor: (np.ndarray / torch.Tensor) hidden states for somu network.
        :param somu_cell_states_actor: (np.ndarray / torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_actor: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_actor: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return somu_hidden_states_actor: (torch.Tensor) hidden states for somu network.
        :return somu_cell_states_actor: (torch.Tensor) cell states for somu network.
        :return scmu_hidden_states_actor: (torch.Tensor) hidden states for scmu network.
        :return scmu_cell_states_actor: (torch.Tensor) hidden states for scmu network.
        :return obs_pred: (torch.Tensor) observation predictions from dynamics models if used else None.
        """
        # [shape: (batch_size, num_agents, obs_dims)]
        obs = check(obs).to(**self.tpdv) 
        batch_size = obs.shape[0]
        # obtain batch (needed for graphnorm if being used), [shape: (batch_size * num_agents)]
        if self.gnn_norm == 'graphnorm':
            batch = torch.arange(batch_size).repeat_interleave(self.num_agents).to(self.device)
        # complete graph edge index (including self-loops), [shape: (2, num_agents * num_agents)] 
        if not self.knn:
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        # random node initialisation
        if self.rni:
            # zero mean std 1 gaussian noise, [shape: (batch_size, num_agents, rni_dims)] 
            rni = torch.normal(0, 1, size=(batch_size, self.num_agents, self.rni_dims)).to(**self.tpdv)  
            # [shape: (batch_size, num_agents, obs_dims + rni_dims)]
            obs_rni = torch.cat((obs, rni), dim=-1)
        # gnn batched observations 
        obs_gnn = Batch.from_data_list([
            Data(
                x=obs_rni[i, :, :] if self.rni else obs[i, :, :], 
                edge_index=knn_graph(x=obs[i, :, :], k=self.k, loop=False).to(self.device) if self.knn else edge_index
            ) for i in range(batch_size)
        ]).to(self.device)
        # [shape: (batch_size, num_agents, 1)]
        masks = check(masks).to(**self.tpdv).reshape(batch_size, self.num_agents, -1)
        if available_actions is not None:
            # [shape: (batch_size, num_agents, action_space_dim)]
            available_actions = check(available_actions).to(**self.tpdv)
        if self.somu_actor:
            # [shape: (batch_size, num_agents, somu_n_layers, somu_lstm_hidden_size)]
            somu_hidden_states_actor = check(somu_hidden_states_actor).to(**self.tpdv)
            somu_cell_states_actor = check(somu_cell_states_actor).to(**self.tpdv)
            # store somu hidden states and cell states
            somu_lstm_hidden_states_list = []
            somu_lstm_cell_states_list = []
        else:
            assert somu_hidden_states_actor == None and somu_cell_states_actor == None
        if self.scmu_actor:
            # [shape: (batch_size, num_agents, scmu_n_layers, scmu_lstm_hidden_size)] 
            scmu_hidden_states_actor = check(scmu_hidden_states_actor).to(**self.tpdv)
            scmu_cell_states_actor = check(scmu_cell_states_actor).to(**self.tpdv)
            # store scmu hidden states and cell states
            scmu_lstm_hidden_states_list = []
            scmu_lstm_cell_states_list = []
        else:
            assert scmu_hidden_states_actor == None and scmu_cell_states_actor == None 
        # store actions and action_log_probs
        actions_list = []
        action_log_probs_list = []
        # store observation predictions if dynamics models are used
        if self.dynamics:
            obs_pred_list = []
        
        # obs_gnn.x [shape: (batch_size * num_agents, obs_dims / (obs_dims + rni_dims))] 
        # --> gnn_layers [shape: (batch_size * num_agents, scmu_input_dims)] 
        if self.gnn_architecture == 'dna_gatv2' or self.gnn_architecture == 'gatv2' or self.gnn_architecture == 'gain':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None, 
                    return_attention_weights=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index,
                    edge_attr=None, 
                    return_attention_weights=None
                )
        elif self.gnn_architecture == 'gat':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None,
                    size=None, 
                    return_attention_weights=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None,
                    size=None, 
                    return_attention_weights=None
                )
        elif self.gnn_architecture == 'gcn':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_weight=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_weight=None
                )
        elif self.gnn_architecture == 'gin':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    size=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    size=None
                )
        # [shape: (batch_size, num_agents, scmu_input_dims)]
        gnn_output = gnn_output.reshape(batch_size, self.num_agents, self.scmu_input_dims)
       
        # iterate over agents 
        for i in range(self.num_agents):
            if self.somu_actor:
                # obs[:, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, obs_dims)],
                # masks[:, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() 
                # [shape: (somu_n_layers, batch_size, 1)],
                # (h_0 [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)], 
                #  c_0 [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)]) -->
                # somu_lstm_output [shape: (batch_size, sequence_length=1, somu_lstm_hidden_size)], 
                # (h_n [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)], 
                #  c_n [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)])
                somu_lstm_output, (somu_hidden_states, somu_cell_states) = \
                    self.somu_lstm_list[i](obs[:, i, :].unsqueeze(dim=1), 
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
                # (h_n [shape: (batch_size, somu_n_layers, somu_lstm_hidden_state], 
                #  c_n [shape: (batch_size, somu_n_layers, somu_lstm_hidden_state)])
                somu_lstm_hidden_states_list.append(somu_hidden_states.transpose(0, 1))
                somu_lstm_cell_states_list.append(somu_cell_states.transpose(0, 1))
                # somu_lstm_output [shape: (batch_size, sequence_length=1, somu_lstm_hidden_size)] --> 
                # [shape: (batch_size, somu_lstm_hidden_size)]
                somu_lstm_output = somu_lstm_output.reshape(batch_size, self.somu_lstm_hidden_size)
                if self.somu_att_actor == True:
                    # concatenate hidden (short-term memory) and cell (long-term memory) states for somu
                    # (h_n [shape: (batch_size, somu_n_layers, somu_lstm_hidden_state], 
                    #  c_n [shape: (batch_size, somu_n_layers, somu_lstm_hidden_state)]) --> 
                    # somu_hidden_cell_states [shape: (batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)]
                    somu_hidden_cell_states = \
                        torch.cat((somu_lstm_hidden_states_list[i], somu_lstm_cell_states_list[i]), dim=1)
                    # self attention for memory from somu
                    # somu_hidden_cell_states [shape: (batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)] --> 
                    # somu_att_output [shape: (batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)]
                    somu_att_output = self.somu_multi_att_layer_list[i](somu_hidden_cell_states, 
                                                                        somu_hidden_cell_states, 
                                                                        somu_hidden_cell_states)[0]
                    # somu_att_output [shape: (batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)] -->
                    # [shape: (batch_size, 2 * somu_n_layers * somu_lstm_hidden_state)]
                    somu_att_output = somu_att_output.reshape(batch_size, 
                                                              2 * self.somu_n_layers * self.somu_lstm_hidden_size)
            if self.scmu_actor:
                # gnn_output[:, i, :].unsqueeze(dim=1) 
                # [shape: (batch_size, sequence_length=1, scmu_input_dims)],
                # masks[:, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() 
                # [shape: (scmu_n_layers, batch_size, 1)], 
                # (h_0 [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)], 
                #  c_0 [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)]) -->
                # scmu_lstm_output [shape: (batch_size, sequence_length=1, scmu_lstm_hidden_state)], 
                # (h_n [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)], 
                #  c_n [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)])
                scmu_lstm_output, (scmu_hidden_states, scmu_cell_states) = \
                    self.scmu_lstm_list[i](gnn_output[:, i, :].unsqueeze(dim=1), 
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
                # (h_n [shape: (batch_size, scmu_n_layers, scmu_lstm_hidden_state)], 
                #  c_n [shape: (batch_size, scmu_n_layers, scmu_lstm_hidden_state)])
                scmu_lstm_hidden_states_list.append(scmu_hidden_states.transpose(0, 1))
                scmu_lstm_cell_states_list.append(scmu_cell_states.transpose(0, 1))
                # scmu_lstm_output [shape: (batch_size, sequence_length=1, scmu_lstm_hidden_size)] --> 
                # [shape: (batch_size, scmu_lstm_hidden_size)]
                scmu_lstm_output = scmu_lstm_output.reshape(batch_size, self.scmu_lstm_hidden_size)
                if self.scmu_att_actor == True:
                    # concatenate hidden (short-term memory) and cell (long-term memory) states for scmu
                    # (h_n [shape: (batch_size, scmu_n_layers, scmu_lstm_hidden_state)], 
                    #  c_n [shape: (batch_size, scmu_n_layers, scmu_lstm_hidden_state)]) --> 
                    # scmu_hidden_cell_states [shape: (batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)]
                    scmu_hidden_cell_states = \
                        torch.cat((scmu_lstm_hidden_states_list[i], scmu_lstm_cell_states_list[i]), dim=1)
                    # self attention for memory from scmu
                    # scmu_hidden_cell_states [shape: (batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)] --> 
                    # scmu_att_output [shape: (batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)]
                    scmu_att_output = self.scmu_multi_att_layer_list[i](scmu_hidden_cell_states, 
                                                                        scmu_hidden_cell_states, 
                                                                        scmu_hidden_cell_states)[0]
                    # scmu_att_output [shape: (batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)] -->
                    # [shape: (batch_size, 2 * scmu_n_layers * scmu_lstm_hidden_state)]
                    scmu_att_output = scmu_att_output.reshape(batch_size, 
                                                              2 * self.scmu_n_layers * self.scmu_lstm_hidden_size)
            # concat_output [shape: (batch_size, fc_input_dims)]
            if self.somu_actor == True and self.scmu_actor == True:
                if self.somu_lstm_actor == True and self.somu_att_actor == True and self.scmu_lstm_actor == True \
                    and self.scmu_att_actor == True:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer, scmu_lstm and 
                    # scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output, 
                        somu_att_output, 
                        scmu_lstm_output,
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == False and self.somu_att_actor == True and self.scmu_lstm_actor == True \
                    and self.scmu_att_actor == True:
                    # concatenate outputs from gnn, somu_multi_att_layer, scmu_lstm and scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_att_output, 
                        scmu_lstm_output,
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == True and self.somu_att_actor == False and self.scmu_lstm_actor == True \
                    and self.scmu_att_actor == True:
                    # concatenate outputs from gnn, somu_lstm, scmu_lstm and scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output,  
                        scmu_lstm_output,
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == True and self.somu_att_actor == True and self.scmu_lstm_actor == False \
                    and self.scmu_att_actor == True:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer and scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output, 
                        somu_att_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == True and self.somu_att_actor == True and self.scmu_lstm_actor == True \
                    and self.scmu_att_actor == False:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer and scmu_lstm
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output, 
                        somu_att_output, 
                        scmu_lstm_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == False and self.somu_att_actor == True and self.scmu_lstm_actor == False \
                    and self.scmu_att_actor == True:
                    # concatenate outputs from gnn, somu_multi_att_layer and scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_att_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == True and self.somu_att_actor == False and self.scmu_lstm_actor == True \
                    and self.scmu_att_actor == False:
                    # concatenate outputs from gnn, somu_lstm and scmu_lstm 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output, 
                        scmu_lstm_output
                        ), 
                    dim=-1)
            elif self.somu_actor == True and self.scmu_actor == False:
                if self.somu_lstm_actor == True and self.somu_att_actor == True:
                    # concatenate outputs from gnn, somu_lstm and somu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output,
                        somu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == False and self.somu_att_actor == True:
                    # concatenate outputs from gnn and somu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == True and self.somu_att_actor == False:
                    # concatenate outputs from gnn and somu_lstm
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output
                        ), 
                    dim=-1)
            elif self.somu_actor == False and self.scmu_actor == True:
                if self.scmu_lstm_actor == True and self.scmu_att_actor == True:
                    # concatenate outputs from gnn, scmu_lstm and scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        scmu_lstm_output,
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.scmu_lstm_actor == False and self.scmu_att_actor == True:
                    # concatenate outputs from gnn and scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.scmu_lstm_actor == True and self.scmu_att_actor == False:
                    # concatenate outputs from gnn and scmu_lstm
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        scmu_lstm_output
                        ), 
                    dim=-1)
            else:
                # all concatenated layers of gnn (including initial observations)
                concat_output = gnn_output[:, i, :]
            # concat_output [shape: (batch_size, fc_input_dims)] --> fc_layers [shape: (batch_size, fc_output_dims)]
            fc_output = self.fc_layers(concat_output)
            # fc_layers --> act [shape: (batch_size, act_dims)]
            actions, action_log_probs = self.act(fc_output, available_actions[:, i, :] \
                                                 if available_actions is not None else None, deterministic)
            # append actions and action_log_probs to respective lists
            actions_list.append(actions)
            action_log_probs_list.append(action_log_probs)
            # get observation predictions if dynamic models are used
            if self.dynamics:
                # concatenate concat output with actions
                # dynamics_input [shape: (batch_size, fc_input_dims + act_dims)]
                dynamics_input = torch.cat((concat_output, actions), dim=-1)
                # store observation predictions from each agent's dynamics model given particular agent's observations
                # and actions
                agent_obs_pred_list = []
                # iterate over agents 
                for j in range(self.num_agents):
                    # dynamics_input [shape: (batch_size, fc_input_dims + act_dims)]
                    # --> dynamics [shape: (batch_size, obs_dims)]
                    agent_obs_pred = self.dynamics_list[j](dynamics_input)
                    # append agent_obs_pred to list
                    agent_obs_pred_list.append(agent_obs_pred)
                # observation predictions from each agent's dynamics model given particular agent's observations and 
                # actions
                # [shape: (batch_size, num_agents, obs_dims)] 
                obs_pred = torch.stack(agent_obs_pred_list, dim=1)
                # append obs_pred to list
                obs_pred_list.append(obs_pred)
        
        # [shape: (batch_size, num_agents, act_dims)]
        # [shape: (batch_size, num_agents, somu_n_layers / scmu_n_layers, 
        #          somu_lstm_hidden_size / scmu_lstm_hidden_size)]
        # [shape: (batch_size, num_agents, num_agents, obs_dims)] / None
        return torch.stack(actions_list, dim=1), torch.stack(action_log_probs_list, dim=1), \
               torch.stack(somu_lstm_hidden_states_list, dim=1) if self.somu_actor else None, \
               torch.stack(somu_lstm_cell_states_list, dim=1) if self.somu_actor else None, \
               torch.stack(scmu_lstm_hidden_states_list, dim=1) if self.scmu_actor else None, \
               torch.stack(scmu_lstm_cell_states_list, dim=1) if self.scmu_actor else None, \
               torch.stack(obs_pred_list, dim=1) if self.dynamics else None

    def evaluate_actions(self, obs, action, masks, available_actions=None, active_masks=None, 
                         somu_hidden_states_actor=None, somu_cell_states_actor=None, scmu_hidden_states_actor=None, 
                         scmu_cell_states_actor=None):
        # use feed forward if no somu and scmu for BOTH actor and critic
        if self.somu_actor == False and self.scmu_actor == False and self.somu_critic == False and \
           self.scmu_critic == False:
            assert somu_hidden_states_actor == None and somu_cell_states_actor == None and \
                   scmu_hidden_states_actor == None and scmu_cell_states_actor == None
            return self.evaluate_actions_feed_forward(obs=obs, 
                                                      action=action, 
                                                      available_actions=available_actions, 
                                                      active_masks=active_masks)
        # else use recurrent
        # note that somu_actor and scmu_actor can be both be False. implies that somu_critic or scmu_critic is True.
        else:
            return self.evaluate_actions_recurrent(obs=obs, 
                                                   action=action, 
                                                   masks=masks, 
                                                   available_actions=available_actions, 
                                                   active_masks=active_masks, 
                                                   somu_hidden_states_actor=somu_hidden_states_actor, 
                                                   somu_cell_states_actor=somu_cell_states_actor, 
                                                   scmu_hidden_states_actor=scmu_hidden_states_actor, 
                                                   scmu_cell_states_actor=scmu_cell_states_actor) 

    def evaluate_actions_feed_forward(self, obs, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        :return obs_pred: (torch.Tensor) observation predictions from dynamics models if used else None.
        """
        # [shape: (mini_batch_size, num_agents, obs_dims)]
        obs = check(obs).to(**self.tpdv) 
        mini_batch_size = obs.shape[0]
        # obtain batch (needed for graphnorm if being used), [shape: (mini_batch_size * num_agents)]
        if self.gnn_norm == 'graphnorm':
            batch = torch.arange(mini_batch_size).repeat_interleave(self.num_agents).to(self.device)
        # complete graph edge index (including self-loops), [shape: (2, num_agents * num_agents)] 
        if not self.knn:
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        # random node initialisation
        if self.rni:
            # zero mean std 1 gaussian noise, [shape: (mini_batch_size, num_agents, rni_dims)] 
            rni = torch.normal(0, 1, size=(mini_batch_size, self.num_agents, self.rni_dims)).to(**self.tpdv)  
            # [shape: (mini_batch_size, num_agents, obs_dims + rni_dims)]
            obs_rni = torch.cat((obs, rni), dim=-1)
        # gnn batched observations 
        obs_gnn = Batch.from_data_list([
            Data(
                x=obs_rni[i, :, :] if self.rni else obs[i, :, :], 
                edge_index=knn_graph(x=obs[i, :, :], k=self.k, loop=False).to(self.device) if self.knn else edge_index
            ) for i in range(mini_batch_size)
        ]).to(self.device)
        # [shape: (mini_batch_size, num_agents, action_space_dim)]  
        action = check(action).to(**self.tpdv)
        # [shape: (mini_batch_size * num_agents, action_space_dim)]
        action = action.reshape(mini_batch_size * self.num_agents, -1) 
        if available_actions is not None:
            # [shape: (mini_batch_size, num_agents, action_space_dim)]
            available_actions = check(available_actions).to(**self.tpdv) 
            # [shape: (mini_batch_size * num_agents, action_space_dim)]
            available_actions = available_actions.reshape(mini_batch_size * self.num_agents, -1)
        if active_masks is not None:
            # [shape: (mini_batch_size, num_agents, 1)]
            active_masks = check(active_masks).to(**self.tpdv)
            # [shape: (mini_batch_size * num_agents, 1)]
            active_masks = active_masks.reshape(mini_batch_size * self.num_agents, -1)
        # store observation predictions if dynamics models are used
        if self.dynamics:
            obs_pred_list = []

        # obs_gnn.x [shape: (mini_batch_size * num_agents, obs_dims / (obs_dims + rni_dims))] 
        # --> gnn_layers [shape: (mini_batch_size * num_agents, scmu_input_dims==fc_input_dims)] 
        if self.gnn_architecture == 'dna_gatv2' or self.gnn_architecture == 'gatv2' or self.gnn_architecture == 'gain':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None, 
                    return_attention_weights=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None, 
                    return_attention_weights=None
                )
        elif self.gnn_architecture == 'gat':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None,
                    size=None, 
                    return_attention_weights=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None,
                    size=None, 
                    return_attention_weights=None
                )
        elif self.gnn_architecture == 'gcn':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_weight=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_weight=None
                )
        elif self.gnn_architecture == 'gin':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    size=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    size=None
                )
        # concat_output [shape: (mini_batch_size * num_agents, scmu_input_dims==fc_input_dims)] --> 
        # fc_layers [shape: (mini_batch_size * num_agents, fc_output_dims)]
        fc_output = self.fc_layers(gnn_output)
        # fc_layers --> act [shape: (mini_batch_size * num_agents, act_dims)], [shape: () == scalar]
        action_log_probs, dist_entropy = \
            self.act.evaluate_actions(fc_output, 
                                      action, 
                                      available_actions, 
                                      active_masks if self._use_policy_active_masks else None)
        # get observation predictions if dynamic models are used
        if self.dynamics:
            # concatenate gnn output with actions
            # dynamics_input [shape: (mini_batch_size * num_agents, fc_input_dims + act_dims)]
            dynamics_input = torch.cat((gnn_output, action), dim=-1)
            # iterate over agents 
            for i in range(self.num_agents):
                # dynamics_input [shape: (mini_batch_size * num_agents, fc_input_dims + act_dims)]
                # --> dynamics [shape: (mini_batch_size * num_agents, obs_dims)]
                obs_pred = self.dynamics_list[i](dynamics_input)
                # append obs_pred to list
                obs_pred_list.append(obs_pred)
            # observation predictions from each agent's dynamics model given agents' observations and actions
            # [shape: (mini_batch_size * num_agents, num_agents, obs_dims)] --> 
            # [shape: (mini_batch_size, num_agents, num_agents, obs_dims)]
            obs_pred = torch.stack(obs_pred_list, dim=1)\
                            .reshape(mini_batch_size, self.num_agents, self.num_agents, self.obs_dims)

        # [shape: (mini_batch_size * num_agents, act_dims)]
        # [shape: () == scalar]
        # [shape: (mini_batch_size, num_agents, num_agents, obs_dims)] / None
        return action_log_probs, dist_entropy, obs_pred if self.dynamics else None

    def evaluate_actions_recurrent(self, obs, action, masks, available_actions=None, active_masks=None, 
                                   somu_hidden_states_actor=None, somu_cell_states_actor=None, 
                                   scmu_hidden_states_actor=None, scmu_cell_states_actor=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be initialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        :param somu_hidden_states_actor: (torch.Tensor) hidden states for somu network.
        :param somu_cell_states_actor: (torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_actor: (torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_actor: (torch.Tensor) hidden states for scmu network.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        :return obs_pred: (torch.Tensor) observation predictions from dynamics models if used else None.
        """
        # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims)]
        obs = check(obs).to(**self.tpdv) 
        mini_batch_size = obs.shape[0]
        # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims)] 
        obs_batch = obs.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims)
        # obtain batch (needed for graphnorm if being used), [shape: (mini_batch_size * data_chunk_length * num_agents)]
        if self.gnn_norm == 'graphnorm':
            batch = torch.arange(mini_batch_size * self.data_chunk_length)\
                         .repeat_interleave(self.num_agents)\
                         .to(self.device)
        # complete graph edge index (including self-loops), [shape: (2, num_agents * num_agents)] 
        if not self.knn:
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        # random node initialisation
        if self.rni:
            # zero mean std 1 gaussian noise, [shape: (mini_batch_size, data_chunk_length, num_agents, rni_dims)] 
            rni = torch.normal(0, 1, size=(mini_batch_size, self.data_chunk_length, self.num_agents, self.rni_dims))\
                       .to(**self.tpdv)  
            # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims + rni_dims)]
            obs_rni = torch.cat((obs, rni), dim=-1)
            # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims + rni_dims)] 
            obs_rni_batch = obs_rni.reshape(
                mini_batch_size * self.data_chunk_length, 
                self.num_agents, 
                self.obs_dims + self.rni_dims
            )
        # gnn batched observations 
        obs_gnn = Batch.from_data_list([
            Data(
                x=obs_rni_batch[i, :, :] if self.rni else obs_batch[i, :, :], 
                edge_index=knn_graph(x=obs_batch[i, :, :], k=self.k, loop=False).to(self.device) \
                           if self.knn else edge_index
            ) for i in range(mini_batch_size * self.data_chunk_length)
        ]).to(self.device)
        # [shape: (mini_batch_size, data_chunk_length, num_agents, action_space_dim)]  
        action = check(action).to(**self.tpdv)
        # [shape: (mini_batch_size * data_chunk_length, num_agents, action_space_dim)] 
        action = action.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, -1) 
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
        if self.somu_actor:
            # [shape: (mini_batch_size, num_agents, somu_n_layers, somu_lstm_hidden_size)]
            somu_hidden_states_actor = check(somu_hidden_states_actor).to(**self.tpdv) 
            somu_cell_states_actor = check(somu_cell_states_actor).to(**self.tpdv)
        else:
            assert somu_hidden_states_actor == None and somu_cell_states_actor == None
        if self.scmu_actor:
            # [shape: (mini_batch_size, num_agents, scmu_n_layers, scmu_lstm_hidden_size)]
            scmu_hidden_states_actor = check(scmu_hidden_states_actor).to(**self.tpdv) 
            scmu_cell_states_actor = check(scmu_cell_states_actor).to(**self.tpdv)
        else:
            assert scmu_hidden_states_actor == None and scmu_cell_states_actor == None
        # store actions and actions_log_probs
        action_log_probs_list = []
        dist_entropy_list = []
        # store observation predictions if dynamics models are used
        if self.dynamics:
            obs_pred_list = []

        # obs_gnn.x [shape: (mini_batch_size * data_chunk_length * num_agents, obs_dims / (obs_dims + rni_dims))] 
        # --> gnn_layers [shape: (mini_batch_size * data_chunk_length * num_agents, scmu_input_dims)] 
        if self.gnn_architecture == 'dna_gatv2' or self.gnn_architecture == 'gatv2' or self.gnn_architecture == 'gain':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None, 
                    return_attention_weights=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None, 
                    return_attention_weights=None
                )
        elif self.gnn_architecture == 'gat':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None,
                    size=None, 
                    return_attention_weights=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None,
                    size=None, 
                    return_attention_weights=None
                )
        elif self.gnn_architecture == 'gcn':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_weight=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_weight=None
                )
        elif self.gnn_architecture == 'gin':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    size=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    size=None
                )
        # [shape: (mini_batch_size, data_chunk_length, num_agents, scmu_input_dims)]
        gnn_output = gnn_output.reshape(mini_batch_size, self.data_chunk_length, self.num_agents, self.scmu_input_dims)

        # iterate over agents 
        for i in range(self.num_agents):
            if self.somu_actor:
                # list to store somu lstm and att outputs over sequence of inputs of len data_chunk_length
                somu_seq_lstm_output_list = []
                somu_seq_att_output_list = []
                # initialise hidden and cell states for somu at start of sequence
                # (h_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], 
                #  c_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
                somu_hidden_states = somu_hidden_states_actor[:, i, :, :].transpose(0, 1).contiguous()
                somu_cell_states = somu_cell_states_actor[:, i, :, :].transpose(0, 1).contiguous()
            if self.scmu_actor:
                # list to store scmu lstm and att outputs over sequence of inputs of len data_chunk_length
                scmu_seq_lstm_output_list = []
                scmu_seq_att_output_list = []
                # initialise hidden and cell states for scmu at start of sequence
                # (h_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], 
                #  c_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)])
                scmu_hidden_states = scmu_hidden_states_actor[:, i, :, :].transpose(0, 1).contiguous()
                scmu_cell_states = scmu_cell_states_actor[:, i, :, :].transpose(0, 1).contiguous()
            if self.somu_actor or self.scmu_actor:
                # iterate over data_chunk_length 
                for j in range(self.data_chunk_length):
                    if self.somu_actor:
                        # obs[:, j, i, :].unsqueeze(dim=1) [shape: (mini_batch_size, sequence_length=1, obs_dims)],
                        # masks[:, j, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() 
                        # [shape: (somu_n_layers, mini_batch_size, 1)],
                        # (h_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], 
                        #  c_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)]) -->
                        # somu_lstm_output [shape: (mini_batch_size, sequence_length=1, somu_lstm_hidden_size)], 
                        # (h_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], 
                        #  c_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
                        somu_lstm_output, (somu_hidden_states, somu_cell_states) = \
                            self.somu_lstm_list[i](obs[:, j, i, :].unsqueeze(dim=1),
                                                   (somu_hidden_states 
                                                    * masks[:, j, i, :].repeat(1, self.somu_n_layers)\
                                                                       .transpose(0, 1)\
                                                                       .unsqueeze(-1)\
                                                                       .contiguous(),
                                                    somu_cell_states 
                                                    * masks[:, j, i, :].repeat(1, self.somu_n_layers)\
                                                                       .transpose(0, 1)\
                                                                       .unsqueeze(-1)\
                                                                       .contiguous()))
                        somu_seq_lstm_output_list.append(somu_lstm_output)
                        if self.somu_att_actor:
                            # concatenate hidden (short-term memory) and cell (long-term memory) states for somu
                            # (h_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], 
                            #  c_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)]) -->
                            # somu_hidden_cell_states 
                            # [shape: (mini_batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)]
                            somu_hidden_cell_states = \
                                torch.cat((somu_hidden_states, somu_cell_states), dim=0).transpose(0, 1)
                            # self attention for memory from somu
                            # somu_hidden_cell_states 
                            # [shape: (mini_batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)] --> 
                            # somu_att_output [shape: (mini_batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)]
                            somu_att_output = self.somu_multi_att_layer_list[i](somu_hidden_cell_states, 
                                                                                somu_hidden_cell_states, 
                                                                                somu_hidden_cell_states)[0]
                            # [shape: (mini_batch_size, 1, 2 * somu_n_layers, somu_lstm_hidden_state)]
                            somu_seq_att_output_list.append(somu_att_output.unsqueeze(1))
                    if self.scmu_actor:
                        # gnn_output[:, j, i, :].unsqueeze(dim=1) 
                        # [shape: (mini_batch_size, sequence_length=1, scmu_input_dims)],
                        # masks[:, j, i, :].repeat(1, self.scmu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() 
                        # [shape: (scmu_n_layers, mini_batch_size, 1)],
                        # (h_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], 
                        #  c_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)]) -->
                        # scmu_lstm_output [shape: (mini_batch_size, sequence_length=1, scmu_lstm_hidden_size)], 
                        # (h_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], 
                        #  c_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)])
                        scmu_lstm_output, (scmu_hidden_states, scmu_cell_states) = \
                            self.scmu_lstm_list[i](gnn_output[:, j, i, :].unsqueeze(dim=1),
                                                   (scmu_hidden_states 
                                                    * masks[:, j, i, :].repeat(1, self.scmu_n_layers)\
                                                                       .transpose(0, 1)\
                                                                       .unsqueeze(-1)\
                                                                       .contiguous(),
                                                    scmu_cell_states 
                                                    * masks[:, j, i, :].repeat(1, self.scmu_n_layers)\
                                                                       .transpose(0, 1)\
                                                                       .unsqueeze(-1)\
                                                                       .contiguous()))
                        scmu_seq_lstm_output_list.append(scmu_lstm_output)
                        if self.scmu_att_actor == True:
                            # concatenate hidden (short-term memory) and cell (long-term memory) states for scmu
                            # (h_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], 
                            #  c_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)]) -->
                            # scmu_hidden_cell_states 
                            # [shape: (mini_batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)]
                            scmu_hidden_cell_states = \
                                torch.cat((scmu_hidden_states, scmu_cell_states), dim=0).transpose(0, 1)
                            # self attention for memory from scmu
                            # scmu_hidden_cell_states 
                            # [shape: (mini_batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)] --> 
                            # scmu_att_output [shape: (mini_batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)]
                            scmu_att_output = self.scmu_multi_att_layer_list[i](scmu_hidden_cell_states, 
                                                                                scmu_hidden_cell_states, 
                                                                                scmu_hidden_cell_states)[0]
                            # [shape: (mini_batch_size, 1, 2 * scmu_n_layers, scmu_lstm_hidden_state)]
                            scmu_seq_att_output_list.append(scmu_att_output.unsqueeze(1))
                if self.somu_actor:
                    # somu_lstm_output
                    # [shape: (mini_batch_size, data_chunk_length, somu_lstm_hidden_size)] -->
                    # [shape: (mini_batch_size * data_chunk_length, somu_lstm_hidden_size)]
                    somu_lstm_output = torch.stack(somu_seq_lstm_output_list, dim=1)\
                                            .reshape(mini_batch_size * self.data_chunk_length, 
                                                     self.somu_lstm_hidden_size)
                    if self.somu_att_actor:
                        # somu_att_output
                        # [shape: (mini_batch_size, data_chunk_length, 2 * somu_n_layers, somu_lstm_hidden_state)] --> 
                        # [shape: (mini_batch_size * data_chunk_length, 2 * somu_n_layers * somu_lstm_hidden_state)]
                        somu_att_output = torch.stack(somu_seq_att_output_list, dim=1)\
                                               .reshape(mini_batch_size * self.data_chunk_length, 
                                                        2 * self.somu_n_layers * self.somu_lstm_hidden_size)
                if self.scmu_actor:
                    # scmu_lstm_output
                    # [shape: (mini_batch_size, data_chunk_length, scmu_lstm_hidden_state)] 
                    # [shape: (mini_batch_size * data_chunk_length, scmu_lstm_hidden_state)]
                    scmu_lstm_output = torch.stack(scmu_seq_lstm_output_list, dim=1)\
                                            .reshape(mini_batch_size * self.data_chunk_length, 
                                                     self.scmu_lstm_hidden_size)
                    if self.scmu_att_actor:
                        # scmu_att_output
                        # [shape: (mini_batch_size, data_chunk_length, 2 * scmu_n_layers, scmu_lstm_hidden_state)] --> 
                        # [shape: (mini_batch_size * data_chunk_length, 2 * scmu_n_layers * scmu_lstm_hidden_state)]
                        scmu_att_output = torch.stack(scmu_seq_att_output_list, dim=1)\
                                               .reshape(mini_batch_size * self.data_chunk_length, 
                                                        2 * self.scmu_n_layers * self.scmu_lstm_hidden_size)
            # concat_output [shape: (mini_batch_size * data_chunk_length, fc_input_dims)]
            if self.somu_actor == True and self.scmu_actor == True:
                if self.somu_lstm_actor == True and self.somu_att_actor == True and self.scmu_lstm_actor == True \
                    and self.scmu_att_actor == True:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer, scmu_lstm and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output,
                        somu_att_output,
                        scmu_lstm_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == False and self.somu_att_actor == True and self.scmu_lstm_actor == True \
                    and self.scmu_att_actor == True:
                    # concatenate outputs from gnn, somu_multi_att_layer, scmu_lstm and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_att_output,
                        scmu_lstm_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == True and self.somu_att_actor == False and self.scmu_lstm_actor == True \
                    and self.scmu_att_actor == True:
                    # concatenate outputs from gnn, somu_lstm, scmu_lstm and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output,
                        scmu_lstm_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == True and self.somu_att_actor == True and self.scmu_lstm_actor == False \
                    and self.scmu_att_actor == True:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output,
                        somu_att_output,
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == True and self.somu_att_actor == True and self.scmu_lstm_actor == True \
                    and self.scmu_att_actor == False:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer and scmu_lstm
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output,
                        somu_att_output,
                        scmu_lstm_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == False and self.somu_att_actor == True and self.scmu_lstm_actor == False \
                    and self.scmu_att_actor == True:
                    # concatenate outputs from gnn, somu_multi_att_layer and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_att_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == True and self.somu_att_actor == False and self.scmu_lstm_actor == True \
                    and self.scmu_att_actor == False:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer, scmu_lstm and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output,
                        scmu_lstm_output
                        ), 
                    dim=-1)
            elif self.somu_actor == True and self.scmu_actor == False:
                if self.somu_lstm_actor == True and self.somu_att_actor == True:
                    # concatenate outputs from gnn, somu_lstm and somu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output,
                        somu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == False and self.somu_att_actor == True:
                    # concatenate outputs from gnn and somu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1),
                        somu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_actor == True and self.somu_att_actor == False:
                    # concatenate outputs from gnn and somu_lstm 
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output
                        ), 
                    dim=-1)
            elif self.somu_actor == False and self.scmu_actor == True:
                if self.scmu_lstm_actor == True and self.scmu_att_actor == True:
                    # concatenate outputs from gnn, scmu_lstm and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        scmu_lstm_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.scmu_lstm_actor == False and self.scmu_att_actor == True:
                    # concatenate outputs from gnn and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.scmu_lstm_actor == True and self.scmu_att_actor == False:
                    # concatenate outputs from gnn and scmu_lstm
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        scmu_lstm_output
                        ), 
                    dim=-1)
            else:
                assert self.somu_critic == True or self.scmu_critic == True
                concat_output = gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1)
            # concat_output [shape: (mini_batch_size * data_chunk_length, fc_input_dims)] --> 
            # fc_layers [shape: (mini_batch_size * data_chunk_length, fc_output_dims)]
            fc_output = self.fc_layers(concat_output)
            # fc_layers --> act [shape: (mini_batch_size * data_chunk_length, act_dims)], [shape: () == scalar]
            action_log_probs, dist_entropy = \
                self.act.evaluate_actions(fc_output, 
                                          action[:, i, :], available_actions[:, i, :] \
                                          if available_actions is not None else None, 
                                          active_masks[:, i, :] \
                                          if self._use_policy_active_masks and active_masks is not None else None)
            # append action_log_probs and dist_entropy to respective lists
            action_log_probs_list.append(action_log_probs)
            dist_entropy_list.append(dist_entropy)
            # get observation predictions if dynamic models are used
            if self.dynamics:
                # concatenate concat output with actions
                # dynamics_input [shape: (mini_batch_size * data_chunk_length, fc_input_dims + act_dims)]
                dynamics_input = torch.cat((concat_output, action[:, i, :]), dim=-1)
                # store observation predictions from each agent's dynamics model given particular agent's observations
                # and actions
                agent_obs_pred_list = []
                # iterate over agents 
                for j in range(self.num_agents):
                    # dynamics_input [shape: (mini_batch_size * data_chunk_length, fc_input_dims + act_dims)]
                    # --> dynamics [shape: (mini_batch_size * data_chunk_length, obs_dims)]
                    agent_obs_pred = self.dynamics_list[j](dynamics_input)
                    # append agent_obs_pred to list
                    agent_obs_pred_list.append(agent_obs_pred)
                # observation predictions from each agent's dynamics model given particular agent's observations and 
                # actions
                # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims)] 
                obs_pred = torch.stack(agent_obs_pred_list, dim=1)
                # append obs_pred to list
                obs_pred_list.append(obs_pred)

        # [shape: (mini_batch_size * data_chunk_length * num_agents, act_dims)]
        # [shape: () == scalar]
        # [shape: (mini_batch_size * data_chunk_length, num_agents, num_agents, obs_dims)] / None
        return torch.stack(action_log_probs_list, dim=1)\
                    .reshape(mini_batch_size * self.data_chunk_length * self.num_agents, -1), \
               torch.stack(dist_entropy_list, dim=0).mean(), \
               torch.stack(obs_pred_list, dim=1) if self.dynamics else None

class GCMNetCritic(nn.Module):
    """
    GCMNet Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO)
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

        self.gnn_architecture = args.gcmnet_gnn_architecture
        self.gnn_output_dims = args.gcmnet_gnn_output_dims
        self.gnn_att_heads = args.gcmnet_gnn_att_heads
        self.gnn_dna_gatv2_multi_att_heads = args.gcmnet_gnn_dna_gatv2_multi_att_heads
        self.gnn_att_concat = args.gcmnet_gnn_att_concat
        self.gnn_cpa_model = args.gcmnet_gnn_cpa_model
        self.n_gnn_layers = args.gcmnet_n_gnn_layers
        self.n_gnn_fc_layers = args.gcmnet_n_gnn_fc_layers
        self.gnn_train_eps = args.gcmnet_gnn_train_eps
        self.gnn_norm = args.gcmnet_gnn_norm

        self.somu_actor = args.gcmnet_somu_actor
        self.scmu_actor = args.gcmnet_scmu_actor
        self.somu_critic = args.gcmnet_somu_critic
        self.scmu_critic = args.gcmnet_scmu_critic
        self.somu_lstm_critic = args.gcmnet_somu_lstm_critic
        self.scmu_lstm_critic = args.gcmnet_scmu_lstm_critic
        self.somu_att_critic = args.gcmnet_somu_att_critic
        self.scmu_att_critic = args.gcmnet_scmu_att_critic
        self.somu_n_layers = args.gcmnet_somu_n_layers
        self.scmu_n_layers = args.gcmnet_scmu_n_layers
        self.somu_lstm_hidden_size = args.gcmnet_somu_lstm_hidden_size
        self.scmu_lstm_hidden_size = args.gcmnet_scmu_lstm_hidden_size
        self.somu_multi_att_n_heads = args.gcmnet_somu_multi_att_n_heads
        self.scmu_multi_att_n_heads = args.gcmnet_scmu_multi_att_n_heads

        assert not (self.somu_critic == False and self.somu_lstm_critic == True)
        assert not (self.scmu_critic == False and self.scmu_lstm_critic == True)
        assert not (self.somu_critic == False and self.somu_att_critic == True)
        assert not (self.scmu_critic == False and self.scmu_att_critic == True)
        assert not (self.somu_critic == True and self.somu_lstm_critic == False and self.somu_att_critic == False)
        assert not (self.scmu_critic == True and self.scmu_lstm_critic == False and self.scmu_att_critic == False)

        self.fc_output_dims = args.gcmnet_fc_output_dims
        self.n_fc_layers = args.gcmnet_n_fc_layers

        self.knn = args.gcmnet_knn
        self.k = args.gcmnet_k

        self.rni = args.gcmnet_rni
        self.rni_ratio = args.gcmnet_rni_ratio

        cent_obs_space = get_shape_from_obs_space(cent_obs_space)
        if len(cent_obs_space) == 3:
            raise NotImplementedError("CNN-based observations not implemented for GCMNet")
        if isinstance(cent_obs_space, (list, tuple)):
            self.obs_dims = cent_obs_space[0]
        else:
            self.obs_dims = cent_obs_space

        self.rni_dims = round(self.obs_dims * self.rni_ratio)

        # model architecture for mappo GCMNetCritic

        # gnn layers
        if self.gnn_architecture == 'dna_gatv2':
            self.gnn_layers = GNNDNALayers(
                input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                block=DNAGATv2Block, 
                output_channels=[self.obs_dims + self.rni_dims if self.rni else self.obs_dims \
                                 for _ in range(self.n_gnn_layers)],
                att_heads=self.gnn_att_heads,
                mul_att_heads=self.gnn_dna_gatv2_multi_att_heads,
                gnn_cpa_model=self.gnn_cpa_model,
                norm_type=self.gnn_norm 
            )
            # calculate relevant input dimensions
            self.scmu_input_dims = (self.n_gnn_layers + 1) * (self.obs_dims + self.rni_dims) \
                                   if self.rni else (self.n_gnn_layers + 1) * self.obs_dims
        elif self.gnn_architecture == 'gcn':
            self.gnn_layers = GNNConcatAllLayers(
                input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                block=GCNBlock, 
                output_channels=[self.gnn_output_dims for _ in range(self.n_gnn_layers)], 
                n_gnn_fc_layers=self.n_gnn_fc_layers,
                norm_type=self.gnn_norm
            )
            # calculate relevant input dimensions
            self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims + self.obs_dims + self.rni_dims \
                                   if self.rni else self.n_gnn_layers * self.gnn_output_dims + self.obs_dims 
        elif self.gnn_architecture == 'gat':
            self.gnn_layers = GNNConcatAllLayers(
                input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                block=GATBlock, 
                output_channels=[self.gnn_output_dims for _ in range(self.n_gnn_layers)], 
                heads=self.gnn_att_heads,
                concat=self.gnn_att_concat,
                gnn_cpa_model=self.gnn_cpa_model,
                norm_type=self.gnn_norm
            )
            # calculate relevant input dimensions
            if self.rni:
                self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims * self.gnn_att_heads + self.obs_dims + \
                                       self.rni_dims if self.gnn_att_concat else \
                                       self.n_gnn_layers * self.gnn_output_dims + self.obs_dims + self.rni_dims
            else: 
                self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims * self.gnn_att_heads + self.obs_dims \
                                       if self.gnn_att_concat else \
                                       self.n_gnn_layers * self.gnn_output_dims + self.obs_dims
        elif self.gnn_architecture == 'gatv2':
            self.gnn_layers = GNNConcatAllLayers(
                input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                block=GATv2Block, 
                output_channels=[self.gnn_output_dims for _ in range(self.n_gnn_layers)], 
                heads=self.gnn_att_heads,
                concat=self.gnn_att_concat,
                gnn_cpa_model=self.gnn_cpa_model,
                norm_type=self.gnn_norm
            )
            # calculate relevant input dimensions
            if self.rni:
                self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims * self.gnn_att_heads + self.obs_dims + \
                                       self.rni_dims if self.gnn_att_concat else \
                                       self.n_gnn_layers * self.gnn_output_dims + self.obs_dims + self.rni_dims
            else: 
                self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims * self.gnn_att_heads + self.obs_dims \
                                       if self.gnn_att_concat else \
                                       self.n_gnn_layers * self.gnn_output_dims + self.obs_dims
        elif self.gnn_architecture == 'gin':
            self.gnn_layers = GNNConcatAllLayers(
                input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                block=GINBlock, 
                output_channels=[self.gnn_output_dims for _ in range(self.n_gnn_layers)], 
                n_gnn_fc_layers=self.n_gnn_fc_layers,
                train_eps=self.gnn_train_eps,
                norm_type=self.gnn_norm
            )
            # calculate relevant input dimensions
            self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims + self.obs_dims + self.rni_dims \
                                   if self.rni else self.n_gnn_layers * self.gnn_output_dims + self.obs_dims
        elif self.gnn_architecture == 'gain':
            self.gnn_layers = GNNConcatAllLayers(
                input_channels=self.obs_dims + self.rni_dims if self.rni else self.obs_dims, 
                block=GAINBlock, 
                output_channels=[self.gnn_output_dims for _ in range(self.n_gnn_layers)],
                n_gnn_fc_layers=self.n_gnn_fc_layers,
                heads=self.gnn_att_heads,
                concat=self.gnn_att_concat,
                train_eps=self.gnn_train_eps,
                norm_type=self.gnn_norm
            )
            # calculate relevant input dimensions
            self.scmu_input_dims = self.n_gnn_layers * self.gnn_output_dims + self.obs_dims + self.rni_dims \
                                   if self.rni else self.n_gnn_layers * self.gnn_output_dims + self.obs_dims

        if self.somu_critic:
            # list of lstms for self observation memory unit (somu) for each agent
            # somu_lstm_input_size is the dimension of the observations
            self.somu_lstm_list = nn.ModuleList([
                nn.LSTM(
                    input_size=self.obs_dims, 
                    hidden_size=self.somu_lstm_hidden_size, 
                    num_layers=self.somu_n_layers, 
                    batch_first=True,
                    device=device
                ) for _ in range(self.num_agents)
            ])
            if self.somu_att_critic:
                # multi-head self attention layer for somu to selectively choose between the lstms outputs
                self.somu_multi_att_layer_list = nn.ModuleList([
                    nn.MultiheadAttention(
                        embed_dim=self.somu_lstm_hidden_size, 
                        num_heads=self.somu_multi_att_n_heads, 
                        dropout=0, 
                        batch_first=True, 
                        device=device
                    ) for _ in range(self.num_agents)
                ])

        if self.scmu_critic:
            # list of lstms for self communication memory unit (scmu) for each agent
            # somu_lstm_input_size are all layers of gnn
            self.scmu_lstm_list = nn.ModuleList([
                nn.LSTM(
                    input_size=self.scmu_input_dims, 
                    hidden_size=self.scmu_lstm_hidden_size, 
                    num_layers=self.scmu_n_layers, 
                    batch_first=True,
                    device=device
                ) for _ in range(self.num_agents)
            ])
            if self.scmu_att_critic:
                # multi-head self attention layer for scmu to selectively choose between the lstms outputs
                self.scmu_multi_att_layer_list = nn.ModuleList([
                    nn.MultiheadAttention(
                        embed_dim=self.scmu_lstm_hidden_size, 
                        num_heads=self.scmu_multi_att_n_heads, 
                        dropout=0, 
                        batch_first=True, 
                        device=device
                    ) for _ in range(self.num_agents)
                ])

        # calculate input dimensions for fc layers
        if self.somu_critic == True and self.scmu_critic == True:
            if self.somu_lstm_critic == True and self.somu_att_critic == True and self.scmu_lstm_critic == True \
                and self.scmu_att_critic == True:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_lstm, somu_multi_att_layer, scmu_lstm and scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers + 1) * self.somu_lstm_hidden_size \
                                     + (2 * self.scmu_n_layers + 1) * self.scmu_lstm_hidden_size
            elif self.somu_lstm_critic == False and self.somu_att_critic == True and self.scmu_lstm_critic == True \
                and self.scmu_att_critic == True:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_multi_att_layer, scmu_lstm and scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers) * self.somu_lstm_hidden_size \
                                     + (2 * self.scmu_n_layers + 1) * self.scmu_lstm_hidden_size
            elif self.somu_lstm_critic == True and self.somu_att_critic == False and self.scmu_lstm_critic == True \
                and self.scmu_att_critic == True:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_lstm, scmu_lstm and scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + self.somu_lstm_hidden_size \
                                     + (2 * self.scmu_n_layers + 1) * self.scmu_lstm_hidden_size
            elif self.somu_lstm_critic == True and self.somu_att_critic == True and self.scmu_lstm_critic == False \
                and self.scmu_att_critic == True:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_lstm, somu_multi_att_layer and scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers + 1) * self.somu_lstm_hidden_size \
                                     + (2 * self.scmu_n_layers) * self.scmu_lstm_hidden_size
            elif self.somu_lstm_critic == True and self.somu_att_critic == True and self.scmu_lstm_critic == True \
                and self.scmu_att_critic == False:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_lstm, somu_multi_att_layer and scmu_lstm
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers + 1) * self.somu_lstm_hidden_size \
                                     + self.scmu_lstm_hidden_size
            elif self.somu_lstm_critic == False and self.somu_att_critic == True and self.scmu_lstm_critic == False \
                and self.scmu_att_critic == True:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_multi_att_layer and scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers) * self.somu_lstm_hidden_size \
                                     + (2 * self.scmu_n_layers) * self.scmu_lstm_hidden_size
            elif self.somu_lstm_critic == True and self.somu_att_critic == False and self.scmu_lstm_critic == True \
                and self.scmu_att_critic == False:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of somu_lstm and scmu_lstm
                self.fc_input_dims = self.scmu_input_dims + self.somu_lstm_hidden_size + self.scmu_lstm_hidden_size
        elif self.somu_critic == True and self.scmu_critic == False:
            if self.somu_lstm_critic == True and self.somu_att_critic == True:
                # all concatenated layers of gnn (including initial observations) + 
                # concatenated outputs of somu_lstm and somu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers + 1) * self.somu_lstm_hidden_size
            elif self.somu_lstm_critic == False and self.somu_att_critic == True:
                # all concatenated layers of gnn (including initial observations) + somu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.somu_n_layers) * self.somu_lstm_hidden_size
            elif self.somu_lstm_critic == True and self.somu_att_critic == False:
                # all concatenated layers of gnn (including initial observations) + somu_lstm
                self.fc_input_dims = self.scmu_input_dims + self.somu_lstm_hidden_size
        elif self.somu_critic == False and self.scmu_critic == True:
            if self.scmu_lstm_critic == True and self.scmu_att_critic == True:
                # all concatenated layers of gnn (including initial observations) +  
                # concatenated outputs of scmu_lstm and scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.scmu_n_layers + 1) * self.scmu_lstm_hidden_size
            elif self.scmu_lstm_critic == False and self.scmu_att_critic == True:
                # all concatenated layers of gnn (including initial observations) + scmu_multi_att_layer
                self.fc_input_dims = self.scmu_input_dims + (2 * self.scmu_n_layers) * self.scmu_lstm_hidden_size
            elif self.scmu_lstm_critic == True and self.scmu_att_critic == False:
                # all concatenated layers of gnn (including initial observations) + scmu_lstm
                self.fc_input_dims = self.scmu_input_dims + self.scmu_lstm_hidden_size
        else:
            # all concatenated layers of gnn (including initial observations)
            self.fc_input_dims = self.scmu_input_dims

        # shared hidden fc layers for to generate actions for each agent
        # fc_output_dims is the list of sizes of output channels fc_block
        self.fc_layers = NNLayers(
            input_channels=self.fc_input_dims, 
            block=MLPBlock, 
            output_channels=[self.fc_output_dims for i in range(self.n_fc_layers)],
            norm_type='none', 
            activation_func='relu', 
            dropout_p=0, 
            weight_initialisation="orthogonal" if self._use_orthogonal else "default"
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # final layer for value function using popart / mlp
        if self._use_popart:
            self.v_out = init_(PopArt(self.fc_output_dims, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.fc_output_dims, 1))
        
        self.to(device)

    def forward(
            self, 
            cent_obs, 
            masks, 
            somu_hidden_states_critic=None, 
            somu_cell_states_critic=None, 
            scmu_hidden_states_critic=None, 
            scmu_cell_states_critic=None
        ):
        """
        Compute value function
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param somu_hidden_states_critic: (np.ndarray / torch.Tensor) hidden states for somu network.
        :param somu_cell_states_critic: (np.ndarray / torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_critic: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_critic: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be initialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return somu_hidden_states_critic: (torch.Tensor) hidden states for somu network.
        :return somu_cell_states_critic: (torch.Tensor) cell states for somu network.
        :return scmu_hidden_states_critic: (torch.Tensor) hidden states for scmu network.
        :return scmu_cell_states_critic: (torch.Tensor) hidden states for scmu network.
        """
        # [shape: (batch_size, num_agents, obs_dims)]
        obs = check(cent_obs).to(**self.tpdv) 
        batch_size = obs.shape[0]
        # obtain batch (needed for graphnorm if being used), [shape: (batch_size * num_agents)]
        if self.gnn_norm == 'graphnorm':
            batch = torch.arange(batch_size).repeat_interleave(self.num_agents).to(self.device)
        # complete graph edge index (including self-loops), [shape: (2, num_agents * num_agents)] 
        if not self.knn:
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        # random node initialisation
        if self.rni:
            # zero mean std 1 gaussian noise, [shape: (batch_size, num_agents, rni_dims)] 
            rni = torch.normal(0, 1, size=(batch_size, self.num_agents, self.rni_dims)).to(**self.tpdv)  
            # [shape: (batch_size, num_agents, obs_dims + rni_dims)]
            obs_rni = torch.cat((obs, rni), dim=-1)
        # gnn batched observations 
        obs_gnn = Batch.from_data_list([
            Data(
                x=obs_rni[i, :, :] if self.rni else obs[i, :, :], 
                edge_index=knn_graph(x=obs[i, :, :], k=self.k, loop=False).to(self.device) if self.knn else edge_index
            ) for i in range(batch_size)
        ]).to(self.device)
        # shape: (batch_size, num_agents, 1)
        masks = check(masks).to(**self.tpdv).reshape(batch_size, self.num_agents, -1)
        if self.somu_critic:
            # shape: (batch_size, num_agents, somu_n_layers, somu_lstm_hidden_size)
            somu_hidden_states_critic = check(somu_hidden_states_critic).to(**self.tpdv)
            somu_cell_states_critic = check(somu_cell_states_critic).to(**self.tpdv)
            # store somu hidden states and cell states
            somu_lstm_hidden_states_list = []
            somu_lstm_cell_states_list = []
        else:
            assert somu_hidden_states_critic == None and somu_cell_states_critic == None
        if self.scmu_critic:
            # shape: (batch_size, num_agents, scmu_n_layers, scmu_lstm_hidden_size) 
            scmu_hidden_states_critic = check(scmu_hidden_states_critic).to(**self.tpdv)
            scmu_cell_states_critic = check(scmu_cell_states_critic).to(**self.tpdv) 
            # store scmu hidden states and cell states
            scmu_lstm_hidden_states_list = []
            scmu_lstm_cell_states_list = []
        else:
            assert scmu_hidden_states_critic == None and scmu_cell_states_critic == None
        # store values
        values_list = []
       
        # obs_gnn.x [shape: (batch_size * num_agents, obs_dims / (obs_dims + rni_dims))] 
        # --> gnn_layers [shape: (batch_size * num_agents, scmu_input_dims)] 
        if self.gnn_architecture == 'dna_gatv2' or self.gnn_architecture == 'gatv2' or self.gnn_architecture == 'gain':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None, 
                    return_attention_weights=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None, 
                    return_attention_weights=None
                )
        elif self.gnn_architecture == 'gat':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None,
                    size=None, 
                    return_attention_weights=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None,
                    size=None, 
                    return_attention_weights=None
                )
        elif self.gnn_architecture == 'gcn':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_weight=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_weight=None
                )
        elif self.gnn_architecture == 'gin':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    size=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    size=None
                )
        # [shape: (batch_size, num_agents, scmu_input_dims)]
        gnn_output = gnn_output.reshape(batch_size, self.num_agents, self.scmu_input_dims)
       
        # iterate over agents 
        for i in range(self.num_agents):
            if self.somu_critic:
                # obs[:, i, :].unsqueeze(dim=1) [shape: (batch_size, sequence_length=1, obs_dims)],
                # masks[:, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() 
                # [shape: (somu_n_layers, batch_size, 1)],
                # (h_0 [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)], 
                #  c_0 [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)]) -->
                # somu_lstm_output [shape: (batch_size, sequence_length=1, somu_lstm_hidden_size)], 
                # (h_n [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)], 
                #  c_n [shape: (somu_n_layers, batch_size, somu_lstm_hidden_state)])
                somu_lstm_output, (somu_hidden_states, somu_cell_states) = \
                    self.somu_lstm_list[i](obs[:, i, :].unsqueeze(dim=1), 
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
                # (h_n [shape: (batch_size, somu_n_layers, somu_lstm_hidden_state], 
                #  c_n [shape: (batch_size, somu_n_layers, somu_lstm_hidden_state)])
                somu_lstm_hidden_states_list.append(somu_hidden_states.transpose(0, 1))
                somu_lstm_cell_states_list.append(somu_cell_states.transpose(0, 1))
                # somu_lstm_output [shape: (batch_size, sequence_length=1, somu_lstm_hidden_size)] --> 
                # [shape: (batch_size, somu_lstm_hidden_size)]
                somu_lstm_output = somu_lstm_output.reshape(batch_size, self.somu_lstm_hidden_size)
                if self.somu_att_critic:
                    # concatenate hidden (short-term memory) and cell (long-term memory) states for somu
                    # (h_n [shape: (batch_size, somu_n_layers, somu_lstm_hidden_state], 
                    #  c_n [shape: (batch_size, somu_n_layers, somu_lstm_hidden_state)]) --> 
                    # somu_hidden_cell_states [shape: (batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)]
                    somu_hidden_cell_states = \
                        torch.cat((somu_lstm_hidden_states_list[i], somu_lstm_cell_states_list[i]), dim=1)
                    # self attention for memory from somu
                    # somu_hidden_cell_states [shape: (batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)] --> 
                    # somu_att_output [shape: (batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)]
                    somu_att_output = self.somu_multi_att_layer_list[i](somu_hidden_cell_states, 
                                                                        somu_hidden_cell_states, 
                                                                        somu_hidden_cell_states)[0]  
                    # somu_att_output [shape: (batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)] -->
                    # [shape: (batch_size, 2 * somu_n_layers * somu_lstm_hidden_state)]
                    somu_att_output = somu_att_output.reshape(batch_size, 
                                                              2 * self.somu_n_layers * self.somu_lstm_hidden_size)
            if self.scmu_critic:
                # gnn_output[:, i, :].unsqueeze(dim=1) 
                # [shape: (batch_size, sequence_length=1, scmu_input_dims)],
                # masks[:, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() 
                # [shape: (scmu_n_layers, batch_size, 1)], 
                # (h_0 [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)], 
                #  c_0 [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)]) -->
                # scmu_lstm_output [shape: (batch_size, sequence_length=1, scmu_lstm_hidden_state)], 
                # (h_n [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)], 
                #  c_n [shape: (scmu_n_layers, batch_size, scmu_lstm_hidden_state)])
                scmu_lstm_output, (scmu_hidden_states, scmu_cell_states) = \
                    self.scmu_lstm_list[i](gnn_output[:, i, :].unsqueeze(dim=1), 
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
                # (h_n [shape: (batch_size, scmu_n_layers, scmu_lstm_hidden_state)], 
                #  c_n [shape: (batch_size, scmu_n_layers, scmu_lstm_hidden_state)])
                scmu_lstm_hidden_states_list.append(scmu_hidden_states.transpose(0, 1))
                scmu_lstm_cell_states_list.append(scmu_cell_states.transpose(0, 1))
                # scmu_lstm_output [shape: (batch_size, sequence_length=1, scmu_lstm_hidden_size)] --> 
                # [shape: (batch_size, scmu_lstm_hidden_size)]
                scmu_lstm_output = scmu_lstm_output.reshape(batch_size, self.scmu_lstm_hidden_size)
                if self.scmu_att_critic:
                    # concatenate hidden (short-term memory) and cell (long-term memory) states for somu and scmu
                    # (h_n [shape: (batch_size, scmu_n_layers, scmu_lstm_hidden_state)], 
                    #  c_n [shape: (batch_size, scmu_n_layers, scmu_lstm_hidden_state)]) --> 
                    # scmu_hidden_cell_states [shape: (batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)]
                    scmu_hidden_cell_states = \
                        torch.cat((scmu_lstm_hidden_states_list[i], scmu_lstm_cell_states_list[i]), dim=1)
                    # self attention for memory from scmu
                    # scmu_hidden_cell_states [shape: (batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)] --> 
                    # scmu_att_output [shape: (batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)]
                    scmu_att_output = self.scmu_multi_att_layer_list[i](scmu_hidden_cell_states, 
                                                                        scmu_hidden_cell_states, 
                                                                        scmu_hidden_cell_states)[0]
                    # scmu_att_output [shape: (batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)] -->
                    # [shape: (batch_size, 2 * scmu_n_layers * scmu_lstm_hidden_state)]
                    scmu_att_output = scmu_att_output.reshape(batch_size, 
                                                              2 * self.scmu_n_layers * self.scmu_lstm_hidden_size)
            # concat_output [shape: (batch_size, fc_input_dims)]
            if self.somu_critic == True and self.scmu_critic == True:
                if self.somu_lstm_critic == True and self.somu_att_critic == True and self.scmu_lstm_critic == True \
                    and self.scmu_att_critic == True:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer, scmu_lstm and 
                    # scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output, 
                        somu_att_output, 
                        scmu_lstm_output,
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == False and self.somu_att_critic == True and self.scmu_lstm_critic == True \
                    and self.scmu_att_critic == True:
                    # concatenate outputs from gnn, somu_multi_att_layer, scmu_lstm and scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_att_output, 
                        scmu_lstm_output,
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == True and self.somu_att_critic == False and self.scmu_lstm_critic == True \
                    and self.scmu_att_critic == True:
                    # concatenate outputs from gnn, somu_lstm, scmu_lstm and scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output,  
                        scmu_lstm_output,
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == True and self.somu_att_critic == True and self.scmu_lstm_critic == False \
                    and self.scmu_att_critic == True:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer and scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output, 
                        somu_att_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == True and self.somu_att_critic == True and self.scmu_lstm_critic == True \
                    and self.scmu_att_critic == False:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer and scmu_lstm
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output, 
                        somu_att_output, 
                        scmu_lstm_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == False and self.somu_att_critic == True and self.scmu_lstm_critic == False \
                    and self.scmu_att_critic == True:
                    # concatenate outputs from gnn, somu_multi_att_layer and scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_att_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == True and self.somu_att_critic == False and self.scmu_lstm_critic == True \
                    and self.scmu_att_critic == False:
                    # concatenate outputs from gnn, somu_lstm and scmu_lstm 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output, 
                        scmu_lstm_output
                        ), 
                    dim=-1)
            elif self.somu_critic == True and self.scmu_critic == False:
                if self.somu_lstm_critic == True and self.somu_att_critic == True:
                    # concatenate outputs from gnn, somu_lstm and somu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output,
                        somu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == False and self.somu_att_critic == True:
                    # concatenate outputs from gnn and somu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == True and self.somu_att_critic == False:
                    # concatenate outputs from gnn and somu_lstm
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        somu_lstm_output
                        ), 
                    dim=-1)
            elif self.somu_critic == False and self.scmu_critic == True:
                if self.scmu_lstm_critic == True and self.scmu_att_critic == True:
                    # concatenate outputs from gnn, scmu_lstm and scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        scmu_lstm_output,
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.scmu_lstm_critic == False and self.scmu_att_critic == True:
                    # concatenate outputs from gnn and scmu_multi_att_layer 
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.scmu_lstm_critic == True and self.scmu_att_critic == False:
                    # concatenate outputs from gnn and scmu_lstm
                    concat_output = torch.cat((
                        gnn_output[:, i, :], 
                        scmu_lstm_output
                        ), 
                    dim=-1)
            else:
                # all concatenated layers of gnn (including initial observations)
                concat_output = gnn_output[:, i, :]
            # concat_output [shape: (batch_size, fc_input_dims)] --> fc_layers [shape: (batch_size, fc_output_dims)]
            fc_output = self.fc_layers(concat_output)
            # fc_layers [shape: (batch_size, fc_output_dims)] --> v_out [shape: (batch_size, 1)]
            values = self.v_out(fc_output)
            values_list.append(values)

        # [shape: (batch_size, num_agents, 1)]
        # [shape: (batch_size, num_agents, somu_n_layers / scmu_n_layers, 
        #          somu_lstm_hidden_size / scmu_lstm_hidden_size)]
        return torch.stack(values_list, dim=1), \
               torch.stack(somu_lstm_hidden_states_list, dim=1) if self.somu_critic else None, \
               torch.stack(somu_lstm_cell_states_list, dim=1) if self.somu_critic else None, \
               torch.stack(scmu_lstm_hidden_states_list, dim=1) if self.scmu_critic else None, \
               torch.stack(scmu_lstm_cell_states_list, dim=1) if self.scmu_critic else None

    def evaluate_actions(self, cent_obs, masks, somu_hidden_states_critic=None, somu_cell_states_critic=None, 
                         scmu_hidden_states_critic=None, scmu_cell_states_critic=None):
        # use feed forward if no somu and scmu for BOTH actor and critic
        if self.somu_actor == False and self.scmu_actor == False and self.somu_critic == False and \
           self.scmu_critic == False:
            assert somu_hidden_states_critic == None and somu_cell_states_critic == None and \
                   scmu_hidden_states_critic == None and scmu_cell_states_critic == None
            return self.evaluate_actions_feed_forward(cent_obs=cent_obs)
        # else use recurrent
        # note that somu_critic and scmu_critic can be both be False. implies that somu_actor or scmu_actor is True.
        else:
            return self.evaluate_actions_recurrent(cent_obs=cent_obs, 
                                                   masks=masks,
                                                   somu_hidden_states_critic=somu_hidden_states_critic, 
                                                   somu_cell_states_critic=somu_cell_states_critic, 
                                                   scmu_hidden_states_critic=scmu_hidden_states_critic, 
                                                   scmu_cell_states_critic=scmu_cell_states_critic)

    def evaluate_actions_feed_forward(self, cent_obs):
        """
        Compute value function
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.

        :return values: (torch.Tensor) value function predictions.
        """
        # [shape: (mini_batch_size, num_agents, obs_dims)]
        obs = check(cent_obs).to(**self.tpdv) 
        mini_batch_size = obs.shape[0]
        # obtain batch (needed for graphnorm if being used), [shape: (mini_batch_size * num_agents)]
        if self.gnn_norm == 'graphnorm':
            batch = torch.arange(mini_batch_size).repeat_interleave(self.num_agents).to(self.device)
        # complete graph edge index (including self-loops), [shape: (2, num_agents * num_agents)] 
        if not self.knn:
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        # random node initialisation
        if self.rni:
            # zero mean std 1 gaussian noise, [shape: (mini_batch_size, num_agents, rni_dims)] 
            rni = torch.normal(0, 1, size=(mini_batch_size, self.num_agents, self.rni_dims)).to(**self.tpdv)  
            # [shape: (mini_batch_size, num_agents, obs_dims + rni_dims)]
            obs_rni = torch.cat((obs, rni), dim=-1)
        # gnn batched observations 
        obs_gnn = Batch.from_data_list([
            Data(
                x=obs_rni[i, :, :] if self.rni else obs[i, :, :], 
                edge_index=knn_graph(x=obs[i, :, :], k=self.k, loop=False).to(self.device) if self.knn else edge_index
            ) for i in range(mini_batch_size)
        ]).to(self.device)

        # obs_gnn.x [shape: (mini_batch_size * num_agents, obs_dims / (obs_dims + rni_dims))] 
        # --> gnn_layers [shape: (mini_batch_size * num_agents, scmu_input_dims==fc_input_dims)] 
        if self.gnn_architecture == 'dna_gatv2' or self.gnn_architecture == 'gatv2' or self.gnn_architecture == 'gain':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None, 
                    return_attention_weights=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None, 
                    return_attention_weights=None
                )
        elif self.gnn_architecture == 'gat':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None,
                    size=None, 
                    return_attention_weights=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None,
                    size=None, 
                    return_attention_weights=None
                )
        elif self.gnn_architecture == 'gcn':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_weight=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_weight=None
                )
        elif self.gnn_architecture == 'gin':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    size=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    size=None
                )
        # concat_output [shape: (mini_batch_size * num_agents, scmu_input_dims==fc_input_dims)] --> 
        # fc_layers [shape: (mini_batch_size * num_agents, fc_output_dims)]
        fc_output = self.fc_layers(gnn_output)
        # fc_layers [shape: (mini_batch_size * num_agents, fc_output_dims)] --> 
        # v_out [shape: (mini_batch_size * num_agents, 1)]
        values = self.v_out(fc_output)

        # [shape: (mini_batch_size * num_agents, 1)]
        return values

    def evaluate_actions_recurrent(self, cent_obs, masks, somu_hidden_states_critic=None, somu_cell_states_critic=None, 
                                   scmu_hidden_states_critic=None, scmu_cell_states_critic=None):
        """
        Compute value function
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param somu_hidden_states_critic: (np.ndarray / torch.Tensor) hidden states for somu network.
        :param somu_cell_states_critic: (np.ndarray / torch.Tensor) cell states for somu network.
        :param scmu_hidden_states_critic: (np.ndarray / torch.Tensor) hidden states for scmu network.
        :param scmu_cell_states_critic: (np.ndarray / torch.Tensor) hidden states for scmu network.

        :return values: (torch.Tensor) value function predictions.
        """
        # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims)]
        obs = check(cent_obs).to(**self.tpdv) 
        mini_batch_size = obs.shape[0]
        # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims)] 
        obs_batch = obs.reshape(mini_batch_size * self.data_chunk_length, self.num_agents, self.obs_dims)
        # obtain batch (needed for graphnorm if being used), [shape: (mini_batch_size * data_chunk_length * num_agents)]
        if self.gnn_norm == 'graphnorm':
            batch = torch.arange(mini_batch_size * self.data_chunk_length)\
                         .repeat_interleave(self.num_agents)\
                         .to(self.device)
        # complete graph edge index (including self-loops), [shape: (2, num_agents * num_agents)] 
        if not self.knn:
            edge_index = complete_graph_edge_index(self.num_agents) 
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        # random node initialisation
        if self.rni:
            # zero mean std 1 gaussian noise, [shape: (mini_batch_size, data_chunk_length, num_agents, rni_dims)] 
            rni = torch.normal(0, 1, size=(mini_batch_size, self.data_chunk_length, self.num_agents, self.rni_dims))\
                       .to(**self.tpdv)  
            # [shape: (mini_batch_size, data_chunk_length, num_agents, obs_dims + rni_dims)]
            obs_rni = torch.cat((obs, rni), dim=-1)
            # [shape: (mini_batch_size * data_chunk_length, num_agents, obs_dims + rni_dims)] 
            obs_rni_batch = obs_rni.reshape(
                mini_batch_size * self.data_chunk_length, 
                self.num_agents, 
                self.obs_dims + self.rni_dims
            )
        # gnn batched observations 
        obs_gnn = Batch.from_data_list([
            Data(
                x=obs_rni_batch[i, :, :] if self.rni else obs_batch[i, :, :], 
                edge_index=knn_graph(x=obs_batch[i, :, :], k=self.k, loop=False).to(self.device) \
                           if self.knn else edge_index
            ) for i in range(mini_batch_size * self.data_chunk_length)
        ]).to(self.device)
        # [shape: (mini_batch_size, data_chunk_length, num_agents, 1)]  
        masks = check(masks).to(**self.tpdv)
        if self.somu_critic:
            # [shape: (mini_batch_size, num_agents, somu_n_layers, somu_lstm_hidden_size)]
            somu_hidden_states_critic = check(somu_hidden_states_critic).to(**self.tpdv) 
            somu_cell_states_critic = check(somu_cell_states_critic).to(**self.tpdv)
        else:
            assert somu_hidden_states_critic == None and somu_cell_states_critic == None
        if self.scmu_critic:
            # [shape: (mini_batch_size, num_agents, scmu_n_layers, scmu_lstm_hidden_size)]
            scmu_hidden_states_critic = check(scmu_hidden_states_critic).to(**self.tpdv) 
            scmu_cell_states_critic = check(scmu_cell_states_critic).to(**self.tpdv)
        else:
            assert scmu_hidden_states_critic == None and scmu_cell_states_critic == None
        # list to store values
        values_list = []

        # obs_gnn.x [shape: (mini_batch_size * data_chunk_length * num_agents, obs_dims / (obs_dims + rni_dims))] 
        # --> gnn_layers [shape: (mini_batch_size * data_chunk_length * num_agents, scmu_input_dims)] 
        if self.gnn_architecture == 'dna_gatv2' or self.gnn_architecture == 'gatv2' or self.gnn_architecture == 'gain':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None, 
                    return_attention_weights=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None, 
                    return_attention_weights=None
                )
        elif self.gnn_architecture == 'gat':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None,
                    size=None, 
                    return_attention_weights=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_attr=None,
                    size=None, 
                    return_attention_weights=None
                )
        elif self.gnn_architecture == 'gcn':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_weight=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    edge_weight=None
                )
        elif self.gnn_architecture == 'gin':
            if self.gnn_norm == 'graphnorm':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    size=None, 
                    batch=batch
                )
            elif self.gnn_norm == 'none':
                gnn_output = self.gnn_layers(
                    x=obs_gnn.x, 
                    edge_index=obs_gnn.edge_index, 
                    size=None
                )
        # [shape: (mini_batch_size, data_chunk_length, num_agents, scmu_input_dims)]
        gnn_output = gnn_output.reshape(mini_batch_size, self.data_chunk_length, self.num_agents, self.scmu_input_dims)

        # iterate over agents 
        for i in range(self.num_agents):
            if self.somu_critic:
                # list to store somu lstm and att outputs over sequence of inputs of len data_chunk_length
                somu_seq_lstm_output_list = []
                somu_seq_att_output_list = []
                # initialise hidden and cell states for somu at start of sequence
                # (h_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], 
                #  c_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
                somu_hidden_states = somu_hidden_states_critic[:, i, :, :].transpose(0, 1).contiguous()
                somu_cell_states = somu_cell_states_critic[:, i, :, :].transpose(0, 1).contiguous()
            if self.scmu_critic:
                # list to store scmu lstm and att outputs over sequence of inputs of len data_chunk_length
                scmu_seq_lstm_output_list = []
                scmu_seq_att_output_list = []
                # initialise hidden and cell states for scmu at start of sequence
                # (h_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], 
                #  c_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)])
                scmu_hidden_states = scmu_hidden_states_critic[:, i, :, :].transpose(0, 1).contiguous()
                scmu_cell_states = scmu_cell_states_critic[:, i, :, :].transpose(0, 1).contiguous()
            if self.somu_critic or self.scmu_critic:
                # iterate over data_chunk_length 
                for j in range(self.data_chunk_length):
                    if self.somu_critic:
                        # obs[:, j, i, :].unsqueeze(dim=1) [shape: (mini_batch_size, sequence_length=1, obs_dims)],
                        # masks[:, j, i, :].repeat(1, self.somu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() 
                        # [shape: (somu_n_layers, mini_batch_size, 1)],
                        # (h_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], 
                        #  c_0 [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)]) -->
                        # somu_lstm_output [shape: (mini_batch_size, sequence_length=1, somu_lstm_hidden_size)], 
                        # (h_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], 
                        #  c_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)])
                        somu_lstm_output, (somu_hidden_states, somu_cell_states) = \
                            self.somu_lstm_list[i](obs[:, j, i, :].unsqueeze(dim=1),
                                                   (somu_hidden_states 
                                                    * masks[:, j, i, :].repeat(1, self.somu_n_layers)\
                                                                       .transpose(0, 1)\
                                                                       .unsqueeze(-1)\
                                                                       .contiguous(),
                                                    somu_cell_states 
                                                    * masks[:, j, i, :].repeat(1, self.somu_n_layers)\
                                                                       .transpose(0, 1)\
                                                                       .unsqueeze(-1)\
                                                                       .contiguous()))
                        somu_seq_lstm_output_list.append(somu_lstm_output)
                        if self.somu_att_critic:
                            # concatenate hidden (short-term memory) and cell (long-term memory) states for somu
                            # (h_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)], 
                            #  c_n [shape: (somu_n_layers, mini_batch_size, somu_lstm_hidden_state)]) -->
                            # somu_hidden_cell_states 
                            # [shape: (mini_batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)]
                            somu_hidden_cell_states = \
                                torch.cat((somu_hidden_states, somu_cell_states), dim=0).transpose(0, 1)
                            # self attention for memory from somu
                            # somu_hidden_cell_states 
                            # [shape: (mini_batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)] --> 
                            # somu_att_output [shape: (mini_batch_size, 2 * somu_n_layers, somu_lstm_hidden_state)]
                            somu_att_output = self.somu_multi_att_layer_list[i](somu_hidden_cell_states, 
                                                                                somu_hidden_cell_states, 
                                                                                somu_hidden_cell_states)[0]
                            # [shape: (mini_batch_size, 1, 2 * somu_n_layers, somu_lstm_hidden_state)]
                            somu_seq_att_output_list.append(somu_att_output.unsqueeze(1))
                    if self.scmu_critic:
                        # gnn_output[:, j, i, :].unsqueeze(dim=1) 
                        # [shape: (mini_batch_size, sequence_length=1, scmu_input_dims)],
                        # masks[:, j, i, :].repeat(1, self.scmu_n_layers).transpose(0, 1).unsqueeze(-1).contiguous() 
                        # [shape: (scmu_n_layers, mini_batch_size, 1)],
                        # (h_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], 
                        #  c_0 [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)]) -->
                        # scmu_lstm_output [shape: (mini_batch_size, sequence_length=1, scmu_lstm_hidden_size)], 
                        # (h_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], 
                        #  c_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)])
                        scmu_lstm_output, (scmu_hidden_states, scmu_cell_states) = \
                            self.scmu_lstm_list[i](gnn_output[:, j, i, :].unsqueeze(dim=1),
                                                   (scmu_hidden_states 
                                                    * masks[:, j, i, :].repeat(1, self.scmu_n_layers)\
                                                                       .transpose(0, 1)\
                                                                       .unsqueeze(-1)\
                                                                       .contiguous(),
                                                    scmu_cell_states 
                                                    * masks[:, j, i, :].repeat(1, self.scmu_n_layers)\
                                                                       .transpose(0, 1)\
                                                                       .unsqueeze(-1)\
                                                                       .contiguous()))
                        scmu_seq_lstm_output_list.append(scmu_lstm_output)
                        if self.scmu_att_critic == True:
                            # concatenate hidden (short-term memory) and cell (long-term memory) states for scmu
                            # (h_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)], 
                            #  c_n [shape: (scmu_n_layers, mini_batch_size, scmu_lstm_hidden_state)]) -->
                            # scmu_hidden_cell_states 
                            # [shape: (mini_batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)]
                            scmu_hidden_cell_states = \
                                torch.cat((scmu_hidden_states, scmu_cell_states), dim=0).transpose(0, 1)
                            # self attention for memory from scmu
                            # scmu_hidden_cell_states 
                            # [shape: (mini_batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)] --> 
                            # scmu_att_output [shape: (mini_batch_size, 2 * scmu_n_layers, scmu_lstm_hidden_state)]
                            scmu_att_output = self.scmu_multi_att_layer_list[i](scmu_hidden_cell_states, 
                                                                                scmu_hidden_cell_states, 
                                                                                scmu_hidden_cell_states)[0]
                            # [shape: (mini_batch_size, 1, 2 * scmu_n_layers, scmu_lstm_hidden_state)]
                            scmu_seq_att_output_list.append(scmu_att_output.unsqueeze(1))
                if self.somu_critic:
                    # somu_lstm_output
                    # [shape: (mini_batch_size, data_chunk_length, somu_lstm_hidden_size)] -->
                    # [shape: (mini_batch_size * data_chunk_length, somu_lstm_hidden_size)]
                    somu_lstm_output = torch.stack(somu_seq_lstm_output_list, dim=1)\
                                            .reshape(mini_batch_size * self.data_chunk_length, 
                                                     self.somu_lstm_hidden_size)
                    if self.somu_att_critic:
                        # somu_att_output
                        # [shape: (mini_batch_size, data_chunk_length, 2 * somu_n_layers, somu_lstm_hidden_state)] --> 
                        # [shape: (mini_batch_size * data_chunk_length, 2 * somu_n_layers * somu_lstm_hidden_state)]
                        somu_att_output = torch.stack(somu_seq_att_output_list, dim=1)\
                                               .reshape(mini_batch_size * self.data_chunk_length, 
                                                        2 * self.somu_n_layers * self.somu_lstm_hidden_size)
                if self.scmu_critic:
                    # scmu_lstm_output
                    # [shape: (mini_batch_size, data_chunk_length, scmu_lstm_hidden_state)] 
                    # [shape: (mini_batch_size * data_chunk_length, scmu_lstm_hidden_state)]
                    scmu_lstm_output = torch.stack(scmu_seq_lstm_output_list, dim=1)\
                                            .reshape(mini_batch_size * self.data_chunk_length, 
                                                     self.scmu_lstm_hidden_size)
                    if self.scmu_att_critic:
                        # scmu_att_output
                        # [shape: (mini_batch_size, data_chunk_length, 2 * scmu_n_layers, scmu_lstm_hidden_state)] --> 
                        # [shape: (mini_batch_size * data_chunk_length, 2 * scmu_n_layers * scmu_lstm_hidden_state)]
                        scmu_att_output = torch.stack(scmu_seq_att_output_list, dim=1)\
                                               .reshape(mini_batch_size * self.data_chunk_length, 
                                                        2 * self.scmu_n_layers * self.scmu_lstm_hidden_size)
            # concat_output [shape: (mini_batch_size * data_chunk_length, fc_input_dims)]
            if self.somu_critic == True and self.scmu_critic == True:
                if self.somu_lstm_critic == True and self.somu_att_critic == True and self.scmu_lstm_critic == True \
                    and self.scmu_att_critic == True:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer, scmu_lstm and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output,
                        somu_att_output,
                        scmu_lstm_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == False and self.somu_att_critic == True and self.scmu_lstm_critic == True \
                    and self.scmu_att_critic == True:
                    # concatenate outputs from gnn, somu_multi_att_layer, scmu_lstm and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_att_output,
                        scmu_lstm_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == True and self.somu_att_critic == False and self.scmu_lstm_critic == True \
                    and self.scmu_att_critic == True:
                    # concatenate outputs from gnn, somu_lstm, scmu_lstm and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output,
                        scmu_lstm_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == True and self.somu_att_critic == True and self.scmu_lstm_critic == False \
                    and self.scmu_att_critic == True:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output,
                        somu_att_output,
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == True and self.somu_att_critic == True and self.scmu_lstm_critic == True \
                    and self.scmu_att_critic == False:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer and scmu_lstm
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output,
                        somu_att_output,
                        scmu_lstm_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == False and self.somu_att_critic == True and self.scmu_lstm_critic == False \
                    and self.scmu_att_critic == True:
                    # concatenate outputs from gnn, somu_multi_att_layer and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_att_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == True and self.somu_att_critic == False and self.scmu_lstm_critic == True \
                    and self.scmu_att_critic == False:
                    # concatenate outputs from gnn, somu_lstm, somu_multi_att_layer, scmu_lstm and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output,
                        scmu_lstm_output
                        ), 
                    dim=-1)
            elif self.somu_critic == True and self.scmu_critic == False:
                if self.somu_lstm_critic == True and self.somu_att_critic == True:
                    # concatenate outputs from gnn, somu_lstm and somu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output,
                        somu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == False and self.somu_att_critic == True:
                    # concatenate outputs from gnn and somu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1),
                        somu_att_output
                        ), 
                    dim=-1)
                elif self.somu_lstm_critic == True and self.somu_att_critic == False:
                    # concatenate outputs from gnn and somu_lstm 
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        somu_lstm_output
                        ), 
                    dim=-1)
            elif self.somu_critic == False and self.scmu_critic == True:
                if self.scmu_lstm_critic == True and self.scmu_att_critic == True:
                    # concatenate outputs from gnn, scmu_lstm and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        scmu_lstm_output, 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.scmu_lstm_critic == False and self.scmu_att_critic == True:
                    # concatenate outputs from gnn and scmu_multi_att_layer
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        scmu_att_output
                        ), 
                    dim=-1)
                elif self.scmu_lstm_critic == True and self.scmu_att_critic == False:
                    # concatenate outputs from gnn and scmu_lstm
                    concat_output = torch.cat((
                        gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1), 
                        scmu_lstm_output
                        ), 
                    dim=-1)
            else:
                assert self.somu_actor == True or self.scmu_actor == True
                concat_output = gnn_output[:, :, i, :].reshape(mini_batch_size * self.data_chunk_length, -1)    
            # concat_output [shape: (mini_batch_size * data_chunk_length, fc_input_dims)] --> 
            # fc_layers [shape: (mini_batch_size * data_chunk_length, fc_output_dims)]
            fc_output = self.fc_layers(concat_output)
            # fc_layers [shape: (mini_batch_size * data_chunk_length, fc_output_dims)] --> 
            # v_out [shape: (mini_batch_size * data_chunk_length, 1)]
            values = self.v_out(fc_output)
            values_list.append(values)

        # [shape: (mini_batch_size * data_chunk_length * num_agents, 1)]
        return torch.stack(values_list, dim=1).reshape(-1, 1)