# ==========================================================================================================================================================
# neural network (nn) module
# purpose: classes and functions to build a scalable neural network model
# ==========================================================================================================================================================

import torch as T
import torch.nn as nn
import torch_geometric.nn as gnn

from functools import partial
from onpolicy.algorithms.utils.dgcn import DGCNConv

def activation_function(activation):
    """
    function that returns ModuleDict of activation functions
    """
    return  nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['sigmoid', nn.Sigmoid()],
        ['softmax', nn.Softmax(1)],
        ['log_softmax', nn.LogSoftmax(1)],
        ['tanh', nn.Tanh()],
        ['hard_tanh', nn.Hardtanh()],
        ['none', nn.Identity()]
    ])[activation]

def weights_initialisation_function_generator(weight_intialisation, 
                                              activation_func, *args, **kwargs):
    """ 
    function that returns functions initialise weights according to specified 
    methods. 
    """
    if weight_intialisation == "xavier_uniform":
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 
                    gain=nn.init.calculate_gain(activation_func))
        return init_weight
    elif weight_intialisation == "xavier_normal":
        def init_weight(m):
            if isinstance(m, nn.Linear):  
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(activation_func))
        return init_weight
    elif weight_intialisation == "kaiming_uniform":
        # recommend for relu / leaky relu
        assert (activation_func == "relu" or activation_func == "leaky_relu"), "Non-linearity recommended to be 'relu' or 'leaky_relu'"
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=kwargs.get("kaiming_a", math.sqrt(5)), mode=kwargs.get("kaiming_mode", "fan_in"), nonlinearity=activation_func)
        return init_weight
    elif weight_intialisation == "kaiming_normal":
        # recommend for relu / leaky relu
        assert (activation_func == "relu" or activation_func == "leaky_relu"), "Non-linearity recommended to be 'relu' or 'leaky_relu'"
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=kwargs.get("kaiming_a", math.sqrt(5)), mode=kwargs.get("kaiming_mode", "fan_in"), nonlinearity=activation_func)
        return init_weight
    elif weight_intialisation == "uniform":
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=kwargs.get("uniform_lower_bound", 0.0), b=kwargs.get("uniform_upper_bound", 1.0))

        return init_weight
    else:
        def init_weight(m):
            pass
        return init_weight

class MLPBlock(nn.Module):
    """
    class to build basic fully connected block 
    """
    
    def __init__(self, input_shape, output_shape, activation_func="relu", weight_initialisation="default", *args, **kwargs):
        
        """
        class constructor that creates the layers attributes for MLPBlock 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input and output units for hidden layer 
        self.input_shape = input_shape
        self.output_shape = output_shape
        # activation function for after batch norm
        self.activation_func = activation_func 
        # dropout probablity
        self.dropout_p = kwargs['dropout_p']

        # basic fc_block. inpuit --> linear --> batch norm --> activation function --> dropout 
        self.block = nn.Sequential(
            # linear hidden layer
            nn.Linear(self.input_shape, self.output_shape, bias = False),
            # layer norm
            nn.LayerNorm(self.output_shape),
            # activation func
            activation_function(self.activation_func)
        )
        
        # weight initialisation
        self.block.apply(weights_initialisation_function_generator(weight_initialisation, activation_func, *args, **kwargs))

    def forward(self, x):
        """ 
        function for forward pass of fc_block 
        """
        x = self.block(x)
        
        return x

class DGCNBlock(nn.Module):
    """ 
    class to build DGCNBlock 
    """

    def __init__(self, input_channels, output_channels, att_heads=1, mul_att_heads=1, groups=1, concat=True, negative_slope=0.2, dropout=0.0, add_self_loops=True, edge_dim=None, fill_value='mean', bias=True, weight_initialisation=None,activation_func =None):
        """ 
        class constructor for attributes of the DGCNBlock 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()

        # input and output channels for dgcn 
        self.input_channels = input_channels
        self.output_channels = output_channels
        # number of heads for gatv2 and multi head attention
        self.att_heads = att_heads
        self.mul_att_heads = mul_att_heads
        # number of groups for grouped operations multi head attention
        self.groups = groups
        # boolean that when set to false, the gatv2 multi-head attentions are averaged instead of concatenated
        self.concat = concat
        # negative slope of leaky relu
        self.negative_slope = negative_slope
        # dropout probablity
        self.dropout = dropout
        # boolean to add self loops
        self.add_self_loops = add_self_loops
        # dimensions of edge attributes if any
        self.edge_dim = edge_dim
        # fill value for edge attributes for self loops
        self.fill_value = fill_value
        # boolean for bias
        self.bias = bias

        # basic dgcn_block. input --> dgcn --> graph norm --> activation func
        self.block = gnn.Sequential('x, edge_index', 
                                    [
                                        # dgcn block 
                                        (DGCNConv(in_channels = input_channels, out_channels = output_channels, att_heads = att_heads, mul_att_heads = mul_att_heads, groups = groups, concat = concat,  
                                                  negative_slope = negative_slope, dropout = dropout, add_self_loops = add_self_loops, edge_dim = edge_dim, fill_value = fill_value, bias = bias), 
                                         'x, edge_index -> x'), 
                                        # graph norm
                                        (gnn.GraphNorm(self.output_channels * self.att_heads if concat == True else self.output_channels), 'x -> x')
                                    ]
        )  

    def forward(self, x, edge_index):
        """ 
        function for forward pass of gatv2_block 
        """
        x = self.block(x, edge_index)
        
        return x

class GATv2Block(nn.Module):
    """ 
    class to build GATv2Block 
    """

    def __init__(self, input_channels, output_channels, num_heads=1, concat=True, dropout=0.0, activation_func="relu"):
        """ 
        class constructor for attributes of the GATv2Block 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()

        # input and output channels for gatv2 (embedding dimension for each node)
        self.input_channels = input_channels
        self.output_channels = output_channels
        # number of heads for gatv2
        self.num_heads = num_heads
        # boolean that when set to false, the multi-head attentions are averaged instead of concatenated
        self.concat = concat
        # dropout probablity
        self.dropout = dropout
        # activation function for after GATv2Conv 
        self.activation_func = activation_func

        # basic gatv2_block. input --> GATv2Conv --> GraphNorm --> activation func
        self.block = gnn.Sequential('x, edge_index', 
                                    [
                                        # GATv2Conv 
                                        (gnn.GATv2Conv(in_channels = self.input_channels, out_channels = self.output_channels, heads = self.num_heads, concat = concat, dropout = dropout), 
                                         'x, edge_index -> x'), 
                                        # graph norm
                                        (gnn.GraphNorm(self.output_channels * self.num_heads if concat == True else self.output_channels), 'x -> x'),
                                        # activation func
                                        activation_function(self.activation_func)
                                    ]
        )

    def forward(self, x, edge_index):
        """ 
        function for forward pass of gatv2_block 
        """
        x = self.block(x, edge_index)
        
        return x

class NNLayers(nn.Module):
    """ 
    class to build layers of blocks
    """
    
    def __init__(self, input_channels, block, output_channels, *args, **kwargs):
        """ 
        class constructor for attributes of NNLayers 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input channels/shape
        self.input_channels = input_channels
        # class of block
        self.block = block
        # output channels/shape
        self.output_channels = output_channels
        self.input_output_list = list(zip(output_channels[:], output_channels[1:]))
        # module list of layers with same args and kwargs
        self.blocks = nn.ModuleList([
            self.block(self.input_channels, self.output_channels[0], *args, **kwargs),
            *[self.block(input_channels, output_channels, *args, **kwargs) for (input_channels, output_channels) in self.input_output_list]   
        ])

    def forward(self, x, *args, **kwargs):
        """ 
        function for forward pass of layers 
        """
        # iterate over each block
        for block in self.blocks:
            x = block(x, *args, **kwargs)
            
        return x 

class DGCNLayers(NNLayers):
    """ 
    class to build layers of dgcn blocks specfically. dgcn is unique from other blocks as it requries past layers as its inputs
    """

    def __init__(self, input_channels, block, output_channels, *args, **kwargs):
        """ 
        class constructor for attributes of DGCNLayers 
        """
        # inherit class constructor attributes from nn_layers

        super().__init__(input_channels, block, output_channels, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        """ 
        function for forward pass of layers
        """
        # add layer dimension to initial input
        x = T.unsqueeze(x, 1)

        # iterate over each block
        for block in self.blocks:
            # output y with shape [num_nodes, out_channels]
            y = block(x, *args, **kwargs)
            # add layer dimensions to output and concatenate y to existing x
            x = T.cat((x, T.unsqueeze(y, 1)), 1)

        return x 

# class DGCNActor(nn.Module):
#     """ 
#     class to build actor model for mappo dgcn 
#     """
    
#     def __init__(self, num_agents, obs_dims, dgcn_output_dims, somu_lstm_hidden_size, num_somu_lstm, scmu_lstm_hidden_size, num_scmu_lstm, somu_multi_att_num_heads, scmu_multi_att_num_heads, actor_fc_output_dims, 
#                  actor_fc_dropout_p, softmax_actions_dims, softmax_actions_dropout_p, *args, **kwargs):
#         """ 
#         class constructor for attributes for the actor model 
#         """
#         # inherit class constructor attributes from nn.Module
#         super().__init__()

#         # number of agents
#         self.num_agents = num_agents
#         # number of rollouts
#         self.n_rollout_threads = n_rollout_threads

#         # model architecture for mappo dgcn actor

#         # dgcn layers
#         self.dgcn_layers = DGCNLayers(input_channels=obs_dims, block=DGCNBlock, output_channels=dgcn_output_dims, activation_func="relu", weight_initialisation="default")

#         # list of lstms for self observation memory unit (somu) for each agent
#         # somu_lstm_input_size is the dimension of the observations
#         self.somu_lstm_list = [T.nn.ModuleList([T.nn.LSTM(input_size=obs_dims, hidden_size=somu_lstm_hidden_size, num_layers=1, batch_first=True, dropout=0) for _ in range(num_somu_lstm)]) for _ in range(num_agents)]

#         # list of lstms for self communication memory unit (scmu) for each agent
#         # somu_lstm_input_size is the last layer of dgcn layer
#         self.scmu_lstm_list = [T.nn.ModuleList([T.nn.LSTM(input_size=dgcn_output_dims[-1], hidden_size=scmu_lstm_hidden_size, num_layers=1, batch_first=True, dropout=0) for _ in range(num_scmu_lstm)]) for _ in range(num_agents)]

#         # weights to generate query, key and value for somu and scmu for each agent
#         self.somu_query_layer_list = [MLPBlock(input_shape=somu_lstm_hidden_size, output_shape=somu_lstm_hidden_size, activation_func="relu", dropout_p= 0, weight_initialisation="default") for _ in range(num_agents)]  
#         self.somu_key_layer_list = [MLPBlock(input_shape=somu_lstm_hidden_size, output_shape=somu_lstm_hidden_size, activation_func="relu", dropout_p=0, weight_initialisation="default") for _ in range(num_agents)] 
#         self.somu_value_layer_list = [MLPBlock(input_shape=somu_lstm_hidden_size, output_shape=somu_lstm_hidden_size, activation_func="relu", dropout_p=0, weight_initialisation="default") for _ in range(num_agents)] 
#         self.scmu_query_layer_list = [MLPBlock(input_shape=scmu_lstm_hidden_size, output_shape=scmu_lstm_hidden_size, activation_func="relu", dropout_p=0, weight_initialisation="default") for _ in range(num_agents)] 
#         self.scmu_key_layer_list = [MLPBlock(input_shape=scmu_lstm_hidden_size, output_shape=scmu_lstm_hidden_size, activation_func="relu", dropout_p=0, weight_initialisation="default") for _ in range(num_agents)] 
#         self.scmu_value_layer_list = [MLPBlock(input_shape=scmu_lstm_hidden_size, output_shape=scmu_lstm_hidden_size, activation_func="relu", dropout_p=0, weight_initialisation="default") for _ in range(num_agents)] 

#         # multi-head self attention layer for somu and scmu to selectively choose between the lstms outputs
#         self.somu_multi_att_layer_list = [T.nn.MultiheadAttention(embed_dim=somu_lstm_hidden_size, num_heads=somu_multi_att_num_heads, dropout=0, batch_first=True) for _ in range(num_agents)]
#         self.scmu_multi_att_layer_list = [T.nn.MultiheadAttention(embed_dim=scmu_lstm_hidden_size, num_heads=scmu_multi_att_num_heads, dropout=0, batch_first=True) for _ in range(num_agents)]

#         # hidden fc layers for to generate actions
#         # input channels are observations + concatenated outputs of somu_multi_att_layer and scmu_multi_att_layer and last layer of dgcn
#         # fc_output_dims is the list of sizes of output channels fc_block
#         self.actor_fc_layers = NNLayers(input_channels=obs_dims + num_somu_lstm * somu_lstm_hidden_size + num_scmu_lstm * scmu_lstm_hidden_size + dgcn_output_dims[-1], block=MLPBlock, output_channels=actor_fc_output_dims, 
#                                         activation_func='relu', dropout_p=0, weight_initialisation="default")

#         # final action layer
#         self.act = ACTLayer(action_space, actor_fc_output_dims[-1], self._use_orthogonal, self._gain)

#         # final fc_blocks for actions with log softmax activation function
#         self.softmax_actions_layer = fc_block(input_shape = actor_fc_output_dims[-1], output_shape = softmax_actions_dims, activation_func = "softmax", dropout_p = softmax_actions_dropout_p, 
#                                               weight_initialisation = "default")
    
#     def forward(self, obs, knn=False):
#         """ 
#         function for forward pass through actor model 
#         """
        
#         # obtain reshape data
#         obs = obs.view(self.n_rollout_threads, self.num_agents, -1)
#         # store actions per env
#         actions_list = []

#         # iterate over number of env rollouts
#         for i in range(self.n_rollout_threads):
#             if knn:
#                 raise NotImplementedError
#             else:
#                 # obtain edge index
#                 edge_index = complete_graph_edge_index(self.num_agents) 
#                 edge_index = T.tensor(edge_index, dtype = T.long).t().contiguous()

#             # observation per env (shape: [num_agents, obs_dims])
#             obs_env = obs[i]
#             # obs_env --> dgcn_layers (shape: [num_agents, num_layers, dgcn_output_dims[-1]])
#             dgcn_output = self.dgcn_layers(obs_env, edge_index)

#             # empty list to store ouputs for somu and scmu (shape: [num_agents, num_somu_lstm / num_scmu_lstm, somu_lstm_hidden_size / scmu_lstm_hidden_size])
#             somu_output_list = [[] for _ in range(self.num_agents)]
#             scmu_output_list = [[] for _ in range(self.num_agents)]

#             # iterate over agents 
#             for j in range(self.num_agents):
#                 # iterate over each somu_lstm in somu_lstm_list
#                 for k in range(self.num_somu_lstm):
#                     # observation per env per agent (shape: [1, obs_dims])
#                     obs_env_agent = T.unsqueeze(obs_env[j], dim=0)
#                     # obs_env_agent --> somu_lstm (shape: [1, somu_lstm_hidden_size])
#                     somu_output_list[j].append(self.somu_lstm_list[k](obs_env_agent))
#                 # iterate over each somu lstm in scmu_lstm_list
#                 for k in range(self.num_scmu_lstm):
#                     # last layer of dgcn_output per agent (shape: [1, dgcn_output_dims[-1]])
#                     dgcn_output_agent = T.unsqueeze(dgcn_output[j, -1, :], dim=0)
#                     # dgcn_output_agent --> scmu_lstm (shape: [1, scmu_lstm_hidden_size])
#                     scmu_output_list[j].append(self.scmu_lstm_list[k](dgcn_output_agent))

#                 # concatenate lstm ouput based on number of lstms (shape: [num_somu_lstm / num_scmu_lstm, somu_lstm_hidden_size / scmu_lstm_hidden_size])
#                 somu_output_list[j] = T.cat(somu_output_list[j], dim=0)
#                 scmu_output_list[j] = T.cat(scmu_output_list[j], dim=0)

#                 # obtain query, key and value for somu_lstm and scmu_lstm_outputs (shape: [num_somu_lstm / num_scmu_lstm, somu_lstm_hidden_size / scmu_lstm_hidden_size])
#                 q_somu = self.somu_query_layer_list[i](somu_output_list[j])
#                 k_somu = self.somu_key_layer_list[i](somu_output_list[j])
#                 v_somu = self.somu_value_layer_list[i](somu_output_list[j])
#                 q_scmu = self.scmu_query_layer_list[i](scmu_output_list[j])
#                 k_scmu = self.scmu_key_layer_list[i](scmu_output_list[j])
#                 v_scmu = self.scmu_value_layer_list[i](scmu_output_list[j])

#                 # q, k, v --> multihead attention (shape: [num_somu_lstm / num_scmu_lstm, somu_lstm_hidden_size / scmu_lstm_hidden_size])
#                 somu_output_list[j] = self.somu_multi_att_layer_list[i](q_somu, k_somu, v_somu) 
#                 scmu_output_list[j] = self.somu_multi_att_layer_list[i](q_scmu, k_scmu, v_scmu)

#             # concatenate outputs from lstms (shape: [num_agents, (num_somu_lstm / num_scmu_lstm) * (somu_lstm_hidden_size / scmu_lstm_hidden_size)])
#             somu_output = T.cat(somu_output_list, dim=0).view(self.num_agents, -1)
#             scmu_output = T.cat(scmu_output_list, dim=0).view(self.num_agents, -1)

#             # concatenate outputs from dgcn, somu and scmu (shape: [num_agents, obs_dims + num_somu_lstm * somu_lstm_hidden_size + num_scmu_lstm * scmu_lstm_hidden_size + dgcn_output_dims[-1]])
#             output = T.cat((obs_env, dgcn_output[:, -1, :], somu_output, scmu_output), dim=-1)
        
#             # output --> actor_fc_layers (shape: [num_agents, actor_fc_output_dims[-1]])
#             output = self.actor_fc_layers(output)

#             # actor_fc_layers --> act (shape: [num_agents, action_space_dim])
#             actions_list.append(self.act(output))

#         # (shape: [n_rollout_threads * num_agents, action_space_dim])
#         return T.cat(actions_list, dim=0)

# class DGCNCritic(nn.Module):
#     """ 
#     class to build model for critic model for mappo dgcn 
#     """
    
#     def __init__(self, gatv2_input_dims, gatv2_output_dims, gatv2_num_heads, gatv2_dropout_p, gatv2_bool_concat, gmt_hidden_dims, gmt_output_dims, critic_fc_output_dims, critic_fc_dropout_p, *args, **kwargs):
#         """ 
#         class constructor for attributes for the model 
#         """
#         # inherit class constructor attributes from nn.Module
#         super().__init__()
            
#         # model architecture for mappo critic
            
#         # gatv2 layers for state inputs 
#         # gnn_input_dims are the dimensions of the initial node embeddings 
#         # gnn_output_dims are the list of dimensions of the the output embeddings of each layer of gatv2 
#         self.critic_state_gatv2_layer = NNLayers(input_channels=gatv2_input_dims, block=GATv2Block, output_channels=gatv2_output_dims, num_heads=1, concat=True, activation_func='relu', dropout_p=0, 
#                                                  weight_initialisation = "default")
        
#         # graph multiset transformer (gmt) for state inputs
#         # in_channels are the dimensions of node embeddings after gatv2 layers
#         # gmt_hidden_dims are the dimensions of the node embeddings post 1 initial linear layer in gmt 
#         # gmt_output_dims are the dimensions of the sole remaining node embedding that represents the entire graph
#         # uses GATv2Conv as Conv block for GMPool_G
#         # remaining inputs are defaults 
#         self.critic_state_gmt_layer = gnn.GraphMultisetTransformer(in_channels=gatv2_output_dims[-1], hidden_channels=gmt_hidden_dims, out_channels=gmt_output_dims, Conv=gnn.GATv2Conv, num_nodes=300, 
#                                                                    pooling_ratio=0.25, pool_sequences=['GMPool_G', 'SelfAtt', 'GMPool_I'], num_heads=4, layer_norm=False)

#         # hidden fc layers post gmt layer
#         # input channels are the dimensions of node embeddings of the one node from gmt
#         # fc_output_dims is the list of sizes of output channels fc_block
#         self.critic_fc_layers = nn_layers(input_channels=gmt_output_dims, block=MLPBlock, output_channels=fc_output_dims, activation_func='relu', dropout_p=0, weight_initialisation = "default")

#         # final layer using popart for value normalisation
#         self.popart = popart(input_shape=fc_output_dims[-1], output_shape=1)
    
#     def forward(self, cent_obs_graph_data, batch):
#         """ 
#         function for forward pass through critic model 
#         """
#         # obtain node embeddings and edge index from data
#         x, edge_index = cent_obs_graph_data.x, cent_obs_graph_data.edge_index
       
#         # x (graph of critic's state representation) --> critic_state_gatv2_layer
#         x = self.critic_state_gatv2_layer(x = x, edge_index = edge_index)
        
#         # critic_state_gatv2_layer --> critic_state_gmt_layer
#         x = self.critic_state_gmt_layer(x = x, edge_index = edge_index, batch = batch)

#         # critic_gmt_layer --> critic_fc_layers
#         x = self.critic_fc_layers(x = x)

#         # critic_fc_layers --> v value
#         v = self.popart(x)
        
#         return v