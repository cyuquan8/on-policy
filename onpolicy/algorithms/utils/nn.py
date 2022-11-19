# ==========================================================================================================================================================
# neural network (nn) module
# purpose: classes and functions to build a scalable neural network model
# ==========================================================================================================================================================

import os
import shutil
import math

import torch as T
import torch.nn as nn
import torch_geometric.nn as gnn

from functools import partial
from .dgcn import DGCNConv
from utils.popart import popart

def activation_function(activation):
    
    """ function that returns ModuleDict of activation functions """
    
    return  nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['sigmoid', nn.Sigmoid()],
        ['softmax', nn.Softmax(1)],
        ['log_softmax', nn.LogSoftmax(1)],
        ['tanh', nn.Tanh()],
        ['hard_tanh', nn.Hardtanh()],
        ['none', nn.Identity()]
    ])[activation]

def weights_initialisation_function_generator(weight_intialisation, activation_func, *args, **kwargs):

    """ function that returns functions initialise weights according to specified methods """
        
    # check weight initialisation
    if weight_intialisation == "xavier_uniform":
        
        # generate function
        def init_weight(m):
            
            # check if linear
            if isinstance(m, nn.Linear):

                # initialise weight
                nn.init.xavier_uniform_(m.weight, gain = nn.init.calculate_gain(activation_func))
        
        return init_weight

    # check weight initialisation
    elif weight_intialisation == "xavier_normal":
        
        # generate function
        def init_weight(m):
            
            # check if linear
            if isinstance(m, nn.Linear):

                # initialise weight
                nn.init.xavier_normal_(m.weight, gain = nn.init.calculate_gain(activation_func))
        
        return init_weight
    
    # check weight initialisation
    elif weight_intialisation == "kaiming_uniform":
        
        # recommend for relu / leaky relu
        assert (activation_func == "relu" or activation_func == "leaky_relu"), "Non-linearity recommended to be 'relu' or 'leaky_relu'"

        # generate function
        def init_weight(m):
            
            # check if linear
            if isinstance(m, nn.Linear):

                # initialise weight
                nn.init.kaiming_uniform_(m.weight, a = kwargs.get("kaiming_a", math.sqrt(5)), mode = kwargs.get("kaiming_mode", "fan_in"), nonlinearity = activation_func)
        
        return init_weight
    
    # check weight initialisation
    elif weight_intialisation == "kaiming_normal":
        
        # recommend for relu / leaky relu
        assert (activation_func == "relu" or activation_func == "leaky_relu"), "Non-linearity recommended to be 'relu' or 'leaky_relu'"

        # generate function
        def init_weight(m):
            
            # check if linear
            if isinstance(m, nn.Linear):

                # initialise weight
                nn.init.kaiming_normal_(m.weight, a = kwargs.get("kaiming_a", math.sqrt(5)), mode = kwargs.get("kaiming_mode", "fan_in"), nonlinearity = activation_func)

        return init_weight

    # check weight initialisation
    elif weight_intialisation == "uniform":

        # generate function
        def init_weight(m):
            
            # check if linear
            if isinstance(m, nn.Linear):

                # initialise weight
                nn.init.uniform_(m.weight, a = kwargs.get("uniform_lower_bound", 0.0), b = kwargs.get("uniform_upper_bound", 1.0))

        return init_weight

    else:
        
        # generate function
        def init_weight(m):

            pass

        return init_weight

class fc_block(nn.Module):
    
    """ class to build basic fully connected block """
    
    def __init__(self, input_shape, output_shape, activation_func, dropout_p, weight_initialisation, *args, **kwargs):
        
        """ class constructor that creates the layers attributes for fc_block """
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input and output units for hidden layer 
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # activation function for after batch norm
        self.activation_func = activation_func 
        
        # dropout probablity
        self.dropout_p = dropout_p
        
        # basic fc_block. inpuit --> linear --> batch norm --> activation function --> dropout 
        self.block = nn.Sequential(
            
            # linear hidden layer
            nn.Linear(self.input_shape, self.output_shape, bias = False),
            
            # batch norm
            nn.BatchNorm1d(self.output_shape),
            
            # activation func
            activation_function(self.activation_func),
            
            # dropout
            nn.Dropout(self.dropout_p),
            
        )
        
        # weight initialisation
        self.block.apply(weights_initialisation_function_generator(weight_initialisation, activation_func, *args, **kwargs))

    def forward(self, x):
        
        """ function for forward pass of fc_block """
        
        x = self.block(x)
        
        return x

class dgcn_block(nn.Module):

    """ class to build dgcn_block """

    def __init__(self, input_channels, output_channels, activation_func, weight_initialisation, att_heads = 1, mul_att_heads = 1, groups = 1, concat = True, negative_slope = 0.2, dropout = 0.0, add_self_loops = True, 
                 edge_dim = None, fill_value = 'mean', bias = True, *args, **kwargs):

        """ class constructor for attributes of the dgcn_block """

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
                                        (gnn.GraphNorm(self.output_channels * self.num_heads if concat == True else self.output_channels), 'x -> x')

                                    ]
        )

        # weight initialisation
        self.block.apply(weights_initialisation_function_generator(weight_initialisation, activation_func, *args, **kwargs))

    def forward(self, x, edge_index):
        
        """ function for forward pass of gatv2_block """
        
        x = self.block(x, edge_index)
        
        return x

class gatv2_block(nn.Module):

    """ class to build gatv2_block """

    def __init__(self, input_channels, output_channels, num_heads, concat, dropout_p, activation_func, weight_initialisation, *args, **kwargs):

        """ class constructor for attributes of the gatv2_block """

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
        self.dropout_p = dropout_p

        # activation function for after GATv2Conv 
        self.activation_func = activation_func

        # basic gatv2_block. input --> GATv2Conv --> GraphNorm --> activation func
        self.block = gnn.Sequential('x, edge_index', 
                                    [

                                        # GATv2Conv 
                                        (gnn.GATv2Conv(in_channels = self.input_channels, out_channels = self.output_channels, heads = self.num_heads, concat = concat, dropout = dropout_p), 
                                         'x, edge_index -> x'), 

                                        # graph norm
                                        (gnn.GraphNorm(self.output_channels * self.num_heads if concat == True else self.output_channels), 'x -> x'),

                                        # activation func
                                        activation_function(self.activation_func)

                                    ]
        )

        # weight initialisation
        self.block.apply(weights_initialisation_function_generator(weight_initialisation, activation_func, *args, **kwargs))

    def forward(self, x, edge_index):
        
        """ function for forward pass of gatv2_block """
        
        x = self.block(x, edge_index)
        
        return x

class nn_layers(nn.Module):
    
    """ class to build layers of blocks (e.g. fc_block) """
    
    def __init__(self, input_channels, block, output_channels, *args, **kwargs):
        
        """ class constructor for attributes of nn_layers """

        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input channels/shape
        self.input_channels = input_channels
        
        # class of block
        self.block = block
        print(output_channels)
        # output channels/shape
        self.output_channels = output_channels
        self.input_output_list = list(zip(output_channels[:], output_channels[1:]))
        
        # module list of layers with same args and kwargs
        self.blocks = nn.ModuleList([
            
            self.block(self.input_channels, self.output_channels[0], *args, **kwargs),
            *[self.block(input_channels, output_channels, *args, **kwargs) for (input_channels, output_channels) in self.input_output_list]   
            
        ])

    def forward(self, x, *args, **kwargs):
        
        """ function for forward pass of layers """
        
        # iterate over each block
        for block in self.blocks:
            
            x = block(x, *args, **kwargs)
            
        return x 

class dgcn_layers(nn_layers):

    """ class to build layers of dgcn blocks specfically """
    """ dgcn is unique from other blocks as it requries past layers as its inputs """

    def __init__(self, input_channels, block, output_channels, *args, **kwargs):

        """ class constructor for attributes of dgcn_layers """

        # inherit class constructor attributes from nn_layers
        super().__init__(self, input_channels, block, output_channels, *args, **kwargs)

    def forward(self, x, *args, **kwargs):

        """ function for forward pass of layers """

        # iterate over each block
        for block in self.blocks:

            # output y with shape [num_nodes, out_channels]
            y = block(x, *args, **kwargs)

            # add layer dimensions to output and concatenate y to existing x
            x = T.cat((x, T.unsqueeze(y, -2)), -2)

        return x 

class mappo_dgcn_actor_model(nn.Module):
    
    """ class to build actor model for mappo dgcn """
    
    def __init__(self, model, model_name, mode, scenario_name, training_name, learning_rate, optimizer, lr_scheduler, obs_dims, dgcn_output_dims, somu_lstm_hidden_size, somu_lstm_num_layers, somu_lstm_dropout, num_somu_lstm, 
                 scmu_lstm_hidden_size, scmu_lstm_num_layers, scmu_lstm_dropout, num_scmu_lstm, somu_multi_att_num_heads, somu_multi_att_dropout, scmu_multi_att_num_heads, scmu_multi_att_dropout, actor_fc_output_dims, 
                 actor_fc_dropout_p, log_softmax_actions_dims, log_softmax_actions_dropout_p, *args, **kwargs):
        
        """ class constructor for attributes for the actor model """
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # model
        self.model = model
        
        # model name
        self.model_name = model_name
        
        # checkpoint filepath 
        self.checkpoint_path = None
        
        # if training model
        if mode != 'test' and mode != 'load_and_train':

            try:
                
                # create directory for saving models if it does not exist
                os.mkdir("saved_models/" + scenario_name + "/" + training_name + "_" + "best_models/")
                
            except:
                
                # remove existing directory and create new directory
                shutil.rmtree("saved_models/" + scenario_name + "/" + training_name + "_" + "best_models/")
                os.mkdir("saved_models/" + scenario_name + "/" + training_name + "_" + "best_models/")

        # checkpoint directory
        self.checkpoint_dir = "saved_models/" + scenario_name + "/" + training_name + "_" + "best_models/"
        
        # learning rate
        self.learning_rate = learning_rate

        # number of somu and scmu lstms
        self.num_somu_lstm = num_somu_lstm
        self.num_scmu_lstm = num_scmu_lstm

        # model architecture for mappo dgcn actor

        # dgcn layers
        self.dgcn_layers = dgcn_layers(input_channels = obs_dims, block = dgcn_block, output_channels = dgcn_output_dims, activation_func = "relu", weight_initialisation = "default")

        # list of lstms for self observation memory unit (somu)
        # somu_lstm_input_size is the dimension of the observations
        # outputs with size of (batch size, sequence length = 1, hidden_size)
        self.somu_lstm_list = [T.nn.LSTM(input_size = obs_dims, hidden_size = somu_lstm_hidden_size, num_layers = somu_lstm_num_layers, batch_first = True, dropout = somu_lstm_dropout) for _ in range(num_somu_lstm)]

        # list of lstms for self communication memory unit (scmu)
        # somu_lstm_input_size is the last layer of dgcn layer
        # outputs with size of (batch size, sequence length = 1, hidden_size)
        self.scmu_lstm_list = [T.nn.LSTM(input_size = dgcn_output_dims[-1], hidden_size = scmu_lstm_hidden_size, num_layers = scmu_lstm_num_layers, batch_first = True, dropout = scmu_lstm_dropout) for _ in range(num_scmu_lstm)]

        # weights to generate query, key and value for somu and scmu
        # outputs with size of (batch size, hidden_size)
        self.somu_query_layer = fc_block(input_shape = somu_lstm_hidden_size, output_shape = somu_lstm_hidden_size, activation_func = "relu", dropout_p =  0, weight_initialisation = "default") 
        self.somu_key_layer = fc_block(input_shape = somu_lstm_hidden_size, output_shape = somu_lstm_hidden_size, activation_func = "relu", dropout_p =  0, weight_initialisation = "default") 
        self.somu_value_layer = fc_block(input_shape = somu_lstm_hidden_size, output_shape = somu_lstm_hidden_size, activation_func = "relu", dropout_p =  0, weight_initialisation = "default") 
        self.scmu_query_layer = fc_block(input_shape = scmu_lstm_hidden_size, output_shape = scmu_lstm_hidden_size, activation_func = "relu", dropout_p =  0, weight_initialisation = "default") 
        self.scmu_key_layer = fc_block(input_shape = scmu_lstm_hidden_size, output_shape = scmu_lstm_hidden_size, activation_func = "relu", dropout_p =  0, weight_initialisation = "default") 
        self.scmu_value_layer = fc_block(input_shape = scmu_lstm_hidden_size, output_shape = scmu_lstm_hidden_size, activation_func = "relu", dropout_p =  0, weight_initialisation = "default") 

        # multi-head self attention layer for somu and scmu to selectively choose between the lstms outputs
        # outputs with size of (batch size, target sequence length = num_somu_lstm / num_scmu_lstm, hidden_size) 
        self.somu_multi_att_layer = T.nn.MultiheadAttention(embed_dim = somu_lstm_hidden_size, num_heads = somu_multi_att_num_heads, dropout = somu_multi_att_dropout, batch_first = True)
        self.scmu_multi_att_layer = T.nn.MultiheadAttention(embed_dim = scmu_lstm_hidden_size, num_heads = scmu_multi_att_num_heads, dropout = scmu_multi_att_dropout, batch_first = True)

        # hidden fc layers for to generate actions
        # input channels are observations + concatenated outputs of somu_multi_att_layer and scmu_multi_att_layer and last layer of dgcn
        # fc_output_dims is the list of sizes of output channels fc_block
        self.actor_fc_layers = nn_layers(input_channels = obs_dims + num_somu_lstm * somu_lstm_hidden_size + num_scmu_lstm * scmu_lstm_hidden_size + dgcn_output_dims[-1], block = fc_block, output_channels = actor_fc_output_dims, 
                                         activation_func = 'relu', dropout_p = actor_fc_dropout_p, weight_initialisation = "default")

        # final fc_blocks for actions with log softmax activation function
        self.softmax_actions_layer = fc_block(input_shape = actor_fc_output_dims[-1], output_shape = softmax_actions_dims, activation_func = "softmax", dropout_p = softmax_actions_dropout_p, 
                                              weight_initialisation = "default")
             
        # check optimizer
        if optimizer == "adam":

            # adam optimizer 
            self.optimizer = T.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        # check learning rate scheduler
        if lr_scheduler == "cosine_annealing_with_warm_restarts":

            self.scheduler = T.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer = self.optimizer, T_0 = kwargs.get('actor_lr_scheduler_T_0', 1000), eta_min = kwargs.get('actor_lr_scheduler_eta_min', 0))
        
        # device for training (cpu/gpu)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        # cast module to device
        self.to(self.device)
    
    def forward(self, x):
            
        """ function for forward pass through actor model """
        
        # x (obs) --> dgcn_layers
        dgcn_output = dgcn_layers(x)

        # empty list to store ouputs for somu_lstm and scmu_lstms
        somu_lstm_output_list = []
        scmu_lstm_output_list = []

        # iterate over each somu_lstm in somu_lstm_list
        for i in range(self.num_somu_lstm):

            # x (obs) --> somu_lstm
            somu_lstm_output_list.append(self.somu_lstm_list[i](x))

        # iterate over each somu lstm in scmu_lstm_list
        for i in range(self.num_scmu_lstm):

            # last layer of dgcn_output --> scmu_lstm
            scmu_lstm_output_list.append(self.scmu_lstm_list[i](dgcn_output[:, -1, :]))

        # concatenate x, outputs from dgcn, somu and scmu
        output = T.squeeze(T.cat((T.unsqueeze(x, dim = -2), dgcn_output[:, -1, :], T.cat(somu_lstm_output_list, dim = -1), T.cat(scmu_lstm_output_list, dim = -1)), dim = -1), dim = -2) 

        # output --> actor_fc_layers
        output = self.actor_fc_layers(output)

        # actor_fc_layers --> softmax_actions_layer
        softmax_actions = self.softmax_actions_layer(output)

        return softmax_actions

class mappo_dgcn_critic_model(nn.Module):
    
    """ class to build model for critic model for mappo dgcn """
    
    def __init__(self, model, model_name, mode, scenario_name, training_name, learning_rate, optimizer, lr_scheduler, gatv2_input_dims, gatv2_output_dims, gatv2_num_heads, gatv2_dropout_p, gatv2_bool_concat, gmt_hidden_dims, 
                 gmt_output_dims, critic_fc_output_dims, critic_fc_dropout_p, *args, **kwargs):
        
        """ class constructor for attributes for the model """
        
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # model
        self.model = model
        
        # model name
        self.model_name = model_name
        
        # checkpoint filepath 
        self.checkpoint_path = None
        
        # if training model
        if mode != 'test' and mode != 'load_and_train':

            try:
                                # create directory for saving models if it does not exist
                os.mkdir("saved_models/" + scenario_name + "/" + training_name + "_" + "best_models/")
                
            except:
                
                # remove existing directory and create new directory
                shutil.rmtree("saved_models/" + scenario_name + "/" + training_name + "_" + "best_models/")
                os.mkdir("saved_models/" + scenario_name + "/" + training_name + "_" + "best_models/")

        # checkpoint directory
        self.checkpoint_dir = "saved_models/" + scenario_name + "/" + training_name + "_" + "best_models/"
        
        # learning rate
        self.learning_rate = learning_rate
            
        # model architecture for mappo critic
            
        # gatv2 layers for state inputs 
        # gnn_input_dims are the dimensions of the initial node embeddings 
        # gnn_output_dims are the list of dimensions of the the output embeddings of each layer of gatv2 
        self.critic_state_gatv2_layer = nn_layers(input_channels = gatv2_input_dims, block = gatv2_block, output_channels = gatv2_output_dims, num_heads = gatv2_num_heads, concat = gatv2_bool_concat, 
                                                  activation_func = 'relu', dropout_p = gatv2_dropout_p, weight_initialisation = "default")
        
        # graph multiset transformer (gmt) for state inputs
        # in_channels are the dimensions of node embeddings after gatv2 layers
        # gmt_hidden_dims are the dimensions of the node embeddings post 1 initial linear layer in gmt 
        # gmt_output_dims are the dimensions of the sole remaining node embedding that represents the entire graph
        # uses GATv2Conv as Conv block for GMPool_G
        # remaining inputs are defaults 
        self.critic_state_gmt_layer = gnn.GraphMultisetTransformer(in_channels = gatv2_output_dims[-1] * gatv2_num_heads if gatv2_bool_concat == True else gatv2_output_dims[-1], hidden_channels = gmt_hidden_dims, 
                                                                   out_channels = gmt_output_dims , Conv = gnn.GATv2Conv, num_nodes = 300, pooling_ratio = 0.25, 
                                                                   pool_sequences = ['GMPool_G', 'SelfAtt', 'GMPool_I'], num_heads = 4, layer_norm = False)

        # hidden fc layers post gmt layer
        # input channels are the dimensions of node embeddings of the one node from gmt
        # fc_output_dims is the list of sizes of output channels fc_block
        self.critic_fc_layers = nn_layers(input_channels = gmt_output_dims, block = fc_block, output_channels = fc_output_dims, activation_func = 'relu', dropout_p = dropout_p, weight_initialisation = "default")

        # final layer using popart for value normalisation
        self.popart = popart(input_shape = fc_output_dims[-1], output_shape = 1)
            
        # check optimizer
        if optimizer == "adam":

            # adam optimizer 
            self.optimizer = T.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        # check learning rate scheduler
        if lr_scheduler == "cosine_annealing_with_warm_restarts":

            self.scheduler = T.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer = self.optimizer, T_0 = kwargs.get('critic_lr_scheduler_T_0', 1000), eta_min = kwargs.get('critic_lr_scheduler_eta_min', 0))
        
        # device for training (cpu/gpu)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        # cast module to device
        self.to(self.device)
    
    def forward(self, data, batch):
            
        """ function for forward pass through critic model """
        
        # obtain node embeddings and edge index from data
        x, edge_index = data.x, data.edge_index
       
        # x (graph of critic's state representation) --> critic_state_gatv2_layer
        x = self.critic_state_gatv2_layer(x = x, edge_index = edge_index)
        
        # critic_state_gatv2_layer --> critic_state_gmt_layer
        x = self.critic_state_gmt_layer(x = x, edge_index = edge_index, batch = batch)

        # critic_gmt_layer --> critic_fc_layers
        x = self.critic_fc_layers(x = x)

        # critic_fc_layers --> v value
        v = self.popart(x)
        
        return v