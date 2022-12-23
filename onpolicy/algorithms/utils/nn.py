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

def weights_initialisation_function_generator(weight_initialisation, 
                                              activation_func, *args, **kwargs):
    """ 
    function that returns functions initialise weights according to specified 
    methods. 
    """
    if weight_initialisation == "xavier_uniform":
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 
                                        gain=nn.init.calculate_gain(activation_func))
        return init_weight
    elif weight_initialisation == "xavier_normal":
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, 
                                       gain=nn.init.calculate_gain(activation_func))
        return init_weight
    elif weight_initialisation == "kaiming_uniform":
        # recommend for relu / leaky relu
        assert (activation_func == "relu" or activation_func == "leaky_relu"), "Non-linearity recommended to be 'relu' or 'leaky_relu'"
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, 
                                         a=kwargs.get("kaiming_a", math.sqrt(5)), 
                                         mode=kwargs.get("kaiming_mode", "fan_in"), 
                                         nonlinearity=activation_func)
        return init_weight
    elif weight_initialisation == "kaiming_normal":
        # recommend for relu / leaky relu
        assert (activation_func == "relu" or activation_func == "leaky_relu"), "Non-linearity recommended to be 'relu' or 'leaky_relu'"
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, 
                                        a=kwargs.get("kaiming_a", math.sqrt(5)), 
                                        mode=kwargs.get("kaiming_mode", "fan_in"), 
                                        nonlinearity=activation_func)
        return init_weight
    elif weight_initialisation == "uniform":
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 
                                 a=kwargs.get("uniform_lower_bound", 0.0), 
                                 b=kwargs.get("uniform_upper_bound", 1.0))
        return init_weight
    elif weight_initialisation == "orthogonal":
        def init_weight(m):
            if isinstance(m, nn.Linear):
                # matrix (orthogonal requires tensor of dim>=2)
                if len(m.weight.shape) >= 2:
                    nn.init.orthogonal_(m.weight, 
                                        gain=nn.init.calculate_gain(activation_func))
                # bias
                else:
                    nn.init.zeros_(m.weight) 
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

    def __init__(self, input_channels, output_channels, att_heads=1, mul_att_heads=1, groups=1, concat=True, negative_slope=0.2, dropout=0.0, add_self_loops=True, edge_dim=None, fill_value='mean', bias=True):
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
                                        (DGCNConv(in_channels=input_channels, out_channels=output_channels, att_heads=att_heads, mul_att_heads=mul_att_heads, groups=groups, concat=concat,  
                                                  negative_slope=negative_slope, dropout=dropout, add_self_loops=add_self_loops, edge_dim=edge_dim, fill_value=fill_value, bias=bias), 
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