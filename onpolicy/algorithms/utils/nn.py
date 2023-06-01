import torch as T
import torch.nn as nn
import torch_geometric.nn as gnn

from onpolicy.algorithms.utils.dna_gatv2_conv import DNAGATv2Conv
from onpolicy.algorithms.utils.gatv2_conv import GATv2Conv

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
        assert (activation_func == "relu" or activation_func == "leaky_relu"), \
            "Non-linearity recommended to be 'relu' or 'leaky_relu'"
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, 
                                         a=kwargs.get("kaiming_a", math.sqrt(5)), 
                                         mode=kwargs.get("kaiming_mode", "fan_in"), 
                                         nonlinearity=activation_func)
        return init_weight
    elif weight_initialisation == "kaiming_normal":
        # recommend for relu / leaky relu
        assert (activation_func == "relu" or activation_func == "leaky_relu"), \
            "Non-linearity recommended to be 'relu' or 'leaky_relu'"
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

class Conv2DAutoPadding(nn.Conv2d):
    """
    class to set padding dynamically based on kernel size to preserve dimensions of height and width after conv
    """
    
    def __init__(self, *args, **kwargs):
        """ 
        class constructor for conv_2d_auto_padding to alter padding attributes of nn.Conv2d 
        """
        # inherit class constructor attributes from nn.Conv2d
        super().__init__(*args, **kwargs)
        
        # dynamically adds padding based on the kernel_size
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) 

class MLPBlock(nn.Module):
    """
    class to build basic fully connected block 
    """
    block_type = "MLPBlock"

    def __init__(self, input_shape, output_shape, activation_func="relu", weight_initialisation="default", *args, 
        **kwargs):
        """
        class constructor that creates the layers attributes for MLPBlock 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input and output units for hidden layer 
        self.input_shape = input_shape
        self.output_shape = output_shape
        # activation function
        self.activation_func = activation_func 
        # dropout probablity
        self.dropout_p = kwargs.get('dropout', 0)

        # basic fc_block. inpuit --> linear --> batch norm --> activation function --> dropout 
        self.block = nn.Sequential(
            # linear hidden layer
            nn.Linear(self.input_shape, self.output_shape, bias=False),
            # layer norm
            nn.LayerNorm(self.output_shape),
            # activation func
            activation_function(self.activation_func),
            # dropout
            nn.Dropout(self.dropout_p)
        )
        
        # weight initialisation
        self.block.apply(weights_initialisation_function_generator(weight_initialisation, activation_func, *args, 
                                                                   *kwargs))

    def forward(self, x):
        """ 
        function for forward pass of MLPBlock 
        """
        x = self.block(x)
        return x

class VGGBlock(nn.Module):
    """ 
    class to build basic, vgg inspired block 
    """ 
    block_type = "VGGBlock"

    def __init__(self, input_channels, output_channels, activation_func, conv, dropout_p, max_pool_kernel):
        """ 
        class constructor that creates the layers attributes for VGGBlock 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input and output channels for conv (num of filters)
        self.input_channels = input_channels
        self.output_channels = output_channels
        # activation function for after batch norm
        self.activation_func = activation_func
        # class of conv
        self.conv = conv
        # dropout probablity
        self.dropout_p = dropout_p
        # size of kernel for maxpooling
        self.max_pool_kernel = max_pool_kernel
        
        # basic vgg_block. input --> conv --> batch norm --> activation func --> dropout --> max pool
        self.block = nn.Sequential(
            # conv
            self.conv(self.input_channels, self.output_channels),
            # batch norm
            nn.BatchNorm2d(self.output_channels),
            # activation func
            activation_function(self.activation_func),
            # dropout
            nn.Dropout2d(self.dropout_p),
            # maxpooling
            # ceil mode to True to ensure take care of odd dimensions accounted for
            nn.MaxPool2d(self.max_pool_kernel, ceil_mode=True)
        )

    def forward(self, x):
        """ 
        function for forward pass of VGGBlock 
        """
        x = self.block(x)
        return x

class DNAGATv2Block(nn.Module):
    """ 
    class to build DNAGATv2Block 
    """
    block_type = "DNAGATv2Block"

    def __init__(
            self, 
            input_channels, 
            output_channels, 
            att_heads=1, 
            mul_att_heads=1, 
            groups=1, 
            negative_slope=0.2, 
            dropout=0.0, 
            add_self_loops=True, 
            edge_dim=None, 
            fill_value='mean', 
            bias=True,
            gnn_cpa_model='none'
        ):
        """ 
        class constructor for attributes of the DNAGATv2Block 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()

        # input and output channels for DNAGATv2Conv 
        self.input_channels = input_channels
        self.output_channels = output_channels
        # number of heads for gatv2 and multi head attention
        self.att_heads = att_heads
        self.mul_att_heads = mul_att_heads
        # number of groups for grouped operations multi head attention
        self.groups = groups
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
        # cardinality preserved attention (cpa) model
        self.gnn_cpa_model = gnn_cpa_model

        # basic DNAGATv2Block. input --> DNAGATv2Conv --> graph norm
        self.block = gnn.Sequential('x, edge_index', 
                                    [
                                        # DNAGATv2Conv block 
                                        (DNAGATv2Conv(in_channels=input_channels, out_channels=output_channels, 
                                                      att_heads=att_heads, mul_att_heads=mul_att_heads, groups=groups, 
                                                      negative_slope=negative_slope, dropout=dropout, 
                                                      add_self_loops=add_self_loops, edge_dim=edge_dim, 
                                                      fill_value=fill_value, bias=bias, gnn_cpa_model=gnn_cpa_model), 
                                         'x, edge_index -> x'), 
                                        # graph norm
                                        (gnn.GraphNorm(self.output_channels), 'x -> x')
                                    ]
        )  

    def forward(self, x, edge_index, *args, **kwargs):
        """ 
        function for forward pass of DNAGATv2Block 
        """
        x = self.block(x, edge_index, *args, **kwargs)
        return x

class GATv2Block(nn.Module):
    """ 
    class to build GATv2Block 
    """
    block_type = "GATv2Block"

    def __init__(
            self, 
            input_channels,
            output_channels,
            heads=1,
            concat=True,
            negative_slope=0.2,
            dropout=0.0,
            add_self_loops=True,
            edge_dim=None,
            fill_value='mean',
            bias=True,
            share_weights=False,
            gnn_cpa_model='none',
            **kwargs,
        ):
        """ 
        class constructor for attributes of the GATv2Block 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()

        # input and output channels for DNAGATv2Conv 
        self.input_channels = input_channels
        self.output_channels = output_channels
        # number of heads for gatv2
        self.heads = heads
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
        # whether to use same matrix for target and source nodes
        self.share_weights = share_weights
        # cardinality preserved attention (cpa) model
        self.gnn_cpa_model = gnn_cpa_model

        # basic GATv2Block. input --> GATv2Conv --> graph norm
        self.block = gnn.Sequential('x, edge_index', 
                                    [
                                        # GATv2Conv block 
                                        (GATv2Conv(in_channels=input_channels, out_channels=output_channels, 
                                                   heads=heads, concat=concat, negative_slope=negative_slope, 
                                                   dropout=dropout, add_self_loops=add_self_loops, edge_dim=edge_dim, 
                                                   fill_value=fill_value, bias=bias, share_weights=share_weights, 
                                                   gnn_cpa_model=gnn_cpa_model), 
                                         'x, edge_index -> x'), 
                                        # graph norm
                                        (gnn.GraphNorm(self.output_channels * self.heads \
                                                       if concat == True else self.output_channels), 'x -> x')
                                    ]
        )  

    def forward(self, x, edge_index, *args, **kwargs):
        """ 
        function for forward pass of DNAGATv2Block 
        """
        x = self.block(x, edge_index, *args, **kwargs)
        return x

class GINBlock(nn.Module):
    """ 
    class to build GINBlock 
    """
    block_type = "GINBlock"

    def __init__(self, input_channels, output_channels, n_gin_fc_layers, activation_func="relu", eps=0.0, 
                 train_eps=False, edge_dim=None):
        """ 
        class constructor for attributes of the GINBlock 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()

        # input and output channels for GINConv
        self.input_channels = input_channels
        self.output_channels = output_channels
        # number of layers of MLP of GINConv
        self.n_gin_fc_layers = n_gin_fc_layers
        # activation function
        self.activation_func = activation_func 
        # epsilon
        self.eps = eps
        # boolean to track if eps is trainable parameter
        self.train_eps = train_eps
        # dimension of edge attributes if any
        self.edge_dim = edge_dim

        # list to store modules for Sequential
        modules = []
        # generate modules
        for i in range(n_gin_fc_layers):
            # first layer
            if i == 0:
                # linear hidden layer
                modules.append(nn.Linear(self.input_channels, self.output_channels, bias=False))   
            # subsequent layers
            else:
                # linear hidden layer
                modules.append(nn.Linear(self.output_channels, self.output_channels, bias=False))
            # layer norm
            modules.append(nn.LayerNorm(self.output_channels))
            # activation function
            modules.append(activation_function(self.activation_func))
        # generate MLP
        self.nn = nn.Sequential(*modules)

        # basic dgcn_block. input --> gin --> graph norm
        self.block = gnn.Sequential('x, edge_index', 
                                    [
                                        # gine block 
                                        (gnn.GINConv(nn=self.nn, eps=self.eps, train_eps=self.train_eps, 
                                                     edge_dim=self.edge_dim), 
                                         'x, edge_index -> x'), 
                                        # graph norm
                                        (gnn.GraphNorm(self.output_channels), 'x -> x')
                                    ]
        )  

    def forward(self, x, edge_index, *args, **kwargs):
        """ 
        function for forward pass of GINBlock 
        """
        x = self.block(x, edge_index, *args, **kwargs)
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
        # input and output channels/shape for each block
        self.input_output_list = list(zip(output_channels[:], output_channels[1:]))
        # module list of layers with same args and kwargs
        self.blocks = nn.ModuleList([
            self.block(self.input_channels, self.output_channels[0], *args, **kwargs),
            *[self.block(input_channels, output_channels, *args, **kwargs)\
            for (input_channels, output_channels) in self.input_output_list]   
        ])

    def forward(self, x, *args, **kwargs):
        """ 
        function for forward pass of layers 
        """
        # iterate over each block
        for block in self.blocks:
            x = block(x, *args, **kwargs)
        return x 

    def get_flat_output_shape(self, input_shape):
        """ 
        function to obtain number of features after flattening after convolution layers 
        """
        # initialise dummy tensor of ones with input shape
        x = T.ones(1, *input_shape)
        # feed dummy tensor to blocks by iterating over each block
        for block in self.blocks:
            x = block(x)
        # flatten resulting tensor and obtain number of features
        n_size = x.view(1, -1).size(1)
        return n_size

class DNAGATv2Layers(NNLayers):
    """ 
    class to build layers of DNAGATv2Conv blocks specfically. 
    DNAGATv2Conv is unique from other blocks as it requries past layers as its inputs
    """

    def __init__(self, input_channels, block, output_channels, *args, **kwargs):
        """ 
        class constructor for attributes of DNAGATv2Layers 
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

class GNNAllLayers(NNLayers):
    """ 
    class to build layers of GNN blocks to include all layers
    """
    def __init__(self, input_channels, block, output_channels, *args, **kwargs):
        """ 
        class constructor for attributes of GNNAllLayers 
        """
        # inherit class constructor attributes from nn_layers
        super().__init__(input_channels, block, output_channels, *args, **kwargs)

        # update GATv2Block accounting for concatenation or averaging of attention heads
        if self.block.block_type == 'GATv2Block':
            # obtain relevant attributes for DNAGATv2Block that determines output size
            self.heads = kwargs.get('heads', 1)
            self.concat = kwargs.get('concat', False)
            # input and output channels/shape for each block
            self.input_output_list = list(zip([i * self.heads for i in output_channels], output_channels[1:]))\
                                     if self.concat else list(zip(output_channels[:], output_channels[1:]))
            # module list of layers with same args and kwargs
            self.blocks = nn.ModuleList([
                self.block(self.input_channels, self.output_channels[0], *args, **kwargs),
                *[self.block(input_channels, output_channels, *args, **kwargs)\
                for (input_channels, output_channels) in self.input_output_list]   
            ])

    def forward(self, x, *args, **kwargs):
        """ 
        function for forward pass of layers
        """
        # create copy of input
        out = x.detach().clone()
        # iterate over each block
        for block in self.blocks:
            # output y with shape [num_nodes, out_channels]
            x = block(x, *args, **kwargs)
            # concatenate to output
            out = T.cat((out, x), -1)
        return out 