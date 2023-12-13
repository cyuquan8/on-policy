import torch as T
import torch.nn as nn
import torch_geometric.nn as gnn

from onpolicy.algorithms.utils.dna_gatv2_conv import DNAGATv2Conv
from onpolicy.algorithms.utils.gain_conv import GAINConv
from onpolicy.algorithms.utils.gatv2_conv import GATv2Conv
from onpolicy.algorithms.utils.gin_gine_conv import GINConv

def activation_function(activation):
    """
    function that returns ModuleDict of activation functions
    """
    return  nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['sigmoid', nn.Sigmoid()],
        ['softmax', nn.Softmax(-1)],
        ['log_softmax', nn.LogSoftmax(-1)],
        ['tanh', nn.Tanh()],
        ['hard_tanh', nn.Hardtanh()],
        ['none', nn.Identity()]
    ])[activation]

def weights_initialisation_function_generator(weight_initialisation, activation_func, *args, **kwargs):
    """ 
    function that returns functions initialise weights according to specified methods. 
    """
    if weight_initialisation == 'xavier_uniform':
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight, 
                    gain=nn.init.calculate_gain(activation_func)
                )
        return init_weight
    elif weight_initialisation == 'xavier_normal':
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(
                    m.weight, 
                    gain=nn.init.calculate_gain(activation_func)
                )
        return init_weight
    elif weight_initialisation == 'kaiming_uniform':
        # recommend for relu / leaky relu
        assert (activation_func == 'relu' or activation_func == 'leaky_relu'), \
            "Non-linearity recommended to be 'relu' or 'leaky_relu'"
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, 
                    a=kwargs.get('kaiming_a', math.sqrt(5)), 
                    mode=kwargs.get('kaiming_mode', 'fan_in'), 
                    nonlinearity=activation_func
                )
        return init_weight
    elif weight_initialisation == 'kaiming_normal':
        # recommend for relu / leaky relu
        assert (activation_func == 'relu' or activation_func == 'leaky_relu'), \
            "Non-linearity recommended to be 'relu' or 'leaky_relu'"
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, 
                    a=kwargs.get('kaiming_a', math.sqrt(5)), 
                    mode=kwargs.get('kaiming_mode', 'fan_in'), 
                    nonlinearity=activation_func
                )
        return init_weight
    elif weight_initialisation == 'uniform':
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.uniform_(
                    m.weight, 
                    a=kwargs.get('uniform_lower_bound', 0.0), 
                    b=kwargs.get('uniform_upper_bound', 1.0)
                )
        return init_weight
    elif weight_initialisation == 'orthogonal':
        def init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(
                    m.weight, 
                    gain=nn.init.calculate_gain(activation_func)
                )
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
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2) 

class MLPBlock(nn.Module):
    """
    class to build MLPBlock
    """
    block_type = 'MLPBlock'

    def __init__(
            self, 
            input_channels, 
            output_channels, 
            norm_type='none', 
            activation_func='relu',
            dropout_p=0, 
            weight_initialisation='default', 
            *args, 
            **kwargs
        ):
        """
        class constructor that creates the layers attributes for MLPBlock 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input and output channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        # normalisation
        self.norm_type = norm_type
        if self.norm_type == 'batchnorm1d':
            self.norm = nn.BatchNorm1d(
                num_features=self.output_channels,
                eps=kwargs.get('batchnorm1d_eps', 1e-5),
                momentum=kwargs.get('batchnorm1d_momentum', 0.1),
                affine=kwargs.get('batchnorm1d_affine', True),
                track_running_stats=kwargs.get('batchnorm1d_track_running_stats', True)
            )
        elif self.norm_type == 'layernorm':
            self.norm = nn.LayerNorm(
                normalized_shape=self.output_channels,
                eps=kwargs.get('layernorm_eps', 1e-5),
                elementwise_affine=kwargs.get('layernorm_elementwise_affine', True)
            )
        elif self.norm_type == 'graphnorm':
            self.norm = gnn.GraphNorm(
                in_channels=self.output_channels,
                eps=kwargs.get('graphnorm_eps', 1e-5)
            )
        elif self.norm_type == 'none':
            self.norm = None
        else:
            raise Exception(f"{self.norm_type} not available for {self.block_type}")
        # activation function
        self.activation_func = activation_func 
        # dropout probablity
        self.dropout_p = dropout_p
        # weight initialisation
        self.weight_initialisation = weight_initialisation

        if self.norm is None:
            # input --> linear --> activation function --> dropout 
            self.block = nn.Sequential(
                nn.Linear(self.input_channels, self.output_channels),
                activation_function(self.activation_func),
                nn.Dropout(self.dropout_p)
            )
        elif self.norm is not None and self.norm_type == 'graphnorm':
            # input --> linear --> normalisation --> activation function --> dropout
            self.block = gnn.Sequential(
                'x, batch', [
                    (nn.Linear(self.input_channels, self.output_channels, bias=False), 'x -> x'),
                    (self.norm, 'x, batch -> x'),
                    (activation_function(self.activation_func), 'x -> x'),
                    (nn.Dropout(self.dropout_p), 'x -> x')
                ]
            ) 
        else:
            # input --> linear --> normalisation --> activation function --> dropout 
            self.block = nn.Sequential(
                nn.Linear(self.input_channels, self.output_channels, bias=False),
                self.norm,
                activation_function(self.activation_func),
                nn.Dropout(self.dropout_p)
            )
        
        # weight initialisation
        self.block.apply(
            weights_initialisation_function_generator(
                self.weight_initialisation, 
                self.activation_func, 
                *args, 
                *kwargs
            )
        )

    def forward(self, x, *args, **kwargs):
        """ 
        function for forward pass of MLPBlock 
        """
        x = self.block(x, *args, **kwargs)
        return x

class ConvBlock(nn.Module):
    """ 
    class to build ConvBlock
    """ 
    block_type = 'ConvBlock'

    def __init__(
            self, 
            input_channels, 
            output_channels,
            conv,
            norm_type='none',  
            activation_func='relu',
            pooling_type='none', 
            dropout_type='none', 
            *args, 
            **kwargs
        ):
        """ 
        class constructor that creates the layers attributes for ConvBlock 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()
        
        # input and output channels for conv (num of filters)
        self.input_channels = input_channels
        self.output_channels = output_channels
        # class of conv
        self.conv = conv
        # normalisation
        self.norm_type = norm_type
        if self.norm_type == 'batchnorm2d':
            self.norm = nn.BatchNorm2d(
                num_features=self.output_channels,
                eps=kwargs.get('batchnorm2d_eps', 1e-5),
                momentum=kwargs.get('batchnorm2d_momentum', 0.1),
                affine=kwargs.get('batchnorm2d_affine', True),
                track_running_stats=kwargs.get('batchnorm2d_track_running_stats', True)
            )
        elif self.norm_type == 'instancenorm2d':
            self.norm = nn.InstanceNorm2d(
                num_features=self.output_channels,
                eps=kwargs.get('instancenorm2d_eps', 1e-5),
                momentum=kwargs.get('instancenorm2d_momentum', 0.1),
                affine=kwargs.get('instancenorm2d_affine', False),
                track_running_stats=kwargs.get('instancenorm2d_track_running_stats', False)
            )
        elif self.norm_type == 'none':
            self.norm = None
        else:
            raise Exception(f"{self.norm_type} not available for {self.block_type}")
        # activation function
        self.activation_func = activation_func
        # pooling
        self.pooling_type = pooling_type
        if self.pooling_type == 'maxpool2d':
            self.pooling = torch.nn.MaxPool2d(
                kernel_size=kwargs.get('maxpool2d_kernel_size', 3), 
                stride=kwargs.get('maxpool2d_stride', None), 
                padding=kwargs.get('maxpool2d_padding', 0), 
                dilation=kwargs.get('maxpool2d_dilation', 1), 
                return_indices=kwargs.get('maxpool2d_return_indices', False),
                ceil_mode=kwargs.get('maxpool2d_ceil_mode', False) # set to True to account for odd dimensions
            )
        elif self.pooling_type == 'avgpool2d':
            self.pooling = torch.nn.AvgPool2d(
                kernel_size=kwargs.get('avgpool2d_kernel_size', 3), 
                stride=kwargs.get('avgpool2d_stride', None), 
                padding=kwargs.get('avgpool2d_padding', 0), 
                ceil_mode=kwargs.get('avgpool2d_ceil_mod', False), # set to True to account for odd dimensions
                count_include_pad=kwargs.get('avgpool2d_count_include_pad', True),
                divisor_override=kwargs.get('avgpool2d_divisor_override', None)
            )
        elif self.pooling_type == 'none':
            self.pooling = None
        else:
            raise Exception(f"{self.pooling_type} not available for {self.block_type}")
        # dropout
        self.dropout_type = dropout_type
        if self.dropout_type == 'dropout':
            self.dropout = torch.nn.Dropout(
                p=kwargs.get('dropout_p', 0.5), 
                inplace=kwargs.get('dropout_inplace', False)
            )
        elif self.dropout_type == 'dropout2d':
            self.dropout = torch.nn.Dropout2d(
                p=kwargs.get('dropout2d_p', 0.5), 
                inplace=kwargs.get('dropout2d_inplace', False)
            )
        elif self.dropout_type == 'none':
            self.dropout = None
        else:
            raise Exception(f"{self.dropout_type} not available for {self.block_type}")
        
        if self.norm_type is None and self.pooling_type is None and self.dropout is None:
            # input --> conv --> activation func
            self.block = nn.Sequential(
                self.conv(self.input_channels, self.output_channels),
                activation_function(self.activation_func),
            )
        elif self.norm_type is not None and self.pooling_type is None and self.dropout is None:
            # input --> conv --> normalisation --> activation func
            self.block = nn.Sequential(
                self.conv(self.input_channels, self.output_channels),
                self.norm,
                activation_function(self.activation_func),
            )
        elif self.norm_type is None and self.pooling_type is not None and self.dropout is None:
            # input --> conv --> activation func --> pooling
            self.block = nn.Sequential(
                self.conv(self.input_channels, self.output_channels),
                activation_function(self.activation_func),
                self.pooling,
            )
        elif self.norm_type is None and self.pooling_type is None and self.dropout is not None:
            # input --> conv --> activation func --> dropout
            self.block = nn.Sequential(
                self.conv(self.input_channels, self.output_channels),
                activation_function(self.activation_func),
                self.dropout,
            )
        elif self.norm_type is not None and self.pooling_type is not None and self.dropout is None:
            # input --> conv --> normalisation --> activation func --> pooling
            self.block = nn.Sequential(
                self.conv(self.input_channels, self.output_channels),
                self.norm,
                activation_function(self.activation_func),
                self.pooling,
            )
        elif self.norm_type is not None and self.pooling_type is None and self.dropout is not None:
            # input --> conv --> normalisation --> activation func --> dropout
            self.block = nn.Sequential(
                self.conv(self.input_channels, self.output_channels),
                self.norm,
                activation_function(self.activation_func),
                self.dropout,
            )
        elif self.norm_type is None and self.pooling_type is not None and self.dropout is not None:
            # input --> conv --> activation func --> pooling --> dropout
            self.block = nn.Sequential(
                self.conv(self.input_channels, self.output_channels),
                activation_function(self.activation_func),
                self.pooling,
                self.dropout,
            )
        elif self.norm_type is not None and self.pooling_type is not None and self.dropout is not None:
            # input --> conv --> normalisation --> activation func --> pooling --> dropout
            self.block = nn.Sequential(
                self.conv(self.input_channels, self.output_channels),
                self.norm,
                activation_function(self.activation_func),
                self.pooling,
                self.dropout,
            )

    def forward(self, x):
        """ 
        function for forward pass of ConvBlock 
        """
        x = self.block(x)
        return x

class DNAGATv2Block(nn.Module):
    """ 
    class to build DNAGATv2Block 
    """
    block_type = 'DNAGATv2Block'

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
            gnn_cpa_model='none',
            norm_type='none',
            activation_func='relu',
            *args,
            **kwargs
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
        # normalisation
        self.norm_type = norm_type
        if self.norm_type == 'graphnorm':
            self.norm = gnn.GraphNorm(
                in_channels=self.output_channels,
                eps=kwargs.get('graphnorm_eps', 1e-5)
            )
        elif self.norm_type == 'none':
            self.norm = None
        else:
            raise Exception(f"{self.norm_type} not available for {self.block_type}")
        # activation function
        self.activation_func = activation_func

        if self.norm is None:
            # input --> DNAGATv2Conv --> activation function
            self.block = gnn.Sequential(
                'x, edge_index, edge_attr, return_attention_weights', [
                    (DNAGATv2Conv(
                        in_channels=input_channels, 
                        out_channels=output_channels, 
                        att_heads=att_heads, 
                        mul_att_heads=mul_att_heads, 
                        groups=groups, 
                        negative_slope=negative_slope, 
                        dropout=dropout, 
                        add_self_loops=add_self_loops, 
                        edge_dim=edge_dim, 
                        fill_value=fill_value, 
                        bias=bias, 
                        gnn_cpa_model=gnn_cpa_model
                    ), 'x, edge_index, edge_attr, return_attention_weights -> x, extra'),
                    (activation_function(self.activation_func), 'x -> x'),
                    (lambda x_1, x_2: (x_1, x_2), 'x, extra -> x')
                ]
            )
        else:
            # input --> DNAGATv2Conv --> normalisation --> activation function
            self.block = gnn.Sequential(
                'x, edge_index, edge_attr, return_attention_weights, batch', [
                    (DNAGATv2Conv(
                        in_channels=input_channels, 
                        out_channels=output_channels, 
                        att_heads=att_heads, 
                        mul_att_heads=mul_att_heads, 
                        groups=groups, 
                        negative_slope=negative_slope, 
                        dropout=dropout, 
                        add_self_loops=add_self_loops, 
                        edge_dim=edge_dim, 
                        fill_value=fill_value, 
                        bias=bias, 
                        gnn_cpa_model=gnn_cpa_model
                    ), 'x, edge_index, edge_attr, return_attention_weights -> x, extra'), 
                    (self.norm, 'x, batch -> x'),
                    (activation_function(self.activation_func), 'x -> x'),
                    (lambda x_1, x_2: (x_1, x_2), 'x, extra -> x')
                ]
            )

    def forward(self, x, edge_index, *args, **kwargs):
        """ 
        function for forward pass of DNAGATv2Block 
        """
        x, extra = self.block(x, edge_index, *args, **kwargs)
        return x, extra

class GATv2Block(nn.Module):
    """ 
    class to build GATv2Block 
    """
    block_type = 'GATv2Block'

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
            norm_type='none',
            activation_func='relu',
            *args,
            **kwargs,
        ):
        """ 
        class constructor for attributes of the GATv2Block 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()

        # input and output channels for GATv2Conv 
        self.input_channels = input_channels
        self.output_channels = output_channels
        # number of heads for gatv2
        self.heads = heads
        # boolean that when set to false, the multi-head attentions are averaged instead of concatenated
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
        # normalisation
        self.norm_type = norm_type
        if self.norm_type == 'graphnorm':
            self.norm = gnn.GraphNorm(
                in_channels=self.output_channels * self.heads if self.concat else self.output_channels,
                eps=kwargs.get('graphnorm_eps', 1e-5)
            )
        elif self.norm_type == 'none':
            self.norm = None
        else:
            raise Exception(f"{self.norm_type} not available for {self.block_type}")
        # activation function
        self.activation_func = activation_func

        if self.norm is None:
            # input --> GATv2Conv --> activation function
            self.block = gnn.Sequential(
                'x, edge_index, edge_attr, return_attention_weights', [
                    (GATv2Conv(
                        in_channels=input_channels, 
                        out_channels=output_channels, 
                        heads=heads, 
                        concat=concat, 
                        negative_slope=negative_slope, 
                        dropout=dropout, 
                        add_self_loops=add_self_loops, 
                        edge_dim=edge_dim, 
                        fill_value=fill_value, 
                        bias=bias, 
                        share_weights=share_weights, 
                        gnn_cpa_model=gnn_cpa_model
                    ), 'x, edge_index, edge_attr, return_attention_weights -> x, extra'),
                    (activation_function(self.activation_func), 'x -> x'),
                    (lambda x_1, x_2: (x_1, x_2), 'x, extra -> x')
                ]
            )
        else:
            # input --> GATv2Conv --> normalisation --> activation function
            self.block = gnn.Sequential(
                'x, edge_index, edge_attr, return_attention_weights, batch', [
                    (GATv2Conv(
                        in_channels=input_channels, 
                        out_channels=output_channels, 
                        heads=heads, 
                        concat=concat, 
                        negative_slope=negative_slope, 
                        dropout=dropout, 
                        add_self_loops=add_self_loops, 
                        edge_dim=edge_dim, 
                        fill_value=fill_value, 
                        bias=bias, 
                        share_weights=share_weights, 
                        gnn_cpa_model=gnn_cpa_model
                    ), 'x, edge_index, edge_attr, return_attention_weights -> x, extra'), 
                    (self.norm, 'x, batch -> x'),
                    (activation_function(self.activation_func), 'x -> x'),
                    (lambda x_1, x_2: (x_1, x_2), 'x, extra -> x')
                ]
            )

    def forward(self, x, edge_index, *args, **kwargs):
        """ 
        function for forward pass of DNAGATv2Block 
        """
        x, extra = self.block(x, edge_index, *args, **kwargs)
        return x, extra

class GINBlock(nn.Module):
    """ 
    class to build GINBlock 
    """
    block_type = 'GINBlock'

    def __init__(
            self, 
            input_channels, 
            output_channels, 
            n_gnn_fc_layers,  
            eps=0.0,
            train_eps=False, 
            norm_type='none',
            *args, 
            **kwargs
        ):
        """ 
        class constructor for attributes of the GINBlock 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()

        # input and output channels for GINConv
        self.input_channels = input_channels
        self.output_channels = output_channels
        # number of layers of MLP of GINConv
        self.n_gnn_fc_layers = n_gnn_fc_layers
        # epsilon
        self.eps = eps
        # boolean to track if eps is trainable parameter
        self.train_eps = train_eps
        # normalisation
        self.norm_type = norm_type

        # mlp 
        self.nn = NNLayers(
            input_channels=input_channels, 
            block=MLPBlock, 
            output_channels=[output_channels for _ in range(n_gnn_fc_layers)],
            norm_type=norm_type,
            *args,
            **kwargs
        )

        if self.norm_type == 'graphnorm':
            # input --> GINConv
            self.block = gnn.Sequential(
                'x, edge_index, size, batch', [
                    (GINConv(
                        nn=self.nn,
                        nn_norm_type=self.norm_type, 
                        eps=self.eps, 
                        train_eps=self.train_eps
                    ), 'x, edge_index, size, batch -> x')
                ]
            )
        else:
            # input --> GINConv
            self.block = gnn.Sequential(
                'x, edge_index, size', [
                    (GINConv(
                        nn=self.nn,
                        nn_norm_type=self.norm_type, 
                        eps=self.eps, 
                        train_eps=self.train_eps
                    ), 'x, edge_index, size -> x')
                ]
            )

    def forward(self, x, edge_index, *args, **kwargs):
        """ 
        function for forward pass of GINBlock 
        """
        x = self.block(x, edge_index, *args, **kwargs)
        return x

class GAINBlock(nn.Module):
    """ 
    class to build GAINBlock 
    """
    block_type = 'GAINBlock'

    def __init__(
            self, 
            input_channels,
            output_channels,
            n_gnn_fc_layers,
            heads=1,
            concat=True,
            negative_slope=0.2,
            dropout=0.0,
            add_self_loops=True,
            edge_dim=None,
            fill_value='mean',
            share_weights=False,
            norm_type='none',
            *args,
            **kwargs
        ):
        """ 
        class constructor for attributes of the GAINBlock 
        """
        # inherit class constructor attributes from nn.Module
        super().__init__()

        # input and output channels for GAINBlock 
        self.input_channels = input_channels
        self.output_channels = output_channels
        # number of layers of MLP of GINConv
        self.n_gnn_fc_layers = n_gnn_fc_layers
        # number of heads for gatv2
        self.heads = heads
        # boolean that when set to false, the multi-head attentions are averaged instead of concatenated
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
        # whether to use same matrix for target and source nodes
        self.share_weights = share_weights
        # normalisation
        self.norm_type = norm_type

        # mlp
        self.nn = NNLayers(
            input_channels=input_channels * heads if concat else input_channels, 
            block=MLPBlock, 
            output_channels=[output_channels for _ in range(n_gnn_fc_layers)],
            heads=heads,
            concat=concat,
            norm_type=norm_type,
            *args,
            **kwargs
        )

        if self.norm_type == 'graphnorm':
            # input --> GAINConv
            self.block = gnn.Sequential(
                'x, edge_index, edge_attr, return_attention_weights, batch', [
                    (GAINConv(
                        nn=self.nn,
                        nn_norm_type=self.norm_type, 
                        in_channels=input_channels, 
                        out_channels=output_channels, 
                        heads=heads,
                        concat=concat, 
                        negative_slope=negative_slope, 
                        dropout=dropout, 
                        add_self_loops=add_self_loops, 
                        edge_dim=edge_dim, 
                        fill_value=fill_value,  
                        share_weights=share_weights
                    ), 'x, edge_index, edge_attr, return_attention_weights, batch -> x, extra')
                ]
            )
        else:
            # input --> GAINConv
            self.block = gnn.Sequential(
                'x, edge_index, edge_attr, return_attention_weights', [
                    (GAINConv(
                        nn=self.nn,
                        nn_norm_type=self.norm_type, 
                        in_channels=input_channels, 
                        out_channels=output_channels, 
                        heads=heads, 
                        concat=concat,
                        negative_slope=negative_slope, 
                        dropout=dropout, 
                        add_self_loops=add_self_loops, 
                        edge_dim=edge_dim, 
                        fill_value=fill_value,  
                        share_weights=share_weights
                    ), 'x, edge_index, edge_attr, return_attention_weights -> x, extra')
                ]
            )

    def forward(self, x, edge_index, *args, **kwargs):
        """ 
        function for forward pass of GAINBlock 
        """
        x, extra = self.block(x, edge_index, *args, **kwargs)
        return x, extra

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

class GNNDNALayers(NNLayers):
    """ 
    class to build layers of DNA-based blocks
    DNA-based blocks are different from other blocks as it requries past layers as its inputs
    """
    def __init__(self, input_channels, block, output_channels, *args, **kwargs):
        """ 
        class constructor for attributes of DNA-based blocks 
        """
        # ensure that DNA-based blocks are used
        assert block.block_type == 'DNAGATv2Block', f"{block.block_type} is not DNA-based block"
        # inherit class constructor attributes from nn_layers
        super().__init__(input_channels, block, output_channels, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        """ 
        function for forward pass of layers
        """
        # list to store extra outputs. assumes extra outputs are packed into a tuple. 
        extra_output_list = []
        # default no extra outputs
        extra_output = False
        # check for extra outputs
        if self.block.block_type == 'DNAGATv2Block':
            extra_output = kwargs.get('return_attention_weights', False)

        # add layer dimension to initial input, [shape: (num_nodes, 1, input_channels==output_channels)]
        x = T.unsqueeze(x, 1)
        # iterate over each block
        for block in self.blocks:
            # output y with extra output, [shape: (num_nodes, input_channels==output_channels)]
            y, extra = block(x, *args, **kwargs)
            # append extra output to list
            if extra_output:
                extra_output_list.append(extra)
            # add layer dimensions to output and concatenate y to existing x
            x = T.cat((x, T.unsqueeze(y, 1)), 1)
        
        if extra_output:
            return x, extra_output_list
        else:
            return x 

class GNNAllLayers(NNLayers):
    """ 
    class to build layers of GNN blocks to include all layers
    """
    def __init__(self, input_channels, block, output_channels, *args, **kwargs):
        """ 
        class constructor for attributes of GNNAllLayers 
        """
        # ensure that DNA-based blocks are not used
        assert block.block_type != 'DNAGATv2Block', f"{block.block_type} cannot be DNA-based block"
        # inherit class constructor attributes from nn_layers
        super().__init__(input_channels, block, output_channels, *args, **kwargs)

        # update GATv2Block accounting for concatenation or averaging of attention heads
        if self.block.block_type == 'GATv2Block':
            # obtain relevant attributes for that determines output size
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
        # list to store extra outputs. assumes extra outputs are packed into a tuple. 
        extra_output_list = []
        # default no extra outputs
        extra_output = False
        # check for extra outputs
        if self.block.block_type == 'GATv2Block' or self.block.block_type == 'GAINBlock':
            extra_output = kwargs.get('return_attention_weights', False)

        # create copy of input
        out = x.clone()
        # iterate over each block
        for block in self.blocks:
            # output y, [shape: (num_nodes, output_channels)]
            if self.block.block_type == 'GINBlock':
                x = block(x, *args, **kwargs)
            else:
                x, extra = block(x, *args, **kwargs)
                # append extra output to list
                if extra_output:
                    extra_output_list.append(extra)
            # concatenate to output, [shape: (num_nodes, input_channels + len(self.blocks) * output_channels)]
            out = T.cat((out, x), -1)

        if extra_output:
            return out, extra_output_list
        else:
            return out