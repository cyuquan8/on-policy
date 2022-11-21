import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros, kaiming_uniform, uniform
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

class DGCNLinear(torch.nn.Module):
    """ 
    class for linear model for purposes of DNA / DGCN 
    """

    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        """ 
        function to set attributes 
        """
        # call nn.Module init
        super().__init__()

        # in and out channels must be divisible by groups
        assert in_channels % groups == 0 and out_channels % groups == 0

        # linear model attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        # set weight
        self.weight = Parameter(torch.Tensor(groups, in_channels // groups, out_channels // groups))

        # set bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # intialise model weights
        self.reset_parameters()

    def reset_parameters(self):

        """ function to reset model weights """

        kaiming_uniform(self.weight, fan = self.weight.size(1), a = math.sqrt(5))
        uniform(self.weight.size(1), self.bias)

    def forward(self, src):
        """ 
        function to conduct forward propagation 
        """

        # Input: [*, in_channels]
        # Output: [*, out_channels]

        # grouping case
        if self.groups > 1:
            # obtain size = *
            size = src.size()[:-1]
            # group src
            src = src.view(-1, self.groups, self.in_channels // self.groups)
            # transpose src to (group, *, in_channels // groups)
            src = src.transpose(0, 1).contiguous()
            # matmul src with weight to obtain out with size (groups, *, out_channels // groups) 
            out = torch.matmul(src, self.weight)
            # transpose src to (*, groups, out_channels // groups)
            out = out.transpose(1, 0).contiguous()
            # resize out to size (*, out_channels)
            out = out.view(size + (self.out_channels, ))
        # no grouping
        else:
            # out with size (*, out_channels) 
            out = torch.matmul(src, self.weight.squeeze(0))

        # add bias if any
        if self.bias is not None:
            out += self.bias

        return out

    def __repr__(self) -> str:  # pragma: no cover
        """ 
        function for representation 
        """
        return (f'{self.__class__.__name__}({self.in_channels}, 'f'{self.out_channels}, groups = {self.groups})')

def restricted_softmax(src, dim: int=-1, margin: float=0.):
    """ 
    function to calculate restricted softmax 
    """
    # find maximum positive exponent from src
    src_max = torch.clamp(src.max(dim = dim, keepdim = True)[0], min = 0.)
    # divide by e^(src_max)
    out = (src - src_max).exp()
    # essentially e^{src_i} / (1 + sum_j(e^(src_j))) 
    out = out / (out.sum(dim = dim, keepdim = True) + (margin - src_max).exp())

    return out

class Attention(torch.nn.Module):
    """ 
    class for attention block 
    """

    def __init__(self, dropout = 0):
        """ 
        class constructor to set attributes 
        """ 
        # call nn.Module init
        super().__init__()

        # dropout for attention
        self.dropout = dropout

    def forward(self, query, key, value):
        """ 
        function to conduct forward pass 
        """ 
        return self.compute_attention(query, key, value)

    def compute_attention(self, query, key, value):
        """ 
        function to compute attention 
        """

        # query: [*, query_entries, dim_k]
        # key: [*, key_entries, dim_k]
        # value: [*, key_entries, dim_v]
        # Output: [*, query_entries, dim_v]

        # check dimensions
        assert query.dim() == key.dim() == value.dim() >= 2
        assert query.size(-1) == key.size(-1)
        assert key.size(-2) == value.size(-2)

        # Score: [*, query_entries, key_entries]

        # matmul query and key
        score = torch.matmul(query, key.transpose(-2, -1))
        # scale score by dimension of key
        score = score / math.sqrt(key.size(-1))
        # restricted softmax
        score = restricted_softmax(score, dim = -1)
        # apply dropout
        score = F.dropout(score, p = self.dropout, training = self.training)

        return torch.matmul(score, value)

    def __repr__(self) -> str:  # pragma: no cover
        """ 
        function for representation 
        """
        return f'{self.__class__.__name__}(dropout = {self.dropout})'

class MultiHead(Attention):
    """ 
    class for multi head attention  
    """

    def __init__(self, in_channels, out_channels, heads = 1, groups = 1, dropout = 0, bias = True):
        """ 
        class constructor to set attributes 
        """ 
        # call init from Attention class
        super().__init__(dropout)

        # set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.groups = groups
        self.bias = bias

        # check dimensions
        assert in_channels % heads == 0 and out_channels % heads == 0
        assert in_channels % groups == 0 and out_channels % groups == 0
        assert max(groups, self.heads) % min(groups, self.heads) == 0

        # create linear models
        self.lin_q = DGCNLinear(in_channels, out_channels, groups, bias)
        self.lin_k = DGCNLinear(in_channels, out_channels, groups, bias)
        self.lin_v = DGCNLinear(in_channels, out_channels, groups, bias)

        # initialise parameters
        self.reset_parameters()

    def reset_parameters(self):
        """ 
        function to reset parameters
        """
        # reset paramters
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()

    def forward(self, query, key, value):
        """ 
        function for forward propagation 
        """

        # query: [*, query_entries, in_channels]
        # key: [*, key_entries, in_channels]
        # value: [*, key_entries, in_channels]
        # Output: [*, query_entries, out_channels]

        # assert dimensions
        assert query.dim() == key.dim() == value.dim() >= 2
        assert query.size(-1) == key.size(-1) == value.size(-1)
        assert key.size(-2) == value.size(-2)

        # pass through linear models
        query = self.lin_q(query)
        key = self.lin_k(key)
        value = self.lin_v(value)

        # query: [*, heads, query_entries, out_channels // heads]
        # key: [*, heads, key_entries, out_channels // heads]
        # value: [*, heads, key_entries, out_channels // heads]
        size = query.size()[:-2]
        out_channels_per_head = self.out_channels // self.heads

        # size [*, heads, query_entries, out_channels // heads]
        query_size = size + (query.size(-2), self.heads, out_channels_per_head)
        query = query.view(query_size).transpose(-2, -3)

        # size [*, heads, key_entries, out_channels // heads]
        key_size = size + (key.size(-2), self.heads, out_channels_per_head)
        key = key.view(key_size).transpose(-2, -3)

        # size [*, heads, key_entries, out_channels // heads]
        value_size = size + (value.size(-2), self.heads, out_channels_per_head)
        value = value.view(value_size).transpose(-2, -3)

        # output: [*, heads, query_entries, out_channels // heads]
        out = self.compute_attention(query, key, value)
        # output: [*, query_entries, heads, out_channels // heads]
        out = out.transpose(-3, -2).contiguous()
        # output: [*, query_entries, out_channels]
        out = out.view(size + (query.size(-2), self.out_channels))

        return out

    def __repr__(self) -> str:  # pragma: no cover
        """ 
        function for representation 
        """
        return (f'{self.__class__.__name__}({self.in_channels}, 'f'{self.out_channels}, heads = {self.heads}, 'f'groups = {self.groups}, dropout = {self.droput}, 'f'bias = {self.bias})')

class DGCNConv(MessagePassing):
    """ 
    class for dynamic graph communication convolution operator 
    """
    # typing 
    _alpha: OptTensor

    def __init__(self, in_channels: int, out_channels: int, att_heads: int = 1, mul_att_heads: int = 1, groups: int = 1, concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0, 
                 add_self_loops: bool = True, edge_dim: Optional[int] = None, fill_value: Union[float, Tensor, str] = 'mean', bias: bool = True, **kwargs):
        """
        class constructor to set attributes
        """
        # set default aggregation method for MesssagePassing
        kwargs.setdefault('aggr', 'add')

        # call init from MessagePassing
        super().__init__(node_dim = 0, **kwargs)

        # g2a attributes

        # input channels for first attention based propagation
        self.in_channels = in_channels
        # output channels after first attention based propagation
        self.out_channels = out_channels
        # number of heads for first attention based propagation
        self.att_heads = att_heads
        # boolean to track to concatenate output after first attention based propagation. mean is used if false
        self.concat = concat
        # negative slope for leaky relu
        self.negative_slope = negative_slope
        # probability of dropout
        self.dropout = dropout
        # boolean to track to add self loops to graph
        self.add_self_loops = add_self_loops
        # dimensions of edge if any
        self.edge_dim = edge_dim
        # fill value for edge attributes for self loops
        self.fill_value = fill_value

        # dna variables
        self.mul_att_heads = mul_att_heads

        # boolean to track if first propagation has been conducted
        self.first_prop = True

        # linear layer for source nodes
        self.lin = Linear(in_channels, att_heads * out_channels, bias = bias, weight_initializer = 'glorot')
        # parameter layer post leaky relu that is node independent
        self.att = Parameter(torch.Tensor(1, att_heads, out_channels))

        # check if there are edge dimensions
        if edge_dim is not None:
            # linear layer for edge dimensions
            self.lin_edge = Linear(edge_dim, att_heads * out_channels, bias = False, weight_initializer = 'glorot')
        else:
            self.lin_edge = None

        # check if there is bias and output concatenation
        if bias and concat:
            # parameter layer for bias
            self.bias = Parameter(torch.Tensor(att_heads * out_channels))
        # check if there is bias and not output concatenation
        elif bias and not concat:
            # parameter layer for bias
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # attribute for attention score
        self._alpha = None

        # multi attention head
        self.multi_head = MultiHead(out_channels * att_heads if concat else out_channels, out_channels * att_heads if concat else out_channels, mul_att_heads, groups, dropout, bias)

        # reset parameters 
        self.reset_parameters()

    def reset_parameters(self):
        """ 
        function to reset paremeters 
        """
        # reset parameters for linear layer
        self.lin.reset_parameters()

        # check if there is linear layer for edge dimensions
        if self.lin_edge is not None:
            # reset parameters for linear layer for edge dimensions
            self.lin_edge.reset_parameters()

        # reinitialise parameter layer and bias
        glorot(self.att)
        zeros(self.bias)

        # reset the parameters for mutli head attention
        self.multi_head.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, return_attention_weights: bool = None):
        """ 
        function to conduct forward propagation 
        """

        # check if the dimensions of the input are of shape [num_nodes, num_layers, channels]
        if x.dim() != 3:
            raise ValueError('Feature shape must be [num_nodes, num_layers, channels].')

        # obtain current node embedding: [num_nodes, num_layers, channels] --> [num_nodes, channels]
        x_curr = torch.squeeze(x[:, -1:], 1)
        # pass current node embedding to lin layer 
        x_curr = self.lin(x_curr).view(-1, self.att_heads, self.out_channels)

        # check to add self_loops to edge indexes
        if self.add_self_loops:
            # obtain number of nodes
            num_nodes = x.size(0)
            # remove all self loops and their attributes
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            # add self loops and correspoding edge attributes of self loops
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value = self.fill_value, num_nodes = num_nodes)

        # set first pass boolean
        self.first_prop = True
        # first propagate based on gatv2 on current node embeddings
        out = self.propagate(edge_index, x = None, x_layer = x_curr, edge_attr = edge_attr, size = None)
        # set first pass boolean
        self.first_prop = False
        
        # update alpha to be returned if required
        alpha = self._alpha
        self._alpha = None

        # check if outputs are concatenated
        if self.concat:
            # concatenate outputs
            out = out.view(-1, self.att_heads * self.out_channels)
        # mean
        else:
            # obtain mean across heads
            out = out.mean(dim = 1)

        # add bias if its exist
        if self.bias is not None:
            out += self.bias

        # second propagate based on dna on all layers of node beddings and output of first propagation from gatv2
        out = self.propagate(edge_index, x = x, x_curr = out, edge_attr = edge_attr, size = None)

        # check if there is need to return attention weights
        if isinstance(return_attention_weights, bool):
            # ensure that the attention weights exist
            assert alpha is not None

            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor, x_layer_j: Tensor, x_layer_i: Tensor, edge_attr: OptTensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:

        # check if its the first propagation
        if self.first_prop == True:
            # concatenate source and target embeddings
            x_curr = x_layer_i + x_layer_j
            # check if there are edge attributes
            if edge_attr is not None:
                # check dimensions of edge dimensions
                if edge_attr.dim() == 1:
                    # resize if 1
                    edge_attr = edge_attr.view(-1, 1)
                # ensure that there is linear layer for edges
                assert self.lin_edge is not None
                # pass edge attributes to lin_edge
                edge_attr = self.lin_edge(edge_attr)
                # change shape from [num_edges, num_heads * out_channels] --> [num_edges, num_heads, out_channels]
                edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
                # concatenate to node embeddings
                x_curr += edge_attr

            # pass node embeddings through leaky relu
            x_curr = F.leaky_relu(x_curr, self.negative_slope)
            # multiply by node independent parameter layer. sum over out_channels with shape [num_edges, num_heads]
            alpha = (x_curr * self.att).sum(dim = -1)
            # calculate softmaxed attention weights with shape [num_edges, num_heads]
            alpha = softmax(alpha, index, ptr, size_i)
            # store attention weights
            self._alpha = alpha
            # apply dropout to attention weights
            alpha = F.dropout(alpha, p = self.dropout, training = self.training)
            # apply attention weights to target nodes: [num_edges, num_heads, out_channels] * [num_edges, num_heads, 1]
            return x_layer_j * alpha.unsqueeze(-1)
        # second propagation
        else:
            # add layer dimensions to output of first propagation to, send it to multi head attention and remove layer dimension
            return self.multi_head(x_layer_i.unsqueeze(1), x_j, x_j).squeeze(1) 

    def __repr__(self) -> str:

        return (f'{self.__class__.__name__}({self.in_channels}, 'f'{self.out_channels}, att_heads = {self.att_heads}), 'f'mult_att_heads={self.multi_head.heads}, 'f'groups={self.multi_head.groups})')