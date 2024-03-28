import typing
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload


class GAINConv(MessagePassing):
    r"""
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        nn_norm_type (str): Normalisation used for nn
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities in case of a bipartite graph.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops
            (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge,
            *i.e.* :math:`\mathbf{\Theta}_{s} = \mathbf{\Theta}_{t}`.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    def __init__(
        self,
        nn: Callable,
        nn_norm_type: str,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.nn = nn
        self.nn_norm_type = nn_norm_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        # linear layer for source and target nodes
        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * in_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * in_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * in_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * in_channels,
                                    bias=bias, weight_initializer='glorot')

        # node independent parameters to calculate attention coefficients
        self.att = Parameter(torch.empty(1, heads, in_channels))

        # linear layer for edge dimensions
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * in_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        # no bias
        self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)

    @overload
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: NoneType = None,
        batch: OptTensor = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
        batch: OptTensor = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
        batch: OptTensor = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
        batch: OptTensor = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # get source and target embeddings and pass them to linear layers
        # (num_nodes, in_channels)
        x_l: OptTensor = None
        x_r: OptTensor = None
        # (num_nodes, in_channels) --> (num_nodes, att_heads, in_channels)
        x_lin_l: OptTensor = None
        x_lin_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_lin_l = self.lin_l(x).view(-1, self.heads, self.in_channels)
            if self.share_weights:
                x_lin_r = x_lin_l
            else:
                x_lin_r = self.lin_r(x).view(-1, self.heads, self.in_channels)
            x_l = x
            x_r = x
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_lin_l = self.lin_l(x).view(-1, self.heads, self.in_channels)
            if x_r is not None:
                x_lin_r = self.lin_r(x).view(-1, self.heads, self.in_channels)
        assert x_l is not None
        assert x_r is not None
        assert x_lin_l is not None
        assert x_lin_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (x: PairTensor, edge_attr: OptTensor)
        # (num_edges, att_heads)
        alpha = self.edge_updater(edge_index, x=(x_lin_l, x_lin_r),
                                  edge_attr=edge_attr)

        # propagate_type: (x: PairTensor, alpha: Tensor) 
        # (num_nodes, att_heads, in_channels)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)
        if self.concat:
            # (num_nodes, att_heads * in_channels)
            out = out.view(-1, self.heads * self.in_channels)
        else:
            # (num_nodes, in_channels) 
            out = out.mean(dim=1)
        # (num_nodes, att_heads * in_channels / in_channels) --> 
        # (num_nodes, out_channels)
        if self.nn_norm_type == 'graphnorm':
            out = self.nn(out, batch=batch)
        else:
            out = self.nn(out)

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out, None

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        # add source and target embeddings to 'emulate' concatentation 
        # given that linear layer is applied already
        # (num_edges, att_heads, in_channels)
        x = x_i + x_j
        
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            # (num_edges, edge_dims) --> (num_edges, att_heads * in_channels)
            edge_attr = self.lin_edge(edge_attr)
            # (num_edges, att_heads * in_channels) --> 
            # (num_edges, att_heads, in_channels)
            edge_attr = edge_attr.view(-1, self.heads, self.in_channels)
            # 'emulate' concatentation to node embeddings
            # (num_edges, att_heads, in_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        # x (num_edges, att_heads, in_channels), 
        # self.att (1, att_heads, in_channels) --> 
        # alpha (num_edges, att_heads)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        # apply attention weights to source nodes 
        # and add original source feature vector
        # (num_edges, att_heads, in_channels] 
        # * (num_edges, att_heads, 1) -->
        # (num_edges, att_heads, in_channels)
        return torch.tile(x_j.unsqueeze(1), (1, self.heads, 1)) \
            * (alpha.unsqueeze(-1) + 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
