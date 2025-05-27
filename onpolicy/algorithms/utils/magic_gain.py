import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.utils.nn import MLPBlock, NNLayers
from torch_geometric.nn.inits import reset
    
class GraphAttentionGAIN(nn.Module):
    """
    Graph-Attentional layer based on GAIN used in MAGIC that can process 
    differentiable communication graphs
    """

    def __init__(
            self, 
            in_features, 
            out_features, 
            dropout, 
            negative_slope, 
            num_agents,
            n_gnn_fc_layers, 
            num_heads=1,
            eps=1., 
            gnn_train_eps=False,
            gnn_norm='none', 
            self_loop_type=2, 
            average=False, 
            normalize=False, 
            device=torch.device("cpu")
        ):
        super(GraphAttentionGAIN, self).__init__()
        """
        Initialization method for the graph-attentional layer

        Arguments:
            in_features (int): number of features in each input node
            out_features (int): number of features in each output node
            dropout (int/float): dropout probability for the coefficients
            negative_slope (int/float): control the angle of the negative slope in leakyrelu
            num_agents (int): number of agents
            n_gnn_fc_layers (int): number of MLP layers in GAIN
            num_heads (int): number of heads of attention
            eps (float): initial value of epsilon in GAIN
            gnn_train_eps (bool): if training the epsilon in GAIN
            gnn_norm (str): normalization method for GAIN ('none', 'graphnorm')
            self_loop_type (int): 0 -- force no self-loop; 1 -- force self-loop; 
                other values -- keep the input adjacency matrix unchanged
            average (bool): if averaging all attention heads
            normalize (bool): if normalizing the coefficients after zeroing out weights using the communication graph
            device (torch.device): specifies the device to run on (cpu/gpu)
        """

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_agents = num_agents
        self.n_gnn_fc_layers = n_gnn_fc_layers
        self.num_heads = num_heads
        self.initial_eps = eps
        self.gnn_train_eps = gnn_train_eps
        self.gnn_norm = gnn_norm
        self.self_loop_type = self_loop_type
        self.average = average
        self.normalize = normalize
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.nn = NNLayers(
            input_channels=in_features if average else in_features * num_heads, 
            block=MLPBlock, 
            output_channels=[out_features for _ in range(n_gnn_fc_layers)],
            norm_type=gnn_norm,
        )
        self.W = nn.Parameter(torch.zeros(size=(in_features, num_heads * in_features)))
        self.a = nn.Parameter(torch.zeros(size=(num_heads, in_features, 1)))
        if gnn_train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        self.leakyrelu = nn.LeakyReLU(self.negative_slope)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialization for the parameters of the graph-attentional layer
        """
        reset(self.nn)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W.data, gain=gain)
        nn.init.xavier_normal_(self.a.data, gain=gain)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, input, adj, batch=None):
        """
        Forward function for the graph attention layer used in MAGIC

        Arguments:
            input (tensor): input of the graph attention layer [batch_size * N * in_features]
            adj (tensor): the learned communication graph (adjancy matrix) by the sub-scheduler [batch_size * N * N]
            batch (Optional[tensor]): batch vector if using graph normalization [batch_size * N]

        Return:
            the output of the graph attention layer
        """

        # perform linear transformation on the input, and generate multiple heads
        # self.W: [in_features * (num_heads*in_features)]
        # h (tensor): the matrix after performing the linear transformation [batch_size * N * num_heads * in_features]
        N = self.num_agents
        h = torch.matmul(input, self.W).reshape(-1, N, self.num_heads, self.in_features)
    
        # force the no self-loop to happen
        if self.self_loop_type == 0:
            adj = adj * (torch.ones(N, N) - torch.eye(N, N)).to(**self.tpdv).unsqueeze(0)
        # force the self-loop to happen
        elif self.self_loop_type == 1:
            adj = torch.eye(N, N).to(**self.tpdv).unsqueeze(0) + \
                adj * (torch.ones(N, N) - torch.eye(N, N)).to(**self.tpdv).unsqueeze(0)
        # the self-loop will be decided by the sub-scheduler
        else:
            pass
            
        # GATv2 attention mechanism

        # [batch_size * N * 1 * num_heads * in_features] --> [batch_size * N * N * num_heads * in_features]
        h_1 = h.unsqueeze(2).repeat(1, 1, self.num_agents, 1, 1)
        # [batch_size * 1 * N * num_heads * in_features] --> [batch_size * N * N * num_heads * in_features]
        h_2 = h.unsqueeze(1).repeat(1, self.num_agents, 1, 1, 1)
        # [batch_size * N * N * num_heads * in_features]
        e = h_1 + h_2
        e = self.leakyrelu(e)
        # [batch_size * N * N * num_heads * in_features] @
        # [num_heads * in_features * 1] --> [batch_size * N * N * num_heads * 1]
        # e (tensor): the matrix of unnormalized coefficients for all heads [batch_size * N * N * num_heads]
        # sometimes the unnormalized coefficients can be large, so regularization might be used 
        # to limit the large unnormalized coefficient values (TODO)
        e = torch.einsum('ijklm,lmn->ijkln', e, self.a).squeeze(-1) 
            
        # adj: [batch_size * N * N * 1]
        adj = adj.unsqueeze(-1)
        # attention (tensor): the matrix of coefficients used for the message aggregation 
        # [batch_size * N * N * num_heads]
        attention = e * adj
        attention = torch.exp(F.log_softmax(attention, dim=2)).nan_to_num() 
        # the weights from agents that should not communicate (send messages) will be 0, the gradients from 
        # the communication graph will be preserved in this way
        # add epsilon as per GAIN
        attention = (attention + self.eps) * adj   
        # normalize: make the sum of weights from all agents be 1
        if self.normalize:
            if self.self_loop_type != 1:
                attention += 1e-15
            attention = attention / attention.sum(dim=2, keepdim=True)
            attention = attention * adj
        # dropout on the coefficients  
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # [batch_size * N * in_features] --> [batch_size * N * in_features * num_heads]
        input_ = input.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)
        # [batch_size * N * N * num_heads] @ [batch_size * N * in_features * num_heads] 
        # --> [batch_size * N * in_features * num_heads]
        output = torch.einsum('ijkl,ikml->ijml', attention, input_)

        if self.average:
            # [batch_size * N * in_features * num_heads] --> [batch_size * N * in_features]
            output = torch.mean(output, dim=-1)
            if self.gnn_norm == 'graphnorm':
                # [batch_size * N * in_features] --> (batch_size * N) * in_features]
                output = output.view(-1, self.in_features)
        else:
            if self.gnn_norm == 'graphnorm':
                # [batch_size * N * in_features * num_heads] --> [(batch_size * N) * (in_features*num_heads)]
                output = output.view(-1, self.in_features * self.num_heads)
            else:
                # [batch_size * N * in_features * num_heads] --> [batch_size * N * (in_features*num_heads)]
                output = output.view(-1, N, self.in_features * self.num_heads)
        
        if self.gnn_norm == 'graphnorm':
            assert batch is not None, "Batch vector is required for graph normalization"
            # [(batch_size * N) * out_features] --> [batch_size * N * out_features]
            output = self.nn(output, batch=batch).view(-1, N, self.out_features)
        else:
            # [batch_size * N * out_features]
            output = self.nn(output)
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(in_features={}, out_features={})'.format(self.in_features, self.out_features)
    