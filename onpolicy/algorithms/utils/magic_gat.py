import torch
import torch.nn as nn
import torch.nn.functional as F
    
class GraphAttentionGAT(nn.Module):
    """
    Graph-Attentional layer based on GAT used in MAGIC that can process 
    differentiable communication graphs
    """

    def __init__(self, in_features, out_features, dropout, negative_slope, num_agents, num_heads=1, bias=True, 
        self_loop_type=2, average=False, normalize=False, device=torch.device("cpu")):
        super(GraphAttentionGAT, self).__init__()
        """
        Initialization method for the graph-attentional layer

        Arguments:
            in_features (int): number of features in each input node
            out_features (int): number of features in each output node
            dropout (int/float): dropout probability for the coefficients
            negative_slope (int/float): control the angle of the negative slope in leakyrelu
            num_agents (int): number of agents
            num_heads (int): number of heads of attention
            bias (bool): if adding bias to the output
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
        self.num_heads = num_heads
        self.self_loop_type = self_loop_type
        self.average = average
        self.normalize = normalize
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.W = nn.Parameter(torch.zeros(size=(in_features, num_heads * out_features)))
        self.a_i = nn.Parameter(torch.zeros(size=(num_heads, out_features, 1)))
        self.a_j = nn.Parameter(torch.zeros(size=(num_heads, out_features, 1)))
        if bias:
            if average:
                self.bias = nn.Parameter(torch.DoubleTensor(out_features))
            else:
                self.bias = nn.Parameter(torch.DoubleTensor(num_heads * out_features))
        else:
            self.register_parameter('bias', None)
        self.leakyrelu = nn.LeakyReLU(self.negative_slope)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialization for the parameters of the graph-attentional layer
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W.data, gain=gain)
        nn.init.xavier_normal_(self.a_i.data, gain=gain)
        nn.init.xavier_normal_(self.a_j.data, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, input, adj):
        """
        Forward function for the graph attention layer used in MAGIC

        Arguments:
            input (tensor): input of the graph attention layer [batch_size * N * in_features]
            adj (tensor): the learned communication graph (adjancy matrix) by the sub-scheduler [batch_size * N * N]

        Return:
            the output of the graph attention layer
        """

        # perform linear transformation on the input, and generate multiple heads
        # self.W: [in_features * (num_heads*out_features)]
        # h (tensor): the matrix after performing the linear transformation [batch_size * N * num_heads * out_features]
        N = self.num_agents
        h = torch.matmul(input, self.W).reshape(-1, N, self.num_heads, self.out_features)
    
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
        
        e = []

        # compute the unnormalized coefficients
        # a_i, a_j (tensors): weight vectors to compute the unnormalized coefficients [num_heads * out_features * 1]
        for head in range(self.num_heads):
            # coeff_i, coeff_j (tensors): intermediate matrices to calculate unnormalized coefficients 
            # [batch_size * N * out_features] @ [out_features * 1] --> [batch_size * N * 1]
            coeff_i = torch.matmul(h[:, :, head, :], self.a_i[head, :, :])
            coeff_j = torch.matmul(h[:, :, head, :], self.a_j[head, :, :])
            # coeff (tensor): the matrix of unnormalized coefficients for each head [batch_size * N * N * 1]
            coeff = coeff_i.expand(-1, -1, N) + coeff_j.transpose(1, 2).expand(-1, N, -1)
            coeff = coeff.unsqueeze(-1)
            
            e.append(coeff)
            
        # e (tensor): the matrix of unnormalized coefficients for all heads [batch_size * N * N * num_heads]
        # sometimes the unnormalized coefficients can be large, so regularization might be used 
        # to limit the large unnormalized coefficient values (TODO)
        e = self.leakyrelu(torch.cat(e, dim=-1)) 
            
        # adj: [batch_size * N * N * 1]
        adj = adj.unsqueeze(-1)
        # attention (tensor): the matrix of coefficients used for the message aggregation 
        # [batch_size * N * N * num_heads]
        attention = e * adj
        attention = F.softmax(attention, dim=2)
        # the weights from agents that should not communicate (send messages) will be 0, the gradients from 
        # the communication graph will be preserved in this way
        attention = attention * adj   
        # normalize: make the sum of weights from all agents be 1
        if self.normalize:
            if self.self_loop_type != 1:
                attention += 1e-15
            attention = attention / attention.sum(dim=2, keepdim=True)
            attention = attention * adj
        # dropout on the coefficients  
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # output (tensor): the matrix of output of the gat layer [N * (num_heads*out_features)]
        output = []
        for head in range(self.num_heads):
            # [batch_size * N * N] @ [batch_size * N * out_features] --> [batch_size * N * out_features]
            h_prime = torch.matmul(attention[:, :, :, head], h[:, :, head, :])
            output.append(h_prime)
        if self.average:
            # [batch_size * N * out_features * num_heads] --> [batch_size * N * out_features]
            output = torch.mean(torch.stack(output, dim=-1), dim=-1)
        else:
            # [batch_size * N * (out_features*num_heads)]
            output = torch.cat(output, dim=-1)
        
        if self.bias is not None:
            output += self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(in_features={}, out_features={})'.format(self.in_features, self.out_features)
    