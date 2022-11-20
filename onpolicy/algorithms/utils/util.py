import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def complete_graph_edge_index(num_nodes):
    """ 
    function to generate the edge index of a complete graph given the number of nodes 
    """
    # empty list to store edge index
    edge_index = []

    # iterate over num_nodes
    for i in range(num_nodes):
        # iterate over num_nodes
        for j in range(num_nodes):
            # append edge index
            edge_index.append([i, j])

    return np.array(edge_index)