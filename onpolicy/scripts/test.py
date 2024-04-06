import torch
import torch_geometric
import numpy as np
import torch.nn as nn
import torch_geometric.nn as gnn
from onpolicy.algorithms.utils.nn import (
    DNAGATv2Block,
    GAINBlock,
    GATBlock,
    GATv2Block,
    GCNBlock,
    GINBlock, 
    GNNConcatAllLayers,
    GNNDNALayers, 
    MLPBlock, 
    NNLayers
)
from onpolicy.algorithms.utils.util import complete_graph_edge_index
from torch_geometric.data import Data, Batch

print("-----------------------")
print("UTILITY FUNCTIONS TEST:")
print("-----------------------")
num_nodes = 3
complete_graph_edge_index_np= complete_graph_edge_index(num_nodes)
complete_graph_edge_index_tensor = torch.tensor(complete_graph_edge_index_np, dtype=torch.long).t().contiguous()
print(f"complete_graph_edge_index_np for {num_nodes} nodes: {complete_graph_edge_index_np}")
print(f"complete_graph_edge_index_np.shape for {num_nodes} nodes: {complete_graph_edge_index_np.shape}")
print(f"complete_graph_edge_index_tensor for {num_nodes} nodes: {complete_graph_edge_index_tensor}")
print(f"complete_graph_edge_index_tensor.size() for {num_nodes} nodes: {complete_graph_edge_index_tensor.size()}")
print("NOTE: Includes self-loops!")

print("------------------")
print("DATA INFOROMATION:")
print("------------------")
x_1 = torch.tensor([[-1, -1], [0, 0], [1, 1]], dtype=torch.float)
x_2 = torch.tensor([[-2, -2], [0, 0], [2, 2]], dtype=torch.float)
x_3 = torch.tensor([[-3, -3], [0, 0], [3, 3]], dtype=torch.float)
x_4 = torch.tensor([[-4, -4], [0, 0], [4, 4]], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
data_1 = Data(x=x_1, edge_index=edge_index)
data_2 = Data(x=x_2, edge_index=edge_index)
data_3 = Data(x=x_3, edge_index=edge_index)
data_4 = Data(x=x_4, edge_index=edge_index)
print(f"edge_index: {edge_index}")
print(f"edge_index.size(): {edge_index.size()}")
print(f"data_1: {data_1}")
print(f"data_1.x: {data_1.x}")
print(f"data_1.x.size(): {data_1.x.size()}")

print("------------------------")
print("BATCH DATA INFOROMATION:")
print("------------------------")
batch_data = Batch.from_data_list([data_1, data_2, data_3, data_4])
batch = torch.arange(batch_data.num_graphs).repeat_interleave(batch_data.x.size(0) // batch_data.num_graphs)
print(f"batch_data: {batch_data}")
print(f"batch_data.x: {batch_data.x}")
print(f"batch_data.x.size(): {batch_data.x.size()}")
print(f"batch_data.edge_index: {batch_data.edge_index}")
print(f"batch_data.edge_index.size(): {batch_data.edge_index.size()}")
print(f"batch_data.num_graphs: {batch_data.num_graphs}")
print(f"batch: {batch}")
print(f"batch.size(): {batch.size()}")

print("-----------------------------")
print("KNN EDGE INDEX ON BATCH DATA:")
print("-----------------------------")
k=2
data_knn_edge_index = torch_geometric.nn.knn_graph(x=data_1.x, k=k, loop=True)
batch_knn_edge_index = torch_geometric.nn.knn_graph(x=batch_data.x, k=k, batch=batch, loop=True)
print(f"knn_edge_index for k={k} for data_1: {data_knn_edge_index}")
print(f"knn_edge_index.size() for k={k} for data_1: {data_knn_edge_index.size()}")
print(f"knn_edge_index for k={k} for batch_data: {batch_knn_edge_index}")
print(f"knn_edge_index.size() for k={k} for batch_data: {batch_knn_edge_index.size()}")
print("NOTE: Includes self-loops!")

print("----------")
print("MLP MODEL:")
print("----------")
mlp_model_norm_type = 'graphnorm' # 'none' 'layernorm' 'batchnorm1d' 'graphnorm'
mlp_model = NNLayers(
	input_channels=2, 
	block=MLPBlock, 
	output_channels=[5 for i in range(2)],
	norm_type=mlp_model_norm_type, 
	activation_func='relu', 
	dropout_p=0, 
	weight_initialisation='default'
)

if mlp_model_norm_type == 'graphnorm':
	mlp_model_output = mlp_model(x=batch_data.x, batch=batch)
else:
	mlp_model_output = mlp_model(x=batch_data.x)
print(f"mlp_model_output: {mlp_model_output}")
print(f"mlp_model_output.size(): {mlp_model_output.size()}")

print("----------------")
print("DNA GATV2 MODEL:")
print("----------------")
dna_gatv2_model_norm_type = 'graphnorm' # 'none' 'graphnorm'
dna_gatv2_model = GNNDNALayers(
	input_channels=2, 
	block=DNAGATv2Block, 
	output_channels=[2 for _ in range(2)],
	att_heads=2,
	mul_att_heads=2,
	gnn_cpa_model='none',
	norm_type=dna_gatv2_model_norm_type 
)

if dna_gatv2_model_norm_type == 'graphnorm':
	dna_gatv2_model_output = dna_gatv2_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, return_attention_weights=None, batch=batch)
else:
	dna_gatv2_model_output = dna_gatv2_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, return_attention_weights=None)
print(f"dna_gatv2_model_output (return_attention_weights == None): {dna_gatv2_model_output}")
print(f"dna_gatv2_model_output.size() (return_attention_weights == None): {dna_gatv2_model_output.size()}")
print("\n")

if dna_gatv2_model_norm_type == 'graphnorm':
	dna_gatv2_model_output, dna_gatv2_model_extra_output_list = dna_gatv2_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, return_attention_weights=True, batch=batch)
else:
	dna_gatv2_model_output, dna_gatv2_model_extra_output_list = dna_gatv2_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, return_attention_weights=True)
print(f"dna_gatv2_model_output (return_attention_weights == True): {dna_gatv2_model_output}")
print(f"dna_gatv2_model_output.size() (return_attention_weights == True): {dna_gatv2_model_output.size()}")
print(f"dna_gatv2_model_extra_output_list: {dna_gatv2_model_extra_output_list}")
for i, tup in enumerate(dna_gatv2_model_extra_output_list):
	print(f"layer {i} edge_index in dna_gatv2_model_extra_output_list: {tup[0]}")
	print(f"layer {i} edge_index.size() in dna_gatv2_model_extra_output_list: {tup[0].size()}")
	print(f"layer {i} alpha in dna_gatv2_model_extra_output_list: {tup[1]}")
	print(f"layer {i} alpha.size() in dna_gatv2_model_extra_output_list: {tup[1].size()}")

print("------------")
print("GCN MODEL:")
print("------------")

gcn_model_norm_type = 'graphnorm' # 'none' 'graphnorm'
gcn_model = GNNConcatAllLayers(
	input_channels=2, 
	block=GCNBlock,
	output_channels=[5 for _ in range(2)], 
	n_gnn_fc_layers=2,
	norm_type=gcn_model_norm_type
)

if gcn_model_norm_type == 'graphnorm':
	gcn_model_output = gcn_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_weight=None, batch=batch)
else:
	gcn_model_output = gcn_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_weight=None)
print(f"gcn_model_output: {gcn_model_output}")
print(f"gcn_model_output.size(): {gcn_model_output.size()}")

print("------------")
print("GAT MODEL:")
print("------------")

gat_model_norm_type = 'graphnorm' # 'none' 'graphnorm'
gat_model = GNNConcatAllLayers(
	input_channels=2, 
	block=GATBlock,
	output_channels=[5 for _ in range(2)], 
	n_gnn_fc_layers=2,
	heads=2,
    concat=True,
    gnn_cpa_model='none',
	norm_type=gat_model_norm_type
)

if gat_model_norm_type == 'graphnorm':
	gat_model_output = gat_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, size=None, return_attention_weights=None, batch=batch)
else:
	gat_model_output = gat_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, size=None, return_attention_weights=None)
print(f"gat_model_output (return_attention_weights == None): {gat_model_output}")
print(f"gat_model_output.size() (return_attention_weights == None): {gat_model_output.size()}")
print("\n")

if gat_model_norm_type == 'graphnorm':
	gat_model_output, gat_model_extra_output_list = gat_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, size=None, return_attention_weights=True, batch=batch)
else:
	gat_model_output, gat_model_extra_output_list = gat_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, size=None, return_attention_weights=True)
print(f"gat_model_output (return_attention_weights == True): {gat_model_output}")
print(f"gat_model_output.size() (return_attention_weights == True): {gat_model_output.size()}")
print(f"gat_model_extra_output_list: {gat_model_extra_output_list}")
for i, tup in enumerate(gat_model_extra_output_list):
	print(f"layer {i} edge_index in gat_model_extra_output_list: {tup[0]}")
	print(f"layer {i} edge_index.size() in gat_model_extra_output_list: {tup[0].size()}")
	print(f"layer {i} alpha in gat_model_extra_output_list: {tup[1]}")
	print(f"layer {i} alpha.size() in gat_model_extra_output_list: {tup[1].size()}")

print("------------")
print("GATV2 MODEL:")
print("------------")

gatv2_model_norm_type = 'graphnorm' # 'none' 'graphnorm'
gatv2_model = GNNConcatAllLayers(
	input_channels=2, 
	block=GATv2Block,
	output_channels=[5 for _ in range(2)], 
	n_gnn_fc_layers=2,
	heads=2,
    concat=True,
    gnn_cpa_model='none',
	norm_type=gatv2_model_norm_type
)

if gatv2_model_norm_type == 'graphnorm':
	gatv2_model_output = gatv2_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, return_attention_weights=None, batch=batch)
else:
	gatv2_model_output = gatv2_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, return_attention_weights=None)
print(f"gatv2_model_output (return_attention_weights == None): {gatv2_model_output}")
print(f"gatv2_model_output.size() (return_attention_weights == None): {gatv2_model_output.size()}")
print("\n")

if gatv2_model_norm_type == 'graphnorm':
	gatv2_model_output, gatv2_model_extra_output_list = gatv2_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, return_attention_weights=True, batch=batch)
else:
	gatv2_model_output, gatv2_model_extra_output_list = gatv2_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, return_attention_weights=True)
print(f"gatv2_model_output (return_attention_weights == True): {gatv2_model_output}")
print(f"gatv2_model_output.size() (return_attention_weights == True): {gatv2_model_output.size()}")
print(f"gatv2_model_extra_output_list: {gatv2_model_extra_output_list}")
for i, tup in enumerate(gatv2_model_extra_output_list):
	print(f"layer {i} edge_index in gatv2_model_extra_output_list: {tup[0]}")
	print(f"layer {i} edge_index.size() in gatv2_model_extra_output_list: {tup[0].size()}")
	print(f"layer {i} alpha in gatv2_model_extra_output_list: {tup[1]}")
	print(f"layer {i} alpha.size() in gatv2_model_extra_output_list: {tup[1].size()}")

print("----------")
print("GIN MODEL:")
print("----------")

gin_model_norm_type = 'graphnorm' # 'none' 'layernorm' 'batchnorm1d' 'graphnorm'
gin_model = GNNConcatAllLayers(
	input_channels=2, 
	block=GINBlock,
	output_channels=[5 for _ in range(2)], 
	n_gnn_fc_layers=2,
	train_eps=False,
	norm_type=gin_model_norm_type
)

if gin_model_norm_type == 'graphnorm':
	gin_model_output = gin_model(x=batch_data.x, edge_index=batch_data.edge_index, size=None, batch=batch)
else:
	gin_model_output = gin_model(x=batch_data.x, edge_index=batch_data.edge_index, size=None)
print(f"gin_model_output: {gin_model_output}")
print(f"gin_model_output.size(): {gin_model_output.size()}")

print("-----------")
print("GAIN MODEL:")
print("-----------")

gain_model_norm_type = 'graphnorm' # 'none' 'layernorm' 'batchnorm1d' 'graphnorm'
gain_model = GNNConcatAllLayers(
	input_channels=2, 
	block=GAINBlock,
	output_channels=[5 for _ in range(2)],
	n_gnn_fc_layers=2,
	heads=2,
	concat=True,
	norm_type=gain_model_norm_type
)

if gain_model_norm_type == 'graphnorm':
	gain_model_output = gain_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, return_attention_weights=None, batch=batch)
else:
	gain_model_output = gain_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, return_attention_weights=None)
print(f"gain_model_output (return_attention_weights == None): {gain_model_output}")
print(f"gain_model_output.size() (return_attention_weights == None): {gain_model_output.size()}")
print("\n")

if gain_model_norm_type == 'graphnorm':
	gain_model_output, gain_model_extra_output_list = gain_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, return_attention_weights=True, batch=batch)
else:
	gain_model_output, gain_model_extra_output_list = gain_model(x=batch_data.x, edge_index=batch_data.edge_index, edge_attr=None, return_attention_weights=True)
print(f"gain_model_output (return_attention_weights == True): {gain_model_output}")
print(f"gain_model_output.size() (return_attention_weights == True): {gain_model_output.size()}")
print(f"gain_model_extra_output_list: {gain_model_extra_output_list}")
for i, tup in enumerate(gain_model_extra_output_list):
	print(f"layer {i} edge_index in gain_model_extra_output_list: {tup[0]}")
	print(f"layer {i} edge_index.size() in gain_model_extra_output_list: {tup[0].size()}")
	print(f"layer {i} alpha in gain_model_extra_output_list: {tup[1]}")
	print(f"layer {i} alpha.size() in gain_model_extra_output_list: {tup[1].size()}")

# mini_batch_size = 5
# data_chunk_length = 10
# num_agents = 3
# hidden_size = 4

# masks = torch.randint(low=0, high=10, size=(mini_batch_size, data_chunk_length, num_agents, 1))
# print(f"masks.shape: {masks.shape}")
# has_zeros = (masks[:, 1:, 0].squeeze() == 0.0).any(dim=0).nonzero().squeeze()
# print(f"has_zeros: {has_zeros}")
# print(f"has_zeros.shape: {has_zeros.shape}")	
# # +1 to correct the masks[1:]
# if has_zeros.dim() == 0:
#     # Deal with scalar
#     has_zeros = [has_zeros.item() + 1]
# else:
#     has_zeros = (has_zeros + 1).numpy().tolist()
# for i in range(len(has_zeros)):
# 	print(masks[:, has_zeros[i], 0])

# # find steps in sequence (data_chunk_length) with any zero for mask for the agent across mini_batch_size
# # t=0 is requires application of mask by default
# # (masks[:, 1:, i].squeeze() == 0.0).any(dim=0).nonzero().squeeze().cpu() 
# # [shape: (mini_batch_size, data_chunk_length, num_agents, 1)] --> 
# # [shape: (mini_batch_size, data_chunk_length - 1)] --> [shape: (data_chunk_length - 1,)] -->
# # [shape: (data_chunk_length - 1, 1)] --> [shape: (data_chunk_length - 1, )]
# has_zeros = (masks[:, 1:, i].squeeze() == 0.0).any(dim=0).nonzero().squeeze().cpu()
# # account for indexing when t=0 is not included
# if has_zeros.dim() == 0:
#     # scalar
#     has_zeros = [has_zeros.item() + 1]
# else:
#     has_zeros = (has_zeros + 1).numpy().tolist()
# # add t=0 and t=data_chunk_length to the list
# has_zeros = [0] + has_zeros + [T]
# for i in range(len(has_zeros) - 1):
#     start_idx = has_zeros[i]
#     end_idx = has_zeros[i + 1]

# h = torch.arange(mini_batch_size * num_agents * hidden_size).reshape(mini_batch_size * num_agents, 1, hidden_size)
# print(f"h: {h}")
# print(f"h.shape: {h.shape}")
# h_hard_1 = h.reshape(mini_batch_size, num_agents, 1, hidden_size).repeat(1, 1, num_agents, 1)
# print(f"h_hard_1: {h_hard_1}")
# print(f"h_hard_1.shape: {h_hard_1.shape}")
# h_hard_2 = h.reshape(mini_batch_size, 1, num_agents, hidden_size).repeat(1, num_agents, 1,  1)
# print(f"h_hard_2: {h_hard_2}")
# print(f"h_hard_2.shape: {h_hard_2.shape}")
# h_hard = torch.cat((h_hard_1, h_hard_2), dim=-1) 
# print(f"h_hard: {h_hard}")
# print(f"h_hard.shape: {h_hard.shape}")
# # mask = (1 - torch.eye(num_agents)).unsqueeze(0).repeat(mini_batch_size, 1, 1).unsqueeze(-1)
# mask = (1 - torch.eye(num_agents)).unsqueeze(0).unsqueeze(-1).repeat(mini_batch_size, 1, 1, hidden_size * 2)
# print(f"mask: {mask}")
# print(f"mask.shape: {mask.shape}")
# h_hard_masked = h_hard[mask == 1]
# h_hard_masked = h_hard_masked.reshape(mini_batch_size * num_agents, num_agents - 1, hidden_size * 2)
# print(f"h_hard_masked: {h_hard_masked}")
# print(f"h_hard_masked.shape: {h_hard_masked.shape}")

# q = torch.arange(mini_batch_size * num_agents * hidden_size).reshape(mini_batch_size * num_agents, 1, hidden_size)
# k = torch.arange(mini_batch_size * num_agents * hidden_size).reshape(mini_batch_size * num_agents, 1, hidden_size)

# # q = q.reshape(mini_batch_size, num_agents, 1, hidden_size).repeat(1, 1, num_agents, 1)
# q = q.reshape(mini_batch_size, num_agents, 1, hidden_size)
# k = k.reshape(mini_batch_size, 1, num_agents, hidden_size).repeat(1, num_agents, 1,  1)
# print(f"q: {q}")
# print(f"q.shape: {q.shape}")
# print(f"k: {k}")
# print(f"k.shape: {k.shape}")
# mask = (1 - torch.eye(num_agents)).unsqueeze(0).unsqueeze(-1).repeat(mini_batch_size, 1, 1, hidden_size)
# print(f"mask: {mask}")
# print(f"mask.shape: {mask.shape}")
# k_masked = k[mask == 1]
# k_masked = k_masked.reshape(mini_batch_size, num_agents, num_agents - 1, hidden_size)
# print(f"k_masked: {k_masked}")
# print(f"k_masked.shape: {k_masked.shape}")
# k_perm = k_masked.transpose(2, 3)
# print(f"k_perm: {k_perm}")
# print(f"k_perm.shape: {k_perm.shape}")
# score = torch.matmul(q, k_perm)
# print(f"score: {score}")
# print(f"score.shape: {score.shape}")
# scaled_score = score / np.sqrt(hidden_size)
# print(f"scaled_score: {scaled_score}")
# print(f"scaled_score.shape: {scaled_score.shape}")
# soft_weight = nn.functional.softmax(scaled_score, dim=-1)
# print(f"soft_weight: {soft_weight}")
# print(f"soft_weight.shape: {soft_weight.shape}")