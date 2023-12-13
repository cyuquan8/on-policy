import torch
import torch_geometric
import numpy as np
import torch.nn as nn
import torch_geometric.nn as gnn
from onpolicy.algorithms.utils.nn import (
    DNAGATv2Block,
    GAINBlock,
    GATv2Block,
    GINBlock, 
    GNNAllLayers,
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
dna_gatv2_model_norm_type = 'graphnorm' # 'none'
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
print("GATV2 MODEL:")
print("------------")

gatv2_model_norm_type = 'graphnorm' # 'none'
gatv2_model = GNNAllLayers(
	input_channels=2, 
	block=GATv2Block,
	output_channels=[5 for _ in range(2)], 
	n_gnn_fc_layers=2,
	heads=2,
    concat=False,
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
gin_model = GNNAllLayers(
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
gain_model = GNNAllLayers(
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