import numpy as np
import torch
import math
from model.sgn_module import embedding_module, lstm_module
from collections import defaultdict
from torch import nn
from torch_geometric.nn import GATConv
import dgl
from model.structural_aggregator import TreeAggregator
import warnings
# class SGN(torch.nn.Module):
#     def __init__(self, device, n_layers=2,
#                  n_heads=2, dropout=0.1):
#         super(SGN, self).__init__()
#
#         self.n_layers = n_layers
#         self.device = device
#         self.node_raw_features = None
#         self.dropout = dropout
#         self.n_heads = n_heads
#         self.n_node_features = 768
#         self.embedding_dimension = self.n_node_features
#         self.embedding_lstm = lstm_module(device=self.device)
#         self.embedding_module = embedding_module(node_features=self.node_raw_features,
#                                                  n_layers=self.n_layers,
#                                                  device=self.device,
#                                                  n_heads=self.n_heads, dropout=self.dropout,
#                                                  )  # .to(self.device)
#         self.structure_info = None
#
#     def compute_stru_embeddings(self, source_feature, neighbor_features):
#
#         sub_graph_embedding = self.embedding_module.compute_embedding(source_feature, neighbor_features)
#         return sub_graph_embedding
#
#     def compute_lstm_embeddings(self, input, hidden, cell):
#
#         output, n_hidden, n_cell = self.embedding_lstm.compute_embedding(input, hidden, cell)
#         return output, n_hidden, n_cell
#
#     def forward(self, source_nodes, destination_nodes, node_raw_features):
#
#         self.node_raw_features = (torch.from_numpy(node_raw_features.astype(np.float)).to(
#             self.device)).float()  # dtype=torch.float64
#         self.n_nodes = self.node_raw_features.shape[0]
#
#         # self.updated_embedding = torch.from_numpy(np.zeros((self.n_nodes, 768))).to(self.device)
#         self.structure_info = torch.from_numpy(np.zeros((self.n_nodes, 768))).to(self.device)
#         mock_info = torch.from_numpy(np.zeros((self.n_nodes, 768))).to(self.device)
#         struc1_cell = torch.from_numpy(np.zeros((self.n_nodes, 768))).to(self.device)
#         struc1_hide = torch.from_numpy(np.zeros((self.n_nodes, 768))).to(self.device)
#         struc2_cell = torch.from_numpy(np.zeros((self.n_nodes, 768))).to(self.device)
#         struc2_hide = torch.from_numpy(np.zeros((self.n_nodes, 768))).to(self.device)
#         neighbor_features = self.neighbor_finder(source_nodes, destination_nodes, node_raw_features)
#         # print((torch.from_numpy(np.array(neighbor_features[1]).astype(np.float)).to(
#         #     self.device)).float().shape)
#         root_extend = self.node_raw_features[1].repeat(len(self.node_raw_features), 1).to(self.device)
#         pa_dict = {}
#         for ans in range(len(destination_nodes)):
#             pa_dict[destination_nodes[ans]] = source_nodes[ans]
#         #
#         mock_info[1] = self.node_raw_features[1]
#         for i in range(self.n_nodes):
#             if i != 0:
#                 if i != 1:
#                     par = pa_dict[i]
#                     mock_info[i] = self.compute_stru_embeddings(self.node_raw_features[i], mock_info[par].float())
#                     mock_info[i], struc1_hide[i], struc1_cell[i] = self.compute_lstm_embeddings(mock_info[i],
#                                                                                                 struc1_hide[par],
#                                                                                                 struc1_cell[par])
#         mock_info = torch.cat((mock_info.float(), root_extend.float()), dim=1)
#         embedding_struc1 = mock_info[self.n_nodes - 1]
#         for i in range(self.n_nodes - 1):
#             if i == 0:
#                 continue
#             now = self.n_nodes - i - 1
#             childs = [k for k, v in pa_dict.items() if v == now]
#             if not childs:
#                 embedding_struc1 += mock_info[now]
#         for i in range(self.n_nodes):
#             # print('ids:', (start_idx+i), i)
#             now = self.n_nodes - i - 1
#             childs = [k for k, v in pa_dict.items() if v == now]
#
#             if not childs:
#                 self.structure_info[now] = self.node_raw_features[now]
#             else:
#                 self.structure_info[now] = self.compute_stru_embeddings(self.node_raw_features[now],
#                                                                         self.structure_info[childs, :].float())
#                 self.structure_info[now], struc2_hide[now], struc2_cell[now] = self.compute_lstm_embeddings(
#                     self.structure_info[now],
#                     torch.mean(struc2_hide[childs, :], dim=0),
#                     torch.mean(struc2_cell[childs, :], dim=0))
#         self.structure_info = torch.cat((self.structure_info.float(), root_extend.float()), dim=1)
#         #     if neighbor_features[i]==[] :
#         #         self.structure_info[i] = self.node_raw_features[i]
#         #     else:
#         #         self.structure_info[i] = self.compute_temporal_embeddings(self.node_raw_features[i], torch.tensor(neighbor_features[i]).to(self.device).float())
#         #
#         # # torch.cuda.empty_cache()
#         # self.structure_info = torch.cat((self.structure_info.float(), self.node_raw_features.float()), dim=1)
#         return embedding_struc1, self.structure_info[1]
#
#     def neighbor_finder(self, source_nodes, destination_nodes, node_raw_features):
#         neighbor_features = []
#         pa_dict = {}
#         for ans in range(len(destination_nodes)):
#             pa_dict[destination_nodes[ans]] = source_nodes[ans]
#         for i in range(self.n_nodes):
#             n_neighbor_features = []
#
#             if i != 0:
#                 if i != 1:
#                     n_neighbor_features.append(node_raw_features[pa_dict[i]])
#                 key = [k for k, v in pa_dict.items() if v == i]
#                 if key != []:
#                     for k in key:
#                         n_neighbor_features.append(node_raw_features[k])
#             neighbor_features.append(n_neighbor_features)
#         return neighbor_features
class SGN(torch.nn.Module):
    def __init__(self, device, n_layers=2, n_heads=2, dropout=0.1):
        super(SGN, self).__init__()

        self.n_layers = n_layers
        self.device = device
        self.node_raw_features = None
        self.dropout = dropout
        self.n_heads = n_heads
        self.n_node_features = 768
        self.embedding_dimension = self.n_node_features
        self.gat_layer = GATConv(self.embedding_dimension, self.embedding_dimension, heads=n_heads)
        self.fc = nn.Linear(self.embedding_dimension * n_heads, self.embedding_dimension)
        self.embedding_lstm = lstm_module(device=self.device)
        self.structural_agg_root = TreeAggregator(self.embedding_dimension, self.embedding_dimension, device, edge_time=True)
        self.structural_agg_leaf = TreeAggregator(self.embedding_dimension, self.embedding_dimension, device, edge_time=True)

    def compute_lstm_embeddings(self, input, hidden, cell):
        output, n_hidden, n_cell = self.embedding_lstm.compute_embedding(input, hidden, cell)
        return output, n_hidden, n_cell

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        n_nodes = len(x)
        destination_nodes = data.destinations
        source_nodes = data.sources
        destination_nodes = [x - 1 for x in destination_nodes]
        source_nodes = [x - 1 for x in source_nodes]
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        # print(destination_nodes)
        # print(source_nodes)
        graph_leaf = dgl.graph((source_nodes, destination_nodes)).to(self.device)
        graph_leaf.ndata['x'] = x
        # graph_leaf.edata['time'] = torch.tensor(data.timestamps, dtype=torch.float).to(self.device)
        # graph_leaf.add_edges(source_nodes, destination_nodes)
        # num_nodes = graph_leaf.number_of_nodes()
        # num_edges = graph_leaf.number_of_edges()
        # print("Number of nodes:", num_nodes)
        # print("Number of edges:", num_edges)
        graph_leaf.ndata['mask'] = (graph_leaf.out_degrees() == 0).float().unsqueeze(dim=-1).clone().detach().requires_grad_(True)
        graph_root = dgl.reverse(graph_leaf, copy_ndata=True, copy_edata=True)
        graph_root.ndata['mask'] = (graph_root.out_degrees() == 0).float().unsqueeze(dim=-1).clone().detach().requires_grad_(True)
        graph_root, graph_leaf = graph_root.to(self.device), graph_leaf.to(self.device)
        # num_edges = graph_root.number_of_edges()
        # print(num_edges)
        root_emb, leaf_emb = self.structural_agg_root(graph_root), self.structural_agg_leaf(graph_leaf)
        # root_emb, leaf_emb = self.structural_agg_root(graph_root)
        return root_emb, leaf_emb
