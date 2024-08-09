import numpy as np
import torch
import math
from model.sgn_module import embedding_module, lstm_module
from collections import defaultdict
from torch import nn
from torch_geometric.nn import GATConv
import dgl
from model.structural_aggregator import TreeAggregator
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
        graph_leaf = dgl.graph((source_nodes, destination_nodes)).to(self.device)
        graph_leaf.ndata['x'] = x
        graph_leaf.ndata['mask'] = (graph_leaf.out_degrees() == 0).float().unsqueeze(dim=-1).clone().detach().requires_grad_(True)
        graph_root = dgl.reverse(graph_leaf, copy_ndata=True, copy_edata=True)
        graph_root.ndata['mask'] = (graph_root.out_degrees() == 0).float().unsqueeze(dim=-1).clone().detach().requires_grad_(True)
        graph_root, graph_leaf = graph_root.to(self.device), graph_leaf.to(self.device)
        root_emb, leaf_emb = self.structural_agg_root(graph_root), self.structural_agg_leaf(graph_leaf)

        return root_emb, leaf_emb
