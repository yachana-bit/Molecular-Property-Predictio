import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool

from config import NUM_NODE_FEATURES, DROPOUT_P


class GCN(nn.Module):
    """
    3-layer Graph Convolutional Network with global mean pooling.
    Outputs a single regression value per graph.

    Args:
        dim_h (int): hidden layer dimension
    """

    def __init__(self, dim_h):
        super().__init__()
        self.conv1 = GCNConv(NUM_NODE_FEATURES, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin   = nn.Linear(dim_h, 1)

    def forward(self, data):
        x, e = data.x, data.edge_index

        x = self.conv1(x, e).relu()
        x = self.conv2(x, e).relu()
        x = self.conv3(x, e)

        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=DROPOUT_P, training=self.training)
        x = self.lin(x)
        return x


class GIN(nn.Module):
    """
    3-layer Graph Isomorphism Network with global add pooling.
    Uses MLP aggregators (BatchNorm + ReLU) for stronger expressiveness.
    Outputs a single regression value per graph.

    Args:
        dim_h (int): hidden layer dimension
    """

    def __init__(self, dim_h):
        super().__init__()

        def mlp(in_dim, out_dim):
            return Sequential(
                Linear(in_dim, out_dim),
                BatchNorm1d(out_dim),
                ReLU(),
                Linear(out_dim, out_dim),
                ReLU(),
            )

        self.conv1 = GINConv(mlp(NUM_NODE_FEATURES, dim_h))
        self.conv2 = GINConv(mlp(dim_h, dim_h))
        self.conv3 = GINConv(mlp(dim_h, dim_h))
        self.lin1  = Linear(dim_h, dim_h)
        self.lin2  = Linear(dim_h, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index)

        h = global_add_pool(h, batch)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=DROPOUT_P, training=self.training)
        h = self.lin2(h)
        return h