import torch
from torch.nn import Linear, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data
from torch_geometric.nn.conv.cg_conv import CGConv
from torch_geometric.utils import add_self_loops, degree
import torch.nn as nn

class Decoder(nn.Module):
    """
    Extract each node acceleration from the graph processed by the processor
    """
    def __init__(self, normalization_stats, node_features_dim=128):
        super().__init__() # from python 3.7
        self.l1 = nn.Linear(node_features_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 2)
        # I need 2 dimensional output for x,y coordinates of the acceleration
        self.normalization_stats = normalization_stats

    def forward(self, data: Data, denormalize=False):
        # Linearize x vector Nx6x2 -> Nx12
        x = data.x
        x = self.l1(x)
        x = torch.sigmoid(x)
        x = self.l2(x)
        x = torch.sigmoid(x)
        x = self.l3(x)
        if denormalize:
            x = self.denormalize(x)

        # x = torch.relu(x)
        return x

    def denormalize(self, norm_acceleration):
        return (norm_acceleration * self.normalization_stats['acceleration']['std']) + self.normalization_stats['acceleration']['mean']
