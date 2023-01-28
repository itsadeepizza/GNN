import torch
from torch import nn
from torch_geometric import nn as tg_nn
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_geometric.utils import add_self_loops
from config import selected_config as conf


class Encoder(nn.Module):
    """positions -> G graph

    x.shape == Nx6x2
    v.shape == N x 128
    """
    def __init__(self, normalization_stats, bounds, device=conf.DEVICE, c=5, edge_features_dim=128,
                 node_features_dim=128, R=0.015):
        super().__init__() # from python 3.7
        self.device = device
        self.bounds = bounds
        self.l1 = nn.Linear(2 * (c + 1) + 4, 32, device=device)
        self.l2 = nn.Linear(32, 64, device=device)
        self.l3 = nn.Linear(64, node_features_dim, device=device)
        self.layer_norm = nn.LayerNorm(node_features_dim, device=self.device)

        self.edge_l1 = nn.Linear(7, 32, device=device)
        self.edge_l2 = nn.Linear(32, 64, device=device)
        self.edge_l3 = nn.Linear(64, edge_features_dim, device=device)
        self.edge_layer_norm = nn.LayerNorm(edge_features_dim, device=self.device)

        self.r = R

        self.normalization_stats = normalization_stats

        # A random learnable vector as e0 parameter
        # self.register_parameter(name='e0', param=torch.nn.Parameter(torch.rand(edge_features_dim, device=device)))
        # self.u0 = torch.nn.Parameter(torch.rand(128))

    def forward(self, position, batch_index) -> Data:
        # Linearize x vector Nx6x2 -> Nx12

        # Do this at the beginning before changing device
        edge_index = self.get_adjacency_matrix(position, batch_index, self.r)
        position = position.to(self.device)
        x = self.position2x(position)
        # x = x.flatten(1)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = self.layer_norm(x)



        # TODO: Self loops or not self loops ?
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # a tensor of size E x 2
        # Ne_ones = torch.ones(edge_index.shape[1], 1, device=self.device)
        # edge_attr = torch.kron(Ne_ones, self.e0)
        # a tensor of size E x 128
        delta_pos = position[edge_index[0, :], -1, :] - position[edge_index[1, :], -1, :]
        dist = torch.sqrt(delta_pos[:, 0]**2 + delta_pos[:, 1]**2).unsqueeze(1)
        last_speeds = position[:, -1, :] - position[:, -2, :]
        delta_speeds = (last_speeds[edge_index[0, :], :] - \
                       last_speeds[edge_index[1, :], :] ) / \
                            self.normalization_stats['velocity']['std']
        abs_speeds = torch.sqrt(delta_speeds[:, 0]**2 + delta_speeds[:, 1]**2).unsqueeze(1)
        relative_motion = (delta_speeds[:, 0] * delta_pos[:, 0] + delta_speeds[:, 1] * delta_pos[:,
                                                                                   1]).unsqueeze(
            1) / (abs_speeds + 1e-6) /  (dist + 1e-6)
        edge_attr = torch.cat((delta_pos, dist, delta_speeds, abs_speeds, relative_motion), dim=1)
        edge_attr = self.edge_l1(edge_attr)
        edge_attr = torch.relu(edge_attr)
        edge_attr = self.edge_l2(edge_attr)
        edge_attr = torch.relu(edge_attr)
        edge_attr = self.edge_l3(edge_attr)
        edge_attr = self.edge_layer_norm(edge_attr)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch_index=batch_index)
        return data

    def position2x(self, position):
        """
        Convert position vectors to x
        position_i = (p_i^tk-C, , ..., p_i^tk-1, p_i^tk)
        x_i        = (p_i^tk, p'_i^tk-C+1, ..., p'_i^tk)
        speed are calculated using difference i.e. p'_i^tk = p_i^tk - p_i^tk-1

        Add also distance from bounds to features
        """
        # 1 calculate (normalised) speeds
        speeds = position[:, 1:, :] - position[:, :-1,:]
        normalized_speeds = (speeds - self.normalization_stats['velocity']['mean']) / self.normalization_stats['velocity']['std']
        # 2 Calculate last position
        last_position = position[:, -1, :].unsqueeze(1)
        # Calculate distance from bounds
        x_position = position[:, -1, 0].repeat(2, 1).swapaxes(0, 1)
        x_bounds = self.bounds[0,:].repeat(len(position), 1)
        x_bounds_distance = torch.clip(x_position - x_bounds, -1, 1)
        y_position = position[:, -1, 1].repeat(2, 1).swapaxes(0, 1)
        y_bounds = self.bounds[1,:].repeat(len(position), 1)
        y_bounds_distance = torch.clip(y_position - y_bounds, -1, 1)
        x = torch.cat([normalized_speeds.flatten(1), last_position.flatten(1), x_bounds_distance, y_bounds_distance], 1)
        return x

    @staticmethod
    def get_adjacency_matrix(position, batch_index, r, max_neigh=conf.MAX_NEIGH):
        """
        Return an intersection matrix based on particle nearer than R
        """
        last_position = position[:, -1, :]
        # TODO: For some hardware configuration, GPU does not work
        A = tg_nn.radius_graph(last_position.to('cpu'), r, batch=batch_index.to('cpu'),
                               max_num_neighbors=max_neigh,
                               loop=True)
        return A.to(conf.DEVICE)



