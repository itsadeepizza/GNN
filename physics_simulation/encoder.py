import torch
from torch import nn
from torch_geometric import nn as tg_nn
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_geometric.utils import add_self_loops


class Encoder(nn.Module):
    """positions -> G graph

    x.shape == Nx6x2
    v.shape == N x 128
    """
    def __init__(self, normalization_stats, bounds, device, c=5, edge_features_dim=128, node_features_dim=128, R=0.015):
        super().__init__() # from python 3.7
        self.device = device
        self.bounds = bounds
        self.l1 = nn.Linear(2 * (c + 1) + 4, 32, device=device)
        self.l2 = nn.Linear(32, 64, device=device)
        self.l3 = nn.Linear(64, node_features_dim, device=device)
        self.layer_norm = nn.LayerNorm(node_features_dim, device=self.device)

        self.r = R

        self.normalization_stats = normalization_stats

        # A random learnable vector as e0 parameter
        self.register_parameter(name='e0', param=torch.nn.Parameter(torch.rand(edge_features_dim, device=device)))
        # self.u0 = torch.nn.Parameter(torch.rand(128))

    def forward(self, position) -> Data:
        # Linearize x vector Nx6x2 -> Nx12
        x = self.position2x(position)
        # x = x.flatten(1)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = self.layer_norm(x)
        # a tensor of size N x 128 defined as N times e0
        # TODO: Add LayerNorm

        edge_index = self.get_adjacency_matrix(position, self.r)
        # TODO: Self loops or not self loops ?
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # a tensor of size E x 2
        Ne_ones = torch.ones(edge_index.shape[1], 1, device=self.device)
        edge_attr = torch.kron(Ne_ones, self.e0)
        # a tensor of size E x 128

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
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
    def get_adjacency_matrix(position, r):
        """
        Return an intersection matrix based on particle nearer than R
        """
        last_position = position[:, -1, :]
        A = tg_nn.radius_graph(last_position, r)
        return A



if __name__ == "__main__":
    N = 5
    position = torch.randn(N, 6, 2)
    # print(position)
    # print(position2x(position))
    r = 2
    model = Encoder()
    data = model(position)
    plt.scatter(x=position[:, -1, 0], y=position[:, -1, 1])
    plt.show()
    print(data)



