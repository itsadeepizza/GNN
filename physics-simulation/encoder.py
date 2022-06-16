import torch
from torch import nn
from torch_geometric import nn as tg_nn
from torch_geometric.data import Data
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    """positions -> G graph

    x.shape == Nx6x2
    v.shape == N x 128
    """
    def __init__(self, c=5):
        super().__init__() # from python 3.7
        self.l1 = nn.Linear(2 * (c + 1), 32)
        self.l2 = nn.Linear(32, 64)
        self.l3 = nn.Linear(64, 128)

        self.r = 2

        self.e0 = torch.nn.Parameter(torch.rand(128))
        # self.u0 = torch.nn.Parameter(torch.rand(128))

    def forward(self, position):
        # Linearize x vector Nx6x2 -> Nx12
        x = self.position2x(position)
        x = x.flatten(1)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        v = self.l3(x)

        edge_index = self.get_adjacency_matrix(position, self.r)
        # a tensor of size N x 128 defined as N times e0

        Ne_ones = torch.ones(edge_index.shape[1], 1)
        edge_attr = torch.kron(Ne_ones, self.e0)

        data = Data(x=v, edge_index=edge_index, edge_attr=edge_attr)
        return data

    @staticmethod
    def position2x(position):
        """
        Convert position vectors to x
        position_i = (p_i^tk-C, , ..., p_i^tk-1, p_i^tk)
        x_i        = (p_i^tk, p'_i^tk-C+1, ..., p'_i^tk)
        speed are calculated using difference i.e. p'_i^tk = p_i^tk - p_i^tk-1
        """
        speeds = position[:, 1:, :] - position[:, :-1,:]
        last_position = position[:, -1, :].unsqueeze(1)
        x = torch.cat([speeds, last_position], 1)
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



