import torch
from torch import nn
from torch_geometric import nn as tg_nn


class Encoder_v(nn.Module):
    """x vector -> v vector 128

    x.shape == Nx6x2
    v.shape == N x 128
    """
    def __init__(self, c=5):
        super().__init__() # from python 3.7
        self.l1 = nn.Linear(2 * (c + 1), 32)
        self.l2 = nn.Linear(32, 64)
        self.l3 = nn.Linear(64, 128)

        e0 = torch.nn.Parameter(torch.rand(128))
        u0 = torch.nn.Parameter(torch.rand(128))

    def forward(self, x):
        # Linearize x vector Nx6x2 -> Nx12
        x = x.flatten(1)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        return x, self.e0, self.u0


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


def get_adjacency_matrix(position, r):
    """
    Return an intersection matrix based on particle nearer than R
    """
    last_position = position[:, -1, :]
    A = tg_nn.radius_graph(last_position, r)
    return A



if __name__ == "__main__":
    N = 2
    position = torch.randn(N, 6, 2)
    print(position)
    # print(position2x(position))
    r = 2
    print(get_adjacency_matrix(position, r))




