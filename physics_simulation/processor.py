import torch
from torch.nn import Linear, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data
from torch_geometric.nn.conv.cg_conv import CGConv
from torch_geometric.utils import add_self_loops, degree


from physics_simulation.new_GATconv import New_GATConv

# A stack of M GNs, each GN has 2 hidden layers + LayerNorm at the end, size is always 128 for all layers

class Processor(torch.nn.Module):
    def __init__(
            self,
            node_in,
            node_out,
            edge_in,
            edge_out,
            M,
            device):
        # Init parent
        super().__init__()
        self.M = M
        self.GNs = [GN(node_in, node_out, edge_in, edge_out, device=device) for i in range(self.M)]

    def all_parameters(self):
        return [param for GN in self.GNs for param in list(GN.parameters())]

    def forward(self, data: Data):
        for i in range(self.M):
            data = self.GNs[i](data)
        return data

def compute_norm(edge_index, x):
    # Step 3: Compute normalization.
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return norm


class GN(MessagePassing):
    def __init__(
            self,
            node_in,
            node_out,
            edge_in,
            edge_out,
            device
    ):
        super().__init__(aggr='add')
        self.device = device
        self.edge_fn1 = Linear(2 * node_in + edge_in, edge_out, device=self.device)
        self.edge_fn2 = Linear(2 * node_in + edge_in, edge_out, device=self.device)
        self.edge_fn = [self.edge_fn1, self.edge_fn2]

        self.node_fn1 = Linear(node_in + edge_out, node_out, device=self.device)
        self.node_fn2 = Linear(node_in + edge_out, node_out, device=self.device)
        self.node_fn = [self.node_fn1, self.node_fn2]

        self.node_layernorm = LayerNorm(node_out, device=self.device)
        self.edge_layernorm = LayerNorm(edge_out, device=self.device)

        self.edge_attr = None

    def forward(self, data):
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)

        x = data.x
        # TODO: Bisogna aggiungere i self loops ai edge_index ? oppure no ?
        #  E se si, poi bisogna togliergli alla fine? O bisognava metterli nell'encoder ?
        # edge_index, _ = add_self_loops(data.edge_index, num_nodes=x.size(0))
        edge_index = data.edge_index
        self.edge_attr = data.edge_attr

        x_residual = x
        edge_attr_residual = self.edge_attr

        for idx in [0, 1]:
            norm = compute_norm(edge_index, x)
            x = self.propagate(edge_index=edge_index, x=x, idx=idx, norm=norm)
            x = F.relu(x)
            self.edge_attr = F.relu(self.edge_attr)

        x = self.node_layernorm(x)
        self.edge_attr = self.edge_layernorm(self.edge_attr)

        x = x + x_residual
        self.edge_attr  = self.edge_attr + edge_attr_residual

        data = Data(x=x, edge_index=data.edge_index, edge_attr=self.edge_attr )
        return data

    def message(self, edge_index, x_i, x_j, idx, norm):
        self.edge_attr = torch.cat([x_i, x_j, self.edge_attr], dim=-1)
        self.edge_attr = self.edge_fn[idx](self.edge_attr)
        return self.edge_attr * norm.view(-1, 1)

    def update(self, x_updated, x, idx):
        # x_updated: (E, edge_out)
        # x: (E, node_in)
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn[idx](x_updated)
        return x_updated



if __name__ == "__main__":
    import encoder
    from decoder import Decoder
    from euler_integrator import integrator

    device = torch.device("cuda")

    encoderNN = encoder.Encoder(device=device)
    proc = Processor(128, 128, 128, 128, M=10, device=device)
    decoder = Decoder().to(device)

    N = 5
    position = torch.randn(N, 6, 2).to(device)
    # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_nn_conv.py
    data = encoderNN(position)
    print(data.edge_index)
    print(data.edge_attr)
    data = proc(data)
    print(data.edge_attr)
    acc = decoder(data)
    print(acc)
    last_pos = integrator(position, acc)
    print("last_pos", last_pos)