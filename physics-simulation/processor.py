import torch
from torch.nn import Linear, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data
from torch_geometric.nn.conv.cg_conv import CGConv
from torch_geometric.utils import add_self_loops, degree


from new_GATconv import New_GATConv

# A stack of M GNs, each GN has 2 hidden layers + LayerNorm at the end, size is always 128 for all layers

M = 10

class Processor(torch.nn.Module):
    def __init__(self):
        # Init parent
        super().__init__()

        # GCN layers
        # Syntax is GCNConv(channel_in, channel_out)
        self.hidden1 = CGConv(128, dim=128) #GATConv could perform convolution over edge attributes (multi-dim weights features)

    def forward(self, data: Data):
        out = self.hidden1(data.x, data.edge_index, data.edge_attr)

        return out

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
    ):
        super().__init__(aggr='add')
        self.edge_fn1 = Linear(2 * node_in + edge_in, edge_out)
        self.edge_fn2 = Linear(2 * node_in + edge_in, edge_out)
        self.edge_fn = [self.edge_fn1, self.edge_fn2]

        self.node_fn1 = Linear(node_in + edge_out, node_out)
        self.node_fn2 = Linear(node_in + edge_out, node_out)
        self.node_fn = [self.node_fn1, self.node_fn2]

        self.node_layernorm = LayerNorm(node_out)
        self.edge_layernorm = LayerNorm(edge_out)

    def forward(self, data):
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)

        x = data.x
        # TODO: Bisogna aggiungere i self loops ai edge_index ? oppure no ?
        #  E se si, poi bisogna togliergli alla fine? O bisognava metterli nell'encoder ?
        # edge_index, _ = add_self_loops(data.edge_index, num_nodes=x.size(0))
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x_residual = x
        edge_attr_residual = edge_attr

        norm = compute_norm(edge_index, x)
        x, edge_attr = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, idx=0, norm=norm)
        x = F.relu(x)
        edge_attr = F.relu(edge_attr)

        norm = compute_norm(edge_index, x)
        x, edge_attr = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, idx=1, norm=norm)
        x = F.relu(x)
        edge_attr = F.relu(edge_attr)

        x = self.node_layernorm(x)
        edge_attr = self.edge_layernorm(edge_attr)

        x = x + x_residual
        edge_attr = edge_attr + edge_attr_residual

        data = Data(x=x, edge_index=data.edge_index, edge_attr=edge_attr)
        return data

    def message(self, edge_index, x_i, x_j, edge_attr, idx, norm):
        edge_attr = torch.cat([x_i, x_j, edge_attr], dim=-1)
        edge_attr = self.edge_fn[idx](edge_attr)
        return norm.view(-1, 1) * edge_attr

    def update(self, x_updated, x, edge_attr, idx):
        # x_updated: (E, edge_out)
        # x: (E, node_in)
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn[idx](x_updated)
        return x_updated, edge_attr



if __name__ == "__main__":
    import encoder

    encoderNN = encoder.Encoder()

    N = 5
    position = torch.randn(N, 6, 2)
    # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_nn_conv.py
    data = encoderNN(position)

    gn = GN(128, 128, 128, 128)
    gn(data)
    print(data)