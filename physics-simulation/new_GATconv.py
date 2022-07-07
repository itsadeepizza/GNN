# https://colab.research.google.com/drive/1daibj3ovfy8wKZWubrT6PhbLY2MTWpUA#scrollTo=irqdVny1FRvn
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.conv import MessagePassing
import math

# Some functions to initialize weights


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class New_GATConv(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, edge_dim=0, **kwargs):
        super(New_GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        print('making att matrix  ------------')
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))
        print('att shape ', self.att.shape)
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edgemat, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if torch.is_tensor(x):
            print('x is ', x.shape)
            print('weight is ', self.weight.shape)
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        out = self.propagate(edge_index, size=size, x=x, edgemat=edgemat)
        edgemat = self.__edgemat__
        self.__edgemat__ = None
        return out, edgemat

    def message(self, edge_index_i, x_i, x_j, size_i, edgemat):
        # Compute attention coefficients.
        # print('edgemat check ',edgemat.shape)
        # print('x_j before view ',x_j.shape)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        # print('x_j after view ',x_j.shape)

        print('att slice for compute ', self.att[:, :, self.out_channels:].shape)

        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
            print('alpha after compute  ', alpha.shape)
        else:
            print('x_i in else  ', x_i.shape)
            x_i = x_i.view(-1, self.heads, self.out_channels)
            print('x_i after view  ', x_i.shape)
            print('x_j after view  ', x_j.shape)
            # print('torch.cat([x_i, x_j] ',torch.cat([x_i, x_j],dim=-1).shape)
            print('self.att ', self.att.shape)
            # edgemat = edgemat.unsqueeze(dim=1)
            print('edgemat before concat', edgemat.shape)
            # print('(torch.cat([x_i, x_j,edgemat.expand(len(edgemat),head,3)], dim=-1) * self.att ',(torch.cat([x_i, x_j,edgemat.expand(len(edgemat),self.heads,3)], dim=-1)*self.att).shape)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            print('alpha after compute with concate  ', alpha.shape)
            alpha = alpha.sum(dim=-1)  # added for approach 4
            alpha = alpha * edgemat
            edgemat = alpha  # edgemat * alpha
            print('edgemat after mul with alpha ', edgemat.shape)
            alpha = alpha.sum(dim=-1)
            print('alpha after mul with edgemat ', alpha.shape)
            # alpha = (torch.cat([x_i, x_j,edgemat.expand(len(edgemat),self.heads,3)], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)
        alpha = alpha.unsqueeze(dim=-1)

        self.__edgemat__ = edgemat

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        print('(x_j * alpha.view(-1, self.heads, 1)).shape ', (x_j * alpha.view(-1, self.heads, 1)).shape)
        return x_j * alpha.view(-1, self.heads, 1)  # ,edgemat

    def update(self, aggr_out):
        # print('in update mehtod edgemat ',edgemat.shape)
        print('aggr_out shape before processing ', aggr_out.shape)

        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
            print('in concat,aggr_out shape after view ', aggr_out.shape)
        else:

            aggr_out = aggr_out.mean(dim=1)
            print('not in concat,aggr_out shape after mean ', aggr_out.shape)

        if self.bias is not None:
            print('in bias ', self.bias.shape)
            aggr_out = aggr_out + self.bias
            print('in bias, aggr_out shape after adding bias ', aggr_out.shape)

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)