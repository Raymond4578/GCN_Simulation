import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, ChebGraphConv


# Define class for GCN model
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


# Define class for ChebyNet model
class ChebyNet(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, enable_bias, K_order, K_layer, droprate):
        super(ChebyNet, self).__init__()
        self.cheb_graph_convs = nn.ModuleList()
        self.K_order = K_order
        self.K_layer = K_layer
        self.cheb_graph_convs.append(ChebGraphConv(K_order, n_feat, n_hid, enable_bias))
        for k in range(1, K_layer-1):
            self.cheb_graph_convs.append(ChebGraphConv(K_order, n_hid, n_hid, enable_bias))
        self.cheb_graph_convs.append(ChebGraphConv(K_order, n_hid, n_class, enable_bias))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, gso):
        for k in range(self.K_layer-1):
            x = self.cheb_graph_convs[k](x, gso)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.cheb_graph_convs[-1](x, gso)
        x = self.log_softmax(x)
        return x