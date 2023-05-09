import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, ChebGraphConv

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

class ChebyNet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, enable_bias, K_order, K_layer, dropout):
        super(ChebyNet, self).__init__()

        self.cheb_graph_convs = nn.ModuleList()
        self.K_order = K_order
        self.K_layer = K_layer
        # Build the layer of the model
        self.cheb_graph_convs.append(ChebGraphConv(K_order, nfeat, nhid, enable_bias))
        for k in range(1, K_layer - 1):
            self.cheb_graph_convs.append(ChebGraphConv(K_order, nhid, nhid, enable_bias))
        self.cheb_graph_convs.append(ChebGraphConv(K_order, nhid, nclass, enable_bias))