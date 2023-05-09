import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys

import torch

from dgl.data import CiteseerGraphDataset
from dgl.data import CoraGraphDataset
from dgl.data import PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset

# def parse_index_file(filename):
#     """Parse index file."""
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index
#
# def sample_mask(idx, l):
#     """Create mask."""
#     # np.zeros()用于生成包含零的数组
#     mask = np.zeros(l)
#     mask[idx] = 1
#     # return np.array(mask, dtype=np.bool)
#     return np.array(mask, dtype=bool)

# def normalize(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx

# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)
#
# def load_data(dataset_str: str):
#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#     # 将已经清洗好分配好的数据加载入内存并按照类存放
#     for i in range(len(names)):
#         with open(f'data/ind.{dataset_str}.{names[i]}', 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))
#
#     # tuple中每一元素都会对应assign到前面的变量中
#     x, y, tx, ty, allx, ally, graph = tuple(objects)
#     # 下面两行得到了测试集的index，我知道那几行数据是作为测试集的
#     test_idx_reorder = parse_index_file(f'data/ind.{dataset_str}.test.index')
#     test_idx_range = np.sort(test_idx_reorder)
#
#     # scipy.sparse.vstack() 表示按行拼接（行数增加），列数必须相同
#     # .tolil()把这个矩阵变成一个list的list。相当于arrary形式的矩阵
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :] #不知道这个在干啥！！！！
#     # graph本来是一个用dict表示的点与点的连接，比如{0: [1, 2, 3]}说明node 0与node 1, node2, node 3连接。
#     # nx.from_dict_of_lists把dict表示的连接用list来表示了
#     #nx.adjacency_matrix定义其为邻接矩阵
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#
#     # 给y也做跟x同样的操作
#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]
#
#     # 得到index
#     idx_test = test_idx_range.tolist()
#     idx_train = range(len(y))
#     idx_val = range(len(y), len(y) + 500) # val表示cross validation
#
#     # train_mask = sample_mask(idx_train, labels.shape[0]) # labels.shape给出一个tuple：(有多少行， 有多少列)
#     # val_mask = sample_mask(idx_val, labels.shape[0])
#     # test_mask = sample_mask(idx_test, labels.shape[0])
#     #
#     # y_train = np.zeros(labels.shape)
#     # y_val = np.zeros(labels.shape)
#     # y_test = np.zeros(labels.shape)
#     # y_train[train_mask, :] = labels[train_mask, :]
#     # y_val[val_mask, :] = labels[val_mask, :]
#     # y_test[test_mask, :] = labels[test_mask, :]
#
#     # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
#
#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(np.where(labels)[1])
#     adj = sparse_mx_to_torch_sparse_tensor(adj)
#
#     idx_val = list(idx_val)
#     idx_test = list(idx_test)
#     idx_test = torch.LongTensor(idx_test)
#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#
#     return adj, features, labels, idx_train, idx_val, idx_test

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion
    to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_paper_raw_data(dataset):
    if dataset == 'cora':
        data = CoraGraphDataset()
    elif dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif dataset == 'pubmed':
        data = PubmedGraphDataset()

    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    return g, features, labels, train_mask, val_mask, test_mask

def get_other_raw_data(dataset):
    # with open('./data/products_dgl.pkl', 'rb') as f:
    #     dataset = pkl.load(f)
    if dataset == 'products':
        dataset_name = 'ogbn-' + dataset
        dataset = DglNodePropPredDataset(name = dataset_name)
    elif dataset == 'mag':
        dataset_name = 'ogbn-' + dataset
        dataset = DglNodePropPredDataset(name = dataset_name)


    split_idx = dataset.get_idx_split()
    idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, labels = dataset[0]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)

    # Flatten the labels to a 1d tensor
    if dataset == 'products':
        labels = labels.flatten()
    elif dataset == 'mag':
        pass

    # Get the feature of the graph
    features = graph.ndata['feat']

    return graph, features, labels, idx_train, idx_val, idx_test

def load_data(dataset):
    if dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed':
        graph, features, labels, idx_train, idx_val, idx_test = get_paper_raw_data(dataset)
    elif dataset == 'products' or dataset == 'mag':
        graph, features, labels, idx_train, idx_val, idx_test = get_other_raw_data(dataset)

    nx_graph = graph.to_networkx()
    # adj = nx.to_scipy_sparse_matrix(nx_graph, dtype=np.float)
    adj = nx.to_scipy_sparse_array(nx_graph, dtype = np.float)

    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, idx_train, idx_val, idx_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)