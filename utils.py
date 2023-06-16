import random
import time

import dgl
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch

from dgl.data import CiteseerGraphDataset
from dgl.data import CoraGraphDataset
from dgl.data import PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset


# random.seed(3991)
np.random.seed(3991)

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

    return g, features, labels

def get_other_raw_data(dataset):
    # with open('./data/products_dgl.pkl', 'rb') as f:
    #     dataset = pkl.load(f)
    if dataset == 'products':
        dataset_name = 'ogbn-' + dataset
        data = DglNodePropPredDataset(name = dataset_name)

        graph, labels = data[0]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        num_nodes = len(labels)

        bei = 2
        num = int(num_nodes * 0.05 * bei)
        subset_idx = list(np.random.permutation(num_nodes)[:num])
        subset_idx.sort()

        graph = dgl.node_subgraph(graph, subset_idx)

        labels = labels.numpy()
        labels = labels[subset_idx]
        labels = torch.LongTensor(labels)
    # elif dataset == 'mag':
    #     dataset_name = 'ogbn-' + dataset
    #     dataset = DglNodePropPredDataset(name = dataset_name)

    # graph, labels = data[0]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)

    # Flatten the labels to a 1d tensor
    if dataset == 'products':
        labels = labels.flatten()
    # elif dataset == 'mag':
    #     pass

    # Get the feature of the graph
    features = graph.ndata['feat']

    return graph, features, labels

# def reform_labels(labels):
#     labels = labels.numpy()
#     unique_labels = np.unique(labels)
#     unique_labels.sort()
#     return unique_labels

def get_idx_train_val_test(labels):
    labels = labels.numpy()
    full_length = labels.shape[0]
    # format train, validate and test index
    train_idx_dict = {}
    for num in np.unique(labels):
        train_idx_dict[num] = []

    counter = 0
    # do loop to get idx_train
    added_idx = []
    while counter <= 1000000:
        # break condition
        # check all value has 20 elements
        all_twenty = True
        for key in train_idx_dict:
            if len(train_idx_dict[key]) <= 19:
                all_twenty = False
                break
        if all_twenty:
            break

        # generate idx for train
        random_idx = random.randint(0, full_length - 1)
        if random_idx not in added_idx:
            counter += 1
            if len(train_idx_dict[labels[random_idx]]) <= 19:
                train_idx_dict[labels[random_idx]].append(random_idx)
                added_idx.append(random_idx)

    idx_train = []
    for key in train_idx_dict:
        idx_train.extend(train_idx_dict[key])
    idx_train.sort()
    idx_train = np.array(idx_train)
    current_length = len(added_idx)
    #
    while 1:
        # break condition
        if len(added_idx) == current_length + 500 + 1000:
            break

        random_idx = random.randint(0, full_length - 1)
        if random_idx not in added_idx:
            added_idx.append(random_idx)

    val_test_idx = added_idx[-1500:]

    idx_val = val_test_idx[:500]
    idx_val.sort()
    idx_test = val_test_idx[500:]
    idx_test.sort()

    idx_val = np.array(idx_val)
    idx_test = np.array(idx_test)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return idx_train, idx_val, idx_test

def load_data(dataset: str = 'cora'):
    t = time.time()
    if dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed':
        graph, features, labels = get_paper_raw_data(dataset)
    # elif dataset == 'products' or dataset == 'mag':
    elif dataset == 'products':
        graph, features, labels = get_other_raw_data(dataset)

    # form index for train, validate and test data
    idx_train, idx_val, idx_test = get_idx_train_val_test(labels)

    # deal with feature
    features = features.numpy()
    features = normalize(features)
    features = torch.FloatTensor(features)

    # deal with adj
    nx_graph = graph.to_networkx()
    adj = nx.to_scipy_sparse_array(nx_graph, dtype=np.float)
    adj = sp.coo_matrix(adj)

    # build symmetric adjacency matrix
    # https://github.com/yao8839836/text_gcn/issues/17
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # renormalization trick (preprocess)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
