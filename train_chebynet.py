import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import load_data, accuracy
from models import ChebyNet


def train(epoch, fastmode, model, optimizer, adj, features, labels, idx_train, idx_val):
    t = time.time()  # 返回当前时间
    model.train()  # 将模型转为训练模式，并将优化器梯度置零
    optimizer.zero_grad()  # 意思是把梯度置零，即把loss关于weight的导数变成0；pytorch中每一轮batch需要设置optimizer.zero_grad
    output = model(features, adj)  # 前向传播求出预测的值
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])  # 函数全称是negative log likelihood loss
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()  # 反向传播求梯度
    optimizer.step()  # 更新所有参数

    # 通过model.eval()转为测试模式，之后计算输出，并单独对测试集计算损失函数和准确率。
    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()  # eval() 函数用来执行一个字符串表达式，并返回表达式的值
        output = model(features, adj)

    # 测试集的损失函数
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test(model, adj, features, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

def get_parameters(args):
    # define device
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # define dataset
    dataset = args.dataset

    # define model parameter
    if args.Ko < 0:
        raise ValueError(f'ERROR: The order Ko is smaller than 0!')
    else:
        Ko = args.Ko
    if args.Kl <= 1:
        raise ValueError(f'ERROR: The layer Kl is smaller than 2!')
    else:
        Kl = args.Kl

    lr = args.lr
    weight_decay = args.weight_decay
    droprate = args.dropout

    n_hid = args.hidden
    epochs = args.epochs
    enable_bias = args.enable_bias
    fastmode = args.fastmode

    return device, dataset, Ko, Kl, lr, weight_decay, droprate, n_hid, epochs, enable_bias, fastmode


def main(args):
    device, dataset, Ko, Kl, lr, weight_decay, droprate, n_hid, epochs, enable_bias, fastmode = get_parameters(args)
    # Load data
    gso, feature, label, idx_train, idx_val, idx_test = load_data(dataset)

    # Prepare model
    model = ChebyNet(n_feat=feature.shape[1],
                     n_hid=n_hid,
                     n_class=np.unique(label).shape[0],
                     enable_bias=enable_bias,
                     K_order=Ko,
                     K_layer=Kl,
                     droprate=droprate)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    if args.cuda:
        model.to(device)
        feature = feature.to(device)
        gso = gso.to(device)
        label = label.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)

    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch, fastmode, model, optimizer, gso, feature, label, idx_train, idx_val)
    print(f"Optimization Finished on {dataset}!")
    print(f"The model is a {Kl} layers ChebyNet with {Ko}-th order Chebyshev Polynomial!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test(model, gso, feature, label, idx_test)
