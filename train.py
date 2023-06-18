from __future__ import division
from __future__ import print_function

import time
import argparse # 这是一个在terminal用来接收输入的参数的
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

import train_gcn

# Training settings
parser = argparse.ArgumentParser() # 然后创建一个解析对象
# parse.add_argument() 向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项
parser.add_argument('--dataset', type = str, default = 'cora',
                    help='Dataset for training.')
parser.add_argument('--model', type = str, default = 'gcn',
                    help='Model for training. ("gcn", "chebynet")')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args() # 调用parse_args()方法进行解析；解析成功之后即可使用。# 这个是使用argparse模块时的必备行，将参数进行关联
# args现在是一个叫Namespace的object
args.cuda = not args.no_cuda and torch.cuda.is_available() # 给args加了一个量cuda=True

np.random.seed(args.seed)
torch.manual_seed(args.seed) # 设置一个随机数种子
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def main():
    if args.model == 'gcn':
        train_gcn.main(args)


if __name__ == '__main__':
    main()
