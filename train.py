import time
import argparse # 这好像是一个在terminal用来接收输入的参数的
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

from products_dgl import get_products_data


# Training settings
parser = argparse.ArgumentParser() # 然后创建一个解析对象
# parse.add_argument() 向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset string.') # 'cora'
parser.add_argument('--model', type=str, default='gcn',
                    help='Model string.') # 'gcn', 'gcn_cheby', 'dense'
parser.add_argument('--no_mps', action='store_true', default=False,
                    help='Disables MPS training.') # GPU参数，默认为False
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
# print(args): Namespace(dropout=0.5, epochs=200, fastmode=False, hidden=16, lr=0.01, no_mps=False, seed=42, weight_decay=0.0005)
args.mps = not args.no_mps and torch.backends.mps.is_available() # 给args加了一个量mps=True
# print(args): Namespace(dropout=0.5, epochs=200, fastmode=False, hidden=16, lr=0.01, mps=True, no_mps=False, seed=42, weight_decay=0.0005)

np.random.seed(args.seed)
torch.manual_seed(args.seed) # 设置一个随机数种子

# Load data
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)

print('adj:', adj.shape)
print(adj)
print('features:', features.shape)
print(features)
print('labels:', labels.shape)
print(labels)
print('idx_train:', idx_train.shape)
print('idx_test:', idx_test.shape)
print('idx_val:', idx_val.shape)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
# 为了使用torch.optim，需先构造一个优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数。
# 要构建一个优化器optimizer，你必须给它一个可进行迭代优化的包含了所有参数（所有的参数必须是变量s）的列表。
# 然后，您可以指定程序优化特定的选项，例如学习速率，权重衰减等。
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# if args.mps:
#     mps_device = torch.device("mps")
#     model.to(mps_device)
#     features = features.to(mps_device)
#     labels = labels.to(mps_device)
#     adj = adj.to(mps_device)
#     # idx_train = idx_train.to(mps_device)
#     # idx_val = idx_val.to(mps_device)
#     # idx_test = idx_test.to(mps_device)


# print(next(model.parameters()).device)


def train(epoch):
    t = time.time() # 返回当前时间
    model.train() # 将模型转为训练模式，并将优化器梯度置零
    optimizer.zero_grad() # 意思是把梯度置零，即把loss关于weight的导数变成0；pytorch中每一轮batch需要设置optimizer.zero_grad
    output = model(features, adj) # 前向传播求出预测的值
    loss_train = F.nll_loss(output[idx_train], labels[idx_train]) # 函数全称是negative log likelihood loss
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward() # 反向传播求梯度
    optimizer.step() # 更新所有参数
    '''由于在算output时已经使用了log_softmax，这里使用的损失函数是NLLloss，如果之前没有加入log运算，
     这里则应使用CrossEntropyLoss
     损失函数NLLLoss() 的输入是一个对数概率向量和一个目标标签. 它不会为我们计算对数概率，
     适合最后一层是log_softmax()的网络. 损失函数 CrossEntropyLoss() 与 NLLLoss() 类似,
     唯一的不同是它为我们去做 softmax.可以理解为：CrossEntropyLoss()=log_softmax() + NLLLoss()
     理论上，对于单标签多分类问题，直接经过softmax求出概率分布，然后把这个概率分布用crossentropy做一个似然估计误差。
     但是softmax求出来的概率分布，每一个概率都是(0,1)的，这就会导致有些概率过小，导致下溢。 考虑到这个概率分布总归是
     要经过crossentropy的，而crossentropy的计算是把概率分布外面套一个-log 来似然
     那么直接在计算概率分布的时候加上log,把概率从（0，1）变为（-∞，0），这样就防止中间会有下溢出。
     所以log_softmax本质上就是将本来应该由crossentropy做的取log工作提到预测概率分布来，跳过了中间的存储步骤，防止中间数值会有下溢出，使得数据更加稳定。
     正是由于把log这一步从计算误差提到前面的步骤中，所以用log_softmax之后，下游的计算误差的function就应该变成NLLLoss
     （NLLloss没有取log这一步，而是直接将输入取反，然后计算其和label的乘积，求和平均)
     '''

    # 通过model.eval()转为测试模式，之后计算输出，并单独对测试集计算损失函数和准确率。
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval() # eval() 函数用来执行一个字符串表达式，并返回表达式的值
        output = model(features, adj)

    # 测试集的损失函数
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
