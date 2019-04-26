# coding = utf-8

import os
import numpy as np
import shutil
import torch
import torch.nn as nn
from random import shuffle
from torch.utils.data import Dataset
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Agg')


class ExpNet(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.linear_base = nn.Linear(2, 100)
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(100, 100))
            layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
        # for 2 class
        self.linear_end = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.tanh(self.linear_base(x))
        x = self.layers(x)
        x = self.linear_end(x)
        return x


# Data
from sklearn.datasets import make_moons, make_circles
# X_original, Y_original = make_moons(n_samples=1000, shuffle=True, noise=0.1)
X_original, Y_original = make_circles(n_samples=1000, shuffle=True, noise=0.1, factor=0.5)

# 随机选取其中800个作为训练集
index = np.arange(0, 1000)
np.random.shuffle(index)
index = index[:800]
X = X_original[index]
Y = Y_original[index]
X_tensor = torch.Tensor(X.reshape(-1, 2)).float()
Y_tensor = torch.Tensor(Y.reshape(-1, 1)).float()

xrange = np.linspace(X_original[:,0].min(), X_original[:,0].max(), 100)
yrange = np.linspace(X_original[:,1].min(), X_original[:,1].max(), 100)
Grid_tensor = [[[xr, yr] for yr in yrange] for xr in xrange]
Grid_tensor = torch.Tensor(np.array(Grid_tensor).reshape((-1, 2)))

net = ExpNet(n_layers=2)
criterion = nn.BCEWithLogitsLoss(reduction='none')
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

batch = 1
dir_out = './images/figures-classification-circle-random%d' % batch
# dir_out = './images/figures-classification-random%d' % batch

if os.path.exists(dir_out):
    shutil.rmtree(dir_out)
os.mkdir(dir_out)

# 上一个决策边界
P_grid = np.zeros((len(xrange), len(yrange)))


def train(iters):
    with torch.no_grad():
        net.eval()
        Y_pred = net(X_tensor)
        loss = criterion(Y_pred, Y_tensor)
    # # 当前net与每一个样本的距离
    # distance, idx = torch.sort(loss.view(-1), dim=0, descending=True)
    # 随机
    idx = [i for i in range(800)]
    shuffle(idx)
    #
    inputs = X_tensor[idx[:batch], :]
    targets = Y_tensor[idx[:batch], :]
    # train
    net.train()
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss = torch.mean(loss)
    loss.backward()
    optimizer.step()

    # plot
    plot(net, [inputs, targets], iters)


def plot(net, current_Tensor, iters):
    global P_grid
    with torch.no_grad():
        net.eval()
        plt.figure(figsize=(20,7), dpi=100)
        plt.subplot(1,2,2)
        # 绘制当前决策边界
        C_grid = net(Grid_tensor)
        C_grid = torch.sigmoid(C_grid)
        C_grid = C_grid.numpy()
        C_grid = C_grid.reshape(len(xrange), len(yrange)).T
        plt.contourf(xrange, yrange, C_grid, 8, alpha=0.3)
        plt.contour(xrange, yrange, C_grid, (0.5,), linewidths=(2,), linestypes=('-',), colors='r')
        # 绘制上一个决策边界
        plt.contour(xrange, yrange, P_grid, (0.5,), linewidths=(1,), linestypes=('--',), colors='b')
        # 绘制原始点
        plt.scatter(X_original[:, 0][Y_original == 0], X_original[:, 1][Y_original == 0], marker='.', alpha=0.8, s=5,
                    c='b')
        plt.scatter(X_original[:, 0][Y_original == 1], X_original[:, 1][Y_original == 1], marker='.', alpha=0.8, s=5,
                    c='g')
        # 绘制当前点
        _X, _Y = current_Tensor
        _X = _X.numpy()
        _Y = _Y.squeeze(-1).numpy()
        plt.scatter(_X[:, 0][_Y == 0], _X[:, 1][_Y == 0], s=25, c='b', marker='v')
        plt.scatter(_X[:, 0][_Y == 1], _X[:, 1][_Y == 1], s=25, c='g', marker='v')

        #####
        plt.subplot(1, 2, 1)
        # 绘制原始点
        plt.scatter(X_original[:, 0][Y_original == 0], X_original[:, 1][Y_original == 0], marker='.', alpha=0.8, s=5,
                    c='b')
        plt.scatter(X_original[:, 0][Y_original == 1], X_original[:, 1][Y_original == 1], marker='.', alpha=0.8, s=5,
                    c='g')
        # 绘制上一个决策边界
        plt.contourf(xrange, yrange, P_grid, 8, alpha=0.3)
        plt.contour(xrange, yrange, P_grid, (0.5,), linewidths=(1,), linestypes=('--',), colors='b')
        # 绘制当前点
        _X, _Y = current_Tensor
        _X = _X.numpy()
        _Y = _Y.squeeze(-1).numpy()
        plt.scatter(_X[:, 0][_Y == 0], _X[:, 1][_Y == 0], s=25, c='b', marker='v')
        plt.scatter(_X[:, 0][_Y == 1], _X[:, 1][_Y == 1], s=25, c='g', marker='v')

        ###
        plt.suptitle('Iters: %d' % iters)
        plt.savefig(os.path.join(dir_out, '%d.jpg' % (iters)))
        plt.close()

        P_grid = C_grid.copy()


for iters in range(1, 4000):
    train(iters)
    print(iters)
    # assert(False)