# coding = utf-8

import os
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle

from torch.utils.data import Dataset

import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')
from matplotlib import pyplot as plt


# Exp 2
class ExpNet(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.linear_base = nn.Linear(1, 100)
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(100, 100))
            layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
        self.linear_end = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.tanh(self.linear_base(x))
        x = self.layers(x)
        x = self.linear_end(x)
        return x


X_original = np.linspace(0, 10, 1000)
Y_original = 8 * np.cos(2 * X_original) + X_original + np.random.randn(1000,)
# 随机选取其中500个作为训练集
index = np.arange(0, 1000)
np.random.shuffle(index)
index = index[:800]
X = X_original[index]
Y = Y_original[index]

net = ExpNet(n_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)

batch = 2
dir_out = './images/figures-regression-NMS%d' % batch

X_tensor = torch.Tensor(X_original.reshape((-1, 1)))
Y_tensor = torch.Tensor(Y_original.reshape((-1, 1)))
if os.path.exists(dir_out):
    shutil.rmtree(dir_out)
os.mkdir(dir_out)

fig = plt.figure(figsize=(10,10), dpi=100)
plt.scatter(X_original, Y_original, s=4, c='black', alpha=0.3)
plt.savefig(os.path.join(dir_out, 'original.jpg'))


def train(iters):
    with torch.no_grad():
        Y_tensor_ = net(X_tensor)
        Y_pred = np.array(Y_tensor_).reshape(-1)
        # 训练之前的曲线
        plt.plot(X_original, Y_pred)
    # 当前net与每一个样本的距离
    distance = torch.pow(Y_tensor_ - Y_tensor, 2)
    distance, idx = torch.sort(distance.view(-1), dim=0, descending=True)
    reserved_Idx = [idx[0].item()]
    d = 2
    for i in idx:
        if len(reserved_Idx) >= batch:
            break
        t = X_tensor[i]
        g = True
        for ri in reserved_Idx:
            rt = X_tensor[ri]
            _d = torch.sqrt(torch.sum((t - rt) ** 2))
            if _d < d:
                g = False
                break
        if g is True:
            reserved_Idx.append(i.item())
    #
    inputs = X_tensor[reserved_Idx, :]
    targets = Y_tensor[reserved_Idx, :]

    # train
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        Y_tensor_ = net(X_tensor)
        Y_pred_ = np.array(Y_tensor_).reshape(-1)
        # 原始所有点
        plt.scatter(X_original, Y_original, s=4, c='black', alpha=0.3)
        # 训练后的曲线
        plt.plot(X_original, Y_pred_, c='r')
        # 当前用于训练的点
        plt.scatter(inputs, targets, s=25, c='r', marker='v')
        plt.title('Iters: %d'% iters)
        plt.savefig(os.path.join(dir_out, '%d.jpg' % (iters)))
        plt.close()


for iters in range(1, 2000):
    train(iters)
    print(iters)

with torch.no_grad():
    Y_tensor_ = net(X_tensor)
    Y_pred = np.array(Y_tensor_).reshape(-1)
    plt.figure(figsize=(10,10), dpi=200)
    plt.scatter(X_original, Y_original, s=4, c='black', alpha=0.3)
    plt.scatter(X, Y, s=4, alpha=0.8)
    plt.plot(X_original, Y_pred)
    plt.savefig(os.path.join(dir_out, 'final.jpg'))
    plt.close()
