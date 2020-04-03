#!/usr/bin/env python
# encoding: utf-8
"""logistic regression
"""

import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


# 定义超参数
batch_size = 64
learning_rate = 1e-3
num_epochs = 10


# 下载训练集 MNIST 手写数字训练集
data_dir = "../datasets/FashionMNIST"
train_dataset = datasets.FashionMNIST(
    root=data_dir, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.FashionMNIST(
    root=data_dir, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#print(f"{len(train_dataset)}, {len(test_dataset)}")
#60000, 10000


# 定义 Logistic Regression 模型，和Linear Regression一模一样
class logsticRegression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(logsticRegression, self).__init__()
        self.logstic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        out = self.logstic(x)
        return out

# 定义简单的前馈神经网络
class neuralNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(neuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim),
            nn.ReLU(True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

#模型
model = neuralNetwork(28 * 28, 300, 100, 10)


# 定义loss和optimizer, 逻辑回归和线性回归就损失函数不同，其余一模一样
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
model = model.to(device=device)


#测试模型
def test_model(model, test_loader):
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        img = img.to(device=device)
        label = label.to(device=device)
        with torch.no_grad():
            out = model(img)
            loss = criterion(out, label)
        eval_loss += loss.item()
        _, pred = torch.max(out, 1)
        eval_acc += (pred == label).float().mean()
    print(f'Loss: {eval_loss/len(test_loader):.6f}, Acc: {eval_acc/len(test_loader):.6f}')


#""" 开始训练
for epoch in range(num_epochs):
    print('*' * 10)
    print(f'epoch {epoch+1}')
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    model.train()
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.view(img.size(0), -1)  # 将图片展开成 28x28
        img = img.to(device=device)
        label = label.to(device=device)
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item()
        _, pred = torch.max(out, 1)
        running_acc += (pred==label).float().mean()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print(f'[{epoch+1}/{num_epochs}] Loss: {running_loss/i:.6f}, Acc: {running_acc/i:.6f}')

    print(f'Finish {epoch+1} epoch, Loss: {running_loss/i:.6f}, Acc: {running_acc/i:.6f}')
    test_model(model, test_loader)
    print(f'Time:{(time.time()-since):.1f} s')
#"""


# save/load 模型
is_save = True
model_fn = "./mlp.pth"
if is_save:
    torch.save(model.state_dict(), model_fn)
else:
    model.load_state_dict(torch.load(model_fn))

# 测试模型
test_model(model, test_loader)
