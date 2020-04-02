#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
print(torch.__version__)


# 返回的数组大小5x4的矩阵
print(torch.Tensor(5, 4))
#使用torch.tensor创建Tensor
print(torch.tensor([5,4]))
# 得到矩阵大小
a = torch.rand(5, 4)
print(a.size())
# numpy 类似的返回5x4大小的矩阵
print(np.ones((5, 4)))
# numpy 和 torch.Tensor 之间的转换
a = torch.rand(5, 4)
b = a.numpy()
print(b)
a = np.array([[3, 4], [3, 6]])
b = torch.from_numpy(a)
print(b)


# 运算和numpy类似
x = torch.rand(5, 4)
y = torch.rand(5, 4)
c = 3
print(c * x)
print(x + y)
print(x.add(y))
# 可以直接进行操作改变原对象，x+y或者x.add()并不会改变x，但是x.add_()则会对x进行改变
x.add_(y)
print(x)


# 将 torch.Tensor 放到 GPU 上
# 判断一下电脑是否支持GPU
torch.cuda.is_available()
a = torch.rand(5, 4)
# a = a.cuda()
print(a)


# ### torch 的自动求导功能
# torch 和大部分框架一样有着自动求导功能，对象不再是 torch.Tensor，而是torch.autograd.Variable
# 本质上Variable和Tensor没有什么区别，不过Variable会放在一个计算图里面，可以进行前向传播和反向传播以及求导  
# 里面的creator表示通过什么操作得到的这个Variable，grad表示反向传播的梯度

from torch.autograd import Variable
# requires_grad 表示是否对其求梯度，默认是False
x = Variable(torch.Tensor([3]), requires_grad=True)
y = Variable(torch.Tensor([5]), requires_grad=True)
z = 2 * x + y + 4
# 对 x 和 y 分别求导，默认只求一次导，如果要多次求导需要设置 retain_graph=True
z.backward()
# x 的导数和 y 的导数
print('dz/dx: {}'.format(x.grad.data))
print('dz/dy: {}'.format(y.grad.data))


# ### 神经网络部分
# 所依赖的主要是 torch.nn 和 torch.nn.functional
# torch.nn 里面有着所有的神经网络的层的操作，其用来构建网络，只有执行一次网络的运算才执行一次 
# torch.nn.functional 表示的是直接对其做一次向前运算操作

# 基本的网络构建类模板
class net_name(torch.nn.Module):
    def __init__(self):
        super(net_name, self).__init__()
        # 可以添加各种网络层
        self.conv1 = torch.nn.Conv2d(3, 10, 3)
        # 具体每种层的参数可以去查看文档
        
    def forward(self, x):
        # 定义向前传播
        out = self.conv1(x)
        return out
