"""
案例:
绘制激活函数Tanh的 函数图像 和 导数图像.

ai补全每个函数的代码示例
Sigmoid激活函数介绍:
激活函数的目的:
给模型增加非线性功能，让模型(神经元)既可以做分类，还可以做回归问题.
激活函数的分类:
Sigmoid:
ReLU:
Tanh:
Softmax:
Sigmoid激活函数:
主要应用于 二分类的输出层，且适用于 浅层神经网络(不超过5层).
数据在[-6，6]之间有效果，在[-3，3]之间效果明显。会将数据值映射到[0, 1]
求导后范围在[0，0.25]
tanh
主要应用于隐藏层,且适用于浅层神经网络(不超过5层) 为什么都是浅层呢ai讲讲
数据在[-3, 3]之间有效果,在[-1, 1]之前效果明显,会将数据值映射值[-1, 1]
求导后范围在[-1, 1]
Softmax:

将多分类的结果以概率的形式展示，且概率和相加为，最终选取概率值最大的分类 作为最终结果.
细节:
绘制活数图像时出现以下提示，需要将anaconda3/Lib/site-packages/torch/ib目录下的ibiomp5md.dl文件删除
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
"""

import torch

from torch.utils.data import TensorDataset #构造数据集对象

from torch.utils.data import DataLoader #数据加载器

from torch import nn #nn模块中有平方损失函数和假设函数

from torch import optim #optim模块中有优化器函数

from sklearn.datasets import make_regression #创建线性回归模型数据集

import matplotlib.pyplot as plt #可视化

plt.rcParams['font.sans-serif']= ['SimHei'] # 用来正常显示中文标签

plt.rcParams['axes.unicode_minus']= False # 用来正常显示负号

# #1.创建画布和坐标轴，1行2列. 讲解这个语法
# fig, axes = plt.subplots(1,  2)
# #2.生成-20 ~20之问的1000个数据点。
# x = torch.linspace(-20, 20,1000)
# # print(f'x: {x}')
# #3.计算上述1000个点，tanh激活函数处理后的值.
# y = torch.tanh(x)
# # print(f'y:{y}')
# #4.在第1个子图中绘制tanh激活函数的图像。
# axes[0].plot(x,y)
# axes[0].set_title('Sigmoid激活函数图像')
# axes[0].grid()
# #5.在第2个图上，绘制tanh激活函数的导数图像。
# #5.1 重新生成-20~ 20之问的 1000个数据点.
# #
# # 参1:起始值，参2:结束值，参3:元素的个数，参4:是否需要求导。
# x = torch.linspace(-20, 20, 1000, requires_grad=True)
# y = torch.tanh(x)
# #5.2 对y进行求导
# y.sum().backward()
# #5.3 获取x的梯度值
# dy_dx = x.grad
# #5.4 在第2个子图中绘制导数图像
# axes[1].plot(x.detach().numpy(), dy_dx.detach().numpy())
# axes[1].set_title('Tanh激活函数导数图像')
# axes[1].grid()
# #6.显示图像
# plt.show()

# 1.定义张量记录分类数据
# scores = torch.tensor([0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75]) 这个数据随便写都可以吗 到底是怎样的过程 是把这个分数分类了吗???
scores = torch.tensor([0.2,0.35, 0.1, 0.46])
#2.dim=0，按行计算
probabilities = torch.softmax(scores, dim=0)

print(probabilities)
# 输出结果 相加等于1
# tensor([0.0212, 0.0177, 0.0202, 0.0202, 0.0638, 0.0287, 0.0185, 0.0522, 0.0183,
#         0.7392])