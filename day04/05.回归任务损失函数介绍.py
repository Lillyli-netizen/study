"""
案例:
演示 回归任务的损失函数介绍.
回归任务常用损失函数如下:
MAE:Mean Absolute Error,平均绝对误差.
公式:
误差绝对值之和/样本总数
类似于L1正则化，权重可以降维O，数据会变得稀疏.

幣端:
在0点不平滑，可能错过最小值.
MSE:Mean Squared Error,均方误差.
公式：误差平方之和十样本总数

弊端:
如果差值过大，可能存在梯度爆炸的情况。

Smooth L1:
就是基于MAE 和MSE做的综合，在[-1,1]是L2(MSE)，其它段时L1.
这样即解决了L1不平滑的问题(θ点不可导，可能错过最小值)又解决了L2(MSE)的 梯度爆炸的问题.
Smooth L1:
到底谁概率和要为1
"""
import torch
import torch.nn as nn

def dm01():
#1.定义函数，演示:MAE 损失函数.
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)
    # 2.定义变量，记录：预测值
    y_pred = torch.tensor([1.0,1.0, 1.9], requires_grad=True)

    # 3.创建mae损失函数对象
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()

    # 4.计算损失值 具体计算的过程ai讲一下
    loss = criterion(y_pred, y_true).detach().numpy()
    print(f'损失值:{loss}')
#2.定义函数，演示:MSE 损失函数，
def dm02():
#1.定义函数，演示:MAE 损失函数.
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)
    # 2.定义变量，记录：预测值
    y_pred = torch.tensor([1.0,1.0, 1.9], requires_grad=True)

    # 3.创建mae损失函数对象
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    # 4.计算损失值 具体计算的过程ai讲一下 为啥换了函数代码结果却是一样的
    loss = criterion(y_pred, y_true).detach().numpy()
    print(f'损失值:{loss}')
#3.定义函数，演示:Smooth L1损失函数.
def dm03():
#1.定义函数，演示:MAE 损失函数.
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)
    # 2.定义变量，记录：预测值
    y_pred = torch.tensor([1.0,1.0, 1.9], requires_grad=True)

    # 3.创建mae损失函数对象
    # criterion = nn.L1Loss()
    criterion = nn.SmoothL1Loss()

    # 4.计算损失值 具体计算的过程ai讲一下 为什么这个损失值和前两个差别很大 谁是最差的
    loss = criterion(y_pred, y_true).detach().numpy()
    print(f'损失值:{loss}')

# 4.测试
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()