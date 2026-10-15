"""
案例:
演示多分类任务的交叉熵损失函数.
损失函数介绍:
概述:
损失函数也叫成本函数，目标函数，代价函数，误差函数，就是用来衡量 模型好坏(模型拟合情况)的.
分类:
分类问题:
多分类交叉熵损失:CrossEntropyLoss
二分类交叉熵损失:BCEL0ss
回归问题:
MAE:Mean Absolute Error,平均绝对误差.
MSE:Mean Squared Error,均方误差.
Smooth L1:结合上述两个的特点做的升级，优化.

设计思路:
多分类交叉熵损失:CrossEntropyLoss
Loss = - Eylog(S(f(x)))
简单记忆:
X:样本
f(x):加权求和
s(f(x)):处理后的概率
y:样本x属于某一个类别的 真实概率。

大白话解释
损失函数结果 = 最小化正确类别概率的对数的最小化

细节:

CrossEntropyLoss = Softmax()

+ 损失计算，后续如果用这个损失函数，则:输出层就不用额外调用 s0ftmax()激活函数了.


"""
import torch
import torch.nn as nn

# 1.定义函数:演示多分类交叉熵损失
def dm01():
    # 1.手动创建样本的真实值-就是上述公式里的y;老师说float才可以 你说老师为啥写了一个又注释了写另一个
    # y_true = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float)
    y_true = torch.tensor([0, 1])

    # 2.手动创建样本的预测值-上述的fx
    y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], requires_grad=True, dtype=torch.float)
    # 3. 创建多分类函数
    criterion = nn.CrossEntropyLoss()  #平均损失, 来源于参数reduction:mean

    # 4.计算损失值 这是什么语法ai讲一下
    loss = criterion(y_pred, y_true)
    print(f'损失值:{loss}')


# 2.测试
if __name__ == '__main__':
    dm01()
