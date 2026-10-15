"""
案例
演示二分类任务的损失函数.

二分类任务的损失函数(BCELOSs):公式:

Loss = -ylog(预测) -(1 -y)log(1 -预测值)

细节:

因为公式中没有包含Sigmoid激活函数，所以使用BCELoss的时候，还需要手动指定 Sigmoid.
"""
import torch
import torch.nn as nn

def dm01():
    # 1.设置真实值
    y_true = torch.tensor([0, 1, 0], dtype=torch.float)

    # 2.设置预测值(概率)
    y_pred = torch.tensor([0.6901, 0.9845, 0.2311], dtype=torch.float)

    # 3. 创建二分类交叉熵损失函数 为啥不能这样写
    # criterion = nn.BCELoss(y_pred, y_true)
    criterion = nn.BCELoss()

    # 4.计算损失值 讲一下.detach().numpy()的目的作用
    loss = criterion(y_pred, y_true).detach().numpy()
    print(f'损失值:{loss}')

if __name__ == '__main__':
    dm01()