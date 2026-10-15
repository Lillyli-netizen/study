"""
案例:
演示参数初始化的7 种方式.
参数初始化的目的:
1.防止梯度消失 或者梯度爆炸.
2.提高收敛速度.
3.打破对称性。
参数初始化的方式:
无法打破对称性的:
个0，全1，固定值
可以打破对称性的:
随机初始化，正态分布初始化，kaiming初始化，xavier初始化
总结:
1.记忆kaiming初始化，xavier初始化，全0初始化.
2.关于初始化的选择上:
激活函数ReLU及其系列:优先用kaiming
激活函数非ReLU:优先用 xavier
如果是浅层网络:可以考虑使用 随机初始化
"""
import torch.nn as nn
#1.均匀分布随机初始化
def dm01():
    # 1.创建一个线性层,输入维度5, 输出维度3  为啥要创建线性层呢  老师说代码只写隐藏和输出 这代码啥意思 什么语法

    linear = nn.Linear(5, 3)

    # 2 对权重w从0-1均匀分布产生参数 为啥加下划线
    nn.init.uniform_(linear.weight)
    # 3 对偏置b从0-1均匀分布产生参数 为啥加下划线
    nn.init.uniform_(linear.bias)

    # print(linear.weight) 这个结果是完全随机的吗 我和老师的也不一样代码一致
    # Parameter
    # containing:
    # tensor([[0.1075, 0.3925, 0.3451, 0.1606, 0.1494],
    #         [0.8683, 0.8374, 0.9203, 0.4307, 0.2774],
    #         [0.3126, 0.9431, 0.1106, 0.2535, 0.8279]], requires_grad=True)
    print(linear.weight.data)
    print(linear.bias.data)
#2.固定初始化
def dm02():
    # 1.创建一个线性层,输入维度5, 输出维度3
    linear = nn.Linear(5, 3)

    # 2 对权重w从0-1进行初始化 什么值是3
    nn.init.constant_(linear.weight, 3)
    # 3 对权重w从0-1进行初始化

    nn.init.constant_(linear.bias, 3)
    # 4 打印生成结果
    # tensor([[3., 3., 3., 3., 3.],
    #         [3., 3., 3., 3., 3.],
    #         [3., 3., 3., 3., 3.]])
    # tensor([3., 3., 3.])
    print(linear.weight.data)
    print(linear.bias.data)
#3.全0初始化
def dm03():
    # 1.创建一个线性层,输入维度5, 输出维度3
    linear = nn.Linear(5, 3)

    # 2 对权重w进行初始化 全0
    nn.init.zeros_(linear.weight)
    # 3 对偏置b进行初始化 全0
    nn.init.zeros_(linear.bias)
    # 4 打印生成结果
    # tensor([[0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0.]])
    # tensor([0., 0., 0.])
    print(linear.weight.data)
    print(linear.bias.data)
#4.全1初始化
# 其他每一种ai补上

#5.正态分布随机初始化
def dm05():
    # 1.创建一个线性层,输入维度5, 输出维度3
    linear = nn.Linear(5, 3)

    # 2 对权重w进行初始化,正态分布初始化 均值为0, 标准差为1 这两个数字是固定的吗
    nn.init.normal_(linear.weight)

    # 4 打印生成结果
    # tensor([[-1.5158, 1.5876, 0.7509, 0.1746, -0.2708],
    #         [-2.2757, -0.6781, -0.4770, -0.1687, -1.1589],
    #         [0.0452, 0.4518, -1.6076, -1.3884, -0.1519]])
    print(linear.weight.data)

#6.kaiming初始化
def dm06():
    # 1.创建一个线性层,输入维度5, 输出维度3
    linear = nn.Linear(5, 3)

    # 2 对权重w进行初始化,这是kaiming的 有啥区别
    nn.init.kaiming_normal_(linear.weight)

    # 4 打印生成结果
    # tensor([[1.5046, -0.5497, 0.3034, -0.0672, 1.0932],
    #         [1.0974, 0.2554, -1.4905, -0.5190, -0.8871],
    #         [-0.5310, -0.2631, 0.9815, -0.3557, -0.0063]])
    print(linear.weight.data)
    #kaiming 正态分布初始化

def dm07():
    # 1.创建一个线性层,输入维度5, 输出维度3
    linear = nn.Linear(5, 3)

    # 2 对权重w进行初始化,这是kaiming的 有啥区别
    nn.init.xavier_normal(linear.weight)

    # 4 打印生成结果
#     D:\001 - cs - profile\02 - study - dl - heima\day04\01.
#     参数的初始化.py: 110: FutureWarning: `nn.init.xavier_normal` is now
#     deprecated in favor
#     of
#     `nn.init.xavier_normal_`.
#     nn.init.xavier_normal(linear.weight)
#
#
# tensor([[-0.5477, 0.1713, -0.2435, -1.2390, -0.2838],
#         [0.0621, 0.6306, -0.0988, -0.1411, -0.2766],
#         [0.4019, 0.0305, 0.3554, 0.0850, -0.3516]])
    print(linear.weight.data)
if __name__ == '__main__':
    # dm01()
    # dm02()
    dm07()