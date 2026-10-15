
"""
案例:
演示张量的基本创建方式.
PyTorch框架属于最常用的深度学习框架，无论是后续要学的ANN(人工神经网络)，还是CNN(卷积神经网络)，RNN(循环神经网络)
底层在处理数据时，都是使用 张量 来处理的.
张量->存储同一类型元素的容器，且元素值必须是 数值才可以.

张量的基本创建方式:
torch.tensor 根据指定数据创建张量
torch.Tensor根据形状创建张量，其也可用来创建指定数据的张量
torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量
alt+shift配合鼠标纵向选择

细节：Tensor方式较之于tensor方式，可以基于 形状来直接创建张量.
小写用的多，大写多的那个功能也很少用
123掌握1
技巧：右键 split right 左边定义右边调用
"""

# 导包
import torch
import numpy as np
# 1.定义函数，演示torch.tensor 根据指定数据创建张量
def dm01():
    # 场景1：标量 张量
    t1 = torch.tensor(10)
    print(f"t1:{t1}，{type(t1)}")
    print('-' * 30)
    # 场景2：二维列表->张量
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.tensor(data)
    print(f"t2:{t2}, {type(t2)}")
    print('-' * 30)
    # 场景3：numpy nd数组->向量
    # 这是什么语法ai讲一下
    data = np.random.randint(0,10, size = (2,3))
    # 这一步发现直接float不写torch.也可以  只是结果不写的出现报黄和dtype=torch.float64
    t3 = torch.tensor(data, dtype = torch.float)
    print(f"t3:{t3}, {type(t3)}")
    print('-' * 30)
    # 场景4：尝试直接创建 指定维度（2行3列）张量tensor() takes 1 positional argument but 2 were given
    # 这个就报错，ai讲讲，是不是Tensor就可以了
    # t4 = torch.tensor(2,4)
    # print(f"t4:{t4}, {type(t4)}")
# 2.定义函数，演示torch.Tensor根据形状创建张量，其也可用来创建指定数据的张量
def dm02():
    # 场景1：标量 张量
    t1 = torch.Tensor(10)
    print(f"t1:{t1}，{type(t1)}")
    print('-' * 30)
    # 场景2：二维列表->张量
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.Tensor(data)
    print(f"t2:{t2}, {type(t2)}")
    print('-' * 30)
    # 场景3：numpy nd数组->向量
    # 这是什么语法ai讲一下
    data = np.random.randint(0,10, size = (2,3))
    # 这一步发现大写不能指定类型
    t3 = torch.Tensor(data)
    print(f"t3:{t3}, {type(t3)}")
    print('-' * 30)
    # 场景4：尝试直接创建 指定维度（2行3列）张量Tensor()
    # 这次换成Tensor就可以了
    t4 = torch.Tensor(2,4)
    print(f"t4:{t4}, {type(t4)}")
# 3.定义函数，演示torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量
def dm03():
    # 场景1：标量 张量
    t1 = torch.IntTensor(10)
    print(f"t1:{t1}，{type(t1)}")
    print('-' * 30)
    # 场景2：二维列表->张量
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.IntTensor(data)
    print(f"t2:{t2}, {type(t2)}")
    print('-' * 30)
    # 场景3：numpy nd数组->向量
    # 这是什么语法ai讲一下
    data = np.random.randint(0,10, size = (2,3))
    # 这一步发现大写不能指定类型
    t3 = torch.IntTensor(data)
    print(f"t3:{t3}, {type(t3)}")
    print('-' * 30)
    # 场景4：如果类型不匹配 会尝试自动转换类型
    # ]])为什么老师问该在这里显示类型却不显示，这里该显示吗，， 老师说张量默认float32 ai讲讲 而不是64说目的是省空间
    # 技巧：ctrl+r进行替换
    data = np.random.randint(0,10, size = (2,3))
    t4 = torch.FloatTensor(data)
    print(f"t4:{t4}, {type(t4)}")
    print('-' * 30)
# 4. 定义测试函数
if __name__ == '__main__':
    dm03()