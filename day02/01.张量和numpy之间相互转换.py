"""
演示张量 和 numpy之间如何相互转换，以及 如何从标量张量中提取其内容.
涉及到的API:
场景1:张量->numpy  nd数组对象
    张量对象.numpy() 共享内存
    张量对象.numpy().copy() 不共享内存，链式编程写法。

场景2:numpy->nd数组 张量.
    from_numpy() 共享内存
    torch.tensor(nd数组) 不共享内存

场景3:从标量张量中 提取其内容。
    标量张量.item()

掌握:
张量->numpy:张量对象.numpy()
numpy->张量:torch.tensor(nd数组)
从标量张量中 提取其内容: 标量张量.item()
"""
import numpy as np
import torch


# 1.定义函数，演示：张量->numpy
def dm01():
    # 1.创建张量
    t1 = torch.tensor([1, 2, 3, 4, 5])
    print(f"t1:{t1}，(元素)类型：{t1.dtype}, (张量)类型：{type(t1)}")
    # 2.张量-numpy
    # n1 = t1.numpy() #共享内存
    n1 = t1.numpy().copy()  #不共享内存，t1不变 n1变了
    print(f"n1:{n1}，(元素)类型：{n1.dtype}, (张量)类型：{type(n1)}")
    # 3.演示上述方式 是否共享内存
    n1[0] = 100
    print(f"t1:{t1}")
    print(f"n1:{n1}")
# 2.定义函数，演示：numpy->张量
def dm02():
    # 1.创建numpy数组
    n1 = np.array([11, 22, 33])
    print(f"n1:{n1}，(元素)类型：{n1.dtype}, (张量)类型：{type(n1)}")
    # 2. 把上述numpy数组转换成张量 为什么又转成tenser 第一个函数不行吗
    # t1 = torch.from_numpy(n1).type(torch.float32) #转换+转类型
    t1 = torch.from_numpy(n1)
    print(f"t1:{t1}，(元素)类型：{t1.dtype}, (张量)类型：{type(t1)}")
    t2 = torch.tensor(n1)
    print(f"t1:{t2}，(元素)类型：{t2.dtype}, (张量)类型：{type(t2)}")
    # 3.演示上述方式是否共享内存 发现第二种不共享内存
    n1[0] = 99
    print(f"n1:{n1}")
    print(f"t1:{t1}")
    print(f"t2:{t2}")



# 3.定义函数，演示：从标量张量（只有一个值）中提取内容
def dm03():
    # 1.创建张量 发现数据类型跟传入的数据改变
    t1 = torch.tensor(100.34)
    # t1 = torch.tensor([2, 3])
    print(f"t1:{t1}，(元素)类型：{t1.dtype}, (张量)类型：{type(t1)}")
    # 2.从张量中提取内容
    value = t1.item()
    print(f"value:{value}，(张量)类型：{type(value)}")
# 4.测试
if __name__ == '__main__':
    dm03()