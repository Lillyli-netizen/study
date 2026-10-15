"""
案例:

演示张量的形状操作。

涉及到的API:reshape() 在不改变张量内容的前提下，对其形状做改变。

unsqueeze() 在指定的轴上增加一个(1)维度。

squeeze() 删除所有为1的维度 等价于:降维

transpose() 一次只能交换2个维度。

permute() 一次可以同时交换多个维度。

view() 只能修改连续的张量的形状， 连续张量= 内存中存储顺序 和 在张量中显示的顺序相同.

contiguous() 把不连续的张量连续的张量，即:基于张量中显示的顺序，修改内存中的存储顺序.
is_contiguous() 判断张量是否是连续的

需要你掌握的函数:

reshape(), permute() view() unsqueeze()
"""
import torch
torch.manual_seed(24)
# 1.定义函数，演示：reshape()
def dm01():
    # 1.定义2行3列的张量
    # t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t1 = torch.randint(1, 10, size=(2, 3))
    # 其实我不太懂取到这几个数字的目的
    print(f't1:{t1}, shape:{t1.shape}, row:{t1.shape[0]}, column:{t1.shape[1]}, {t1.shape[-1]},total:{t1.shape[0] * t1.shape[1]}')
    # 2.演示reshape()函数，把t1一>3行2列，1行6列，6行1列. reshape是不改变内容的
    # t2 = t1.reshape(6, 1)
    # t2 = t1.reshape(1, 6)

    t2 = t1.reshape(3, 2)
    print(f't2:{t2}, shape:{t2.shape}')
    # 2.演示reshape()函数，把t1一2行5列的结果.
    # t3 = t1.reshape(2, 5) #这个会报错 ai讲讲
    # print(f't3:{t3}, shape:{t3.shape}')

# 2.定义函数，演示：unsqueeze(),squeeze()
def dm02():
    # 定义2行3列的张量
    t1 = torch.randint(1, 10, size=(2, 3))
    print(f't1:{t1}, shape:{t1.shape}')
    # 2.在0维上,添加一个维度 发现就是增加了一个[],ai能不能讲讲为什么是这样的
    t2 = t1.unsqueeze(0)
    print(f't2:{t2}, shape:{t2.shape}') #[1, 2, 3]

    # 3.在1维上,添加一个维度 发现就是增加了一个[]之外中间空了一行,ai能不能讲讲为什么是这样的 为什么不是2,4
    t3 = t1.unsqueeze(1)
    print(f't3:{t3}, shape:{t3.shape}') #[2, 1, 3]

    # 4.在1维上,添加一个维度 发现是每个数字外面增加了一个[]之外中间空了一行,ai能不能讲讲为什么是这样的
    t4 = t1.unsqueeze(2)
    print(f't4:{t4}, shape:{t4.shape}') #[2, 3, 1]
    # 5.三维上就不可以了
    # 6.删除所有为1的维度
    t6 = torch.randint(1, 10, size=(2, 1, 3, 1, 1))
    print(f't6:{t6}, shape:{t6.shape}') #[2, 1, 3, 1, 1]
    t7 = t6.squeeze()
    print(f't7:{t7}, shape:{t7.shape}') #[2, 3]

# 3.定义函数，演示：transpose(),permute()
def dm03():
    # 1.定义张量
    t1 = torch.randint(1, 10, size=(2, 3, 4))
    print(f't1:{t1}, shape:{t1.shape}') #[2, 3, 4]

    t2 = t1.transpose(0, 1)
    # t2 = t1.transpose(0, 2) 改成-1也可以
    print(f't2:{t2}, shape:{t2.shape}') #[3, 2, 4]
    #3.改变维度从(2,3,4)-(4,2,3)
    t3 = t1.permute(2, 0, 1)
    print(f't3: {t3}, shape: {t3.shape}')
# 4.定义函数，演示：view(),contiguous(), is_contiguous()
def dm04():
# ai讲一下整体是什么思路
    # 1.定义张量
    t1 = torch.randint(1, 10, size=(2, 3))
    print(f'tl: {t1}, shape: {t1.shape}')
    # 2.判断张量是否连续.即:张量中的顺序和 内存中存储顺序是否一致.

    print(t1.is_contiguous()) # True

    # 3.通过 view()函数，修改上述张量的形状.从(2，3)(3，2)
    t2 = t1.view(3, 2)

    print(f't2: {t2}, shape: {t2.shape}')
    print(t2.is_contiguous())  # True

    # 4.通过 transpose()交换维度 交换之后，不连续了.
    t3 = t1.transpose(0, 1)
    print(f't3: {t3}, shape: {t3.shape}')
    print(t3.is_contiguous()) # False
    # 5.尝试把 t3张量 从(3,2),通过 view() 转成(2,3).
    # t4 = t3.view(2, 3)
    # print(f't4: {t4}, shape: {t4.shape}')

    # 6.
    t5 = t3.contiguous().view(2, 3)
    print(t5.is_contiguous()) #True

# 5.测试
if __name__ == '__main__':
    dm04()