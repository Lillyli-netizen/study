"""
案例:

演示张量常用的运算函数.

涉及到的API(函数)如下:

sum(),max(),min(),mean() 都有dim参数，表示列，1表

pow(), sqrt(), exp(), log(), log2(), log10() 没有idim参数

掌握的函数:

sum(),max(),min(),mean()
"""

# 1.导包
import torch
# 1.定义张量， 记录初值
t1 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print(f't1:{t1}')
# 2.演示 有dim参数 函数
# sum（）求和 ai每个讲讲
print(t1.sum(dim=0))
print(t1.sum(dim=1))
print(t1.sum())
print('-' * 30)

# max求最大值，min同理 ai自己补上讲讲为什么每一行结果是这样的
# torch.return_types.max(
# values=tensor([4, 5, 6]),
# indices=tensor([1, 1, 1]))
# torch.return_types.max(
# values=tensor([3, 6]),
# indices=tensor([2, 2]))
# tensor(6)
print(t1.max(dim=0))
print(t1.max(dim=1))
print(t1.max())
print('-' * 30)
# mean会存在float问题 dtype改掉就可以了 ai加上 pow出现两种 ** 和pow均可 每一种都给示例代码