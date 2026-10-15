"""
案例:
alt+回车导包
alt+shift可以把代码往下调整
演示如何创建全0，全1，指定值的张量.

涉及到的函数如下:

torch.ones 和 torch.ones_like 创建全1张量
torch.zeros 和 torch.zeros_like 创建0张量
torch.full 和torch.full_like 创建全为指定值张量

需要你掌握的函数:

zeros(),full()
"""

# 导包
import torch
# 场景torch.ones 和 torch.ones_like 创建全1张量
#3行2列 全1矩阵
t1 = torch.ones(2, 3)
print(f"t1:{t1}，{type(t1)}")
print('-' * 30)
# t2:3行2列

t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f't2: {t2}, type: {type(t2)}')
print('-' * 30)

#t3 -> 基于t2的形状，创建全1张量.
t3 = torch.ones_like(t2)

print(f't3: {t3}, type: {type(t3)}')
print('-' * 30)




# 场景torch.zeros 和 torch.zeros_like 创建0张量
t1 = torch.zeros(2, 3)
print(f"t1:{t1}，{type(t1)}")
print('-' * 30)
# t2:3行2列

t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f't2: {t2}, type: {type(t2)}')
print('-' * 30)

#t3 -> 基于t2的形状，创建全1张量.
t3 = torch.zeros_like(t2)

print(f't3: {t3}, type: {type(t3)}')
print('-' * 30)

# 场景torch.full 和torch.full_like 创建全为指定值张量
# ai能讲一下这个语法吗 2行3列全255张量
# 越接近255越白 什么灰度图ai讲讲
t1 = torch.full((2, 3), fill_value=255)
print(f"t1:{t1}，{type(t1)}")
print('-' * 30)
# t2:3行2列

t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f't2: {t2}, type: {type(t2)}')
print('-' * 30)

#t3 -> 基于t2的形状，创建全1张量.
t3 = torch.full_like(t2, 255)

print(f't3: {t3}, type: {type(t3)}')
print('-' * 30)