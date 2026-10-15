"""
案例:
演示张量的拼接。

涉及到的API:
- cat(): 在指定维度上拼接张量，要求其他维度必须一致
- stack(): 在新维度上堆叠张量，要求所有维度必须一致
"""
import torch

torch.manual_seed(24)

# 1.定义张量
t1 = torch.randint(1, 10, (2, 3))
t2 = torch.randint(1, 10, (2, 3))
print(f't1: {t1}, shape: {t1.shape}')
print(f't2: {t2}, shape: {t2.shape}')
print('-' * 30)

# 2.演示张量的拼接
# 思路1: cat()拼接张量
# 在dim=0上拼接 (2,3)+(2,3)=(4,3)
t3 = torch.cat([t1, t2], dim=0)
print(f't3: {t3}, shape: {t3.shape}')

# 在dim=1上拼接 (2,3)+(2,3)=(2,6)
t4 = torch.cat([t1, t2], dim=1)
print(f't4: {t4}, shape: {t4.shape}')

# 在dim=-1上拼接 (效果同dim=1)
t5 = torch.cat([t1, t2], dim=-1)
print(f't5: {t5}, shape: {t5.shape}')

# dim=2会报错，因为张量只有2个维度(0和1)
# t6 = torch.cat([t1, t2], dim=2)  # 错误
print('-' * 30)

# 思路2: stack()拼接张量，在新维度上堆叠拼接时绝不能有未知的维度
# 在dim=0上堆叠 (2,3)+(2,3)=(2,2,3)
t7 = torch.stack([t1, t2], dim=0)
print(f't7: {t7}, shape: {t7.shape}')

# 在dim=1上堆叠 (2,3)+(2,3)=(2,2,3)
t8 = torch.stack([t1, t2], dim=1)
print(f't8: {t8}, shape: {t8.shape}')

# 在dim=2上堆叠 (2,3)+(2,3)=(2,3,2)
t9 = torch.stack([t1, t2], dim=2)
print(f't9: {t9}, shape: {t9.shape}')