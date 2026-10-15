"""
案例:

创建指定类型的张量。

涉及到的函数:

type(torch支持的数据类型)
half()/double()/float()/short()/int()/long()
"""
import torch
# 场景1：直接创建指定类型的张量
t1 = torch.tensor([1, 2, 3, 4, 5], dtype = torch.float) #默认是32
print(f"t1:{t1}，(元素)类型：{t1.dtype}, (张量)类型：{type(t1)}")
print('-' * 30)

# 场景2：创建好张量后：类型转换
# 思路1：type()函数，推荐掌握
t2 = t1.type(torch.int16)
print(f"t2:{t2}，(元素)类型：{t2.dtype}, (张量)类型：{type(t2)}")
print('-' * 30)
# 思路2：half()/double()/float()/short()/int()/long()
print(t2.half())  #float16
# ai补全剩下每一种,默认的标记出来