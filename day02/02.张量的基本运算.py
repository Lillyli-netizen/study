"""
演示张量的基本运算。

涉及到的API:
    add(), sub(), mulO), divO), neg() >加减乘除,取反,substract,multiply,divide
    add_O,sub_O, mL_O, div_O, neg_0  功能同上, 只不过可以改源数据,类献于 Pandas部分的 inplace = True

需要你记忆的:
1.+, -, *, /符号来替代上述的 加减乘除函数.
2.如果是张量 和 数值运算，则:该数值会和张量中的每个值依次进行 对应的运算。
"""
import torch

# 导包

# 1.创建张量
t1 = torch.tensor([1, 2, 3])
# 2.演示加法
# t2 = t1.add(10) #不会修改原数据
t2 = t1 + 10

t2 = t1.add_(10) #修改原数据
t1 += 10 #修改原数据

# 其他函数效果同上 ai补上每个代码示例 注意除法有小数 还有一个//
# 3.打印结果
print(f"t1:{t1}")
print(f"t2:{t2}")
