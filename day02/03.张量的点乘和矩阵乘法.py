"""
案例:
演示张量的点乘和 矩阵乘法操作.
这些公式我都比较迷茫 ai讲讲
点乘:
要求:两个张量的维度保持一致，对应元素直接做 相应的操作.
API:
t1 * t2
t1.mul(t2)
#multiply:乘法
矩阵乘法:
要求:两个张量，第一个张量的列数，等于第二个张量的行数(A列=B行)
结果:A行B列
API:
t1 @ t2
t1.matmul(t2)
t1.dot(t2)
扩展:只针对于一维张量有效.

"""
import torch


# 1.定义函数,演示张量:点乘
def dm01():
    # 1.定义张量,2行3列 注意要加两层[]
    t1 = torch.tensor([[1, 2, 3], [3, 4, 5]])
    print(f"t1:{t1}")

    t2 = torch.tensor([[1, 2, 3], [3, 4, 5]])
    print(f"t2:{t2}")

    # 3.演示张量点乘 就是在原本的位置做乘法 两种方式均可

    # t3 = t1 * t2
    t3 = t1.mul(t2)
    print(f"t3:{t3}")
# 2.定义函数,演示张量:矩阵乘法
def dm02():
    # 条件 a列=b行 结果 a行b列 ai讲讲
    # 1.定义张量,2行3列 注意要加两层[]
    t1 = torch.tensor([[1, 2, 3], [3, 4, 5]])
    print(f"t1:{t1}")

    t2 = torch.tensor([[1, 2], [3, 3] ,[4, 5]])
    print(f"t2:{t2}")

    # 3.演示张量矩阵乘法 拿第一行的123 * 对面竖着的 134 ai讲讲

    # t3 = t1 @ t2
    t3 = t1.matmul(t2)  #效果同上，matrix multiply(矩阵乘法)

    # t3 = t1.dot(t2) 一维才可以
    print(f"t3:{t3}")
#5.演示 dot()函数.

# t3 = t1.dot(t2)

#报错，dot()只针对于一维张量有效. 这个又是怎么算的ai讲讲

t4 = torch.tensor([1, 2, 3])

t5 = torch.tensor([4, 5, 6])

t6 = t4.dot(t5)

print(f't6: {t6}')


if __name__ == '__main__':
    dm02()