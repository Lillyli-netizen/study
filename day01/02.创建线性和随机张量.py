"""
insert 插入和替换切换
案例:

演示PyTorch中如何创建 线性 和 随机张量.
涉及到的函数:
torch.arange() 和 torch.linspace() 创建性张量
torch.random.initial_seed() f torch.random.manual_seed()随机种子设置
torch.rand/randn()创建随机浮点类型张量
torch.randint(low,high,size=()) 创建随机整数类型张量
要掌探的函数:
arange(), linspace(), manual_seed(), randint()
"""
# 导包
import torch
# 1.定义函数，演示：创建线性张量

def dm01():
    # 场景1：创建指定范围的线性张量
    # 参1：起始值 参2：结束值 参3：步长
    t1 = torch.arange(0, 10, 2)
    print(f"t1:{t1}, {type(t1)}")
    print('-' * 30)
    # 场景2：创建指定范围的线性张量->等差数列
    t2 = torch.linspace(1, 10, 4)
    # 参1：起始值 参2：结束值 参3：元素的个数
    print(f"t2:{t2}, {type(t2)}")
    print('-' * 30)


# 2. 定义函数，演示：创建随机张量
def dm02():
    # step1: 设置随机种子.我的问题是设置了就直接用了吗
    # torch.initial_seed() #默认采用当前系统时间戳作为随机种子
    torch.manual_seed(3) #设置随机种子 ？啥意思 为啥写了就不变了 随便一个参数就可以吗 为什么我和其他人出来的随机也一模一样



    # step2:创建随机张量.
    # 场景1：均匀分布的（01）随机张量
    t1 = torch.rand(size=(2,3))
    print(f"t1:{t1}, {type(t1)}")
    print('-' * 30)
    # 场景2：符合正态分布的随机张量
    t2 = torch.randn(size=(2, 3))
    print(f"t2:{t2}, {type(t2)}")
    print('-' * 30)
    # 场景3：创建随机整数张量
    t3 = torch.randint(1, 10, size=(4, 5))
    print(f"t3:{t3}, {type(t3)}")
    print('-' * 30)
# 3.测试函数
if __name__ == '__main__':
    dm02()