"""
案例:

演示神经网络搭建流程.

深度学习案例的4个步骤:

1.准备数据.

2.搭建神经网络

3.模型训练

4.模型测试

神经网络搭建流程:

1.定义一个类，继承:nn.Module

2.在_init__(方法中，搭建神经网络.

3.在forward()方法中.完成:前向传播.
这里试了好几次最后alt+enter直接安装好了
"""
import torch
import torch.nn as nn
from torchsummary import summary

# todo 1.搭建神经网络,自定义继承 nn.Module

class ModelDemo(nn.Module):
    # 1.在init魔法方法中，完成初始化:父类成员，及 神经网络搭建

    def __init__(self):
        # 1.1初始化父类成员.
        super().__init__()
        # 1.2搭建神经网络 一隐藏层+输出层
        # 隐藏层1:输入特征数3，输出特征数3 这个类名可以随便写
        self.linear1 = nn.Linear(3, 3)
        # 隐藏层2:输入特征数3，输出特征数2
        self.linear2 = nn.Linear(3, 2)
        # 输出:输入特征数2，输出特征数2
        self.output= nn.Linear(2, 2)
        # 老师说以后就写到以上部分
        # 1.3 对隐藏层进行初始化 注意不能只写linear12这些名称 self也得带上 为啥这就得是权重0
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        # 隐藏层2
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    # todo: 1.2 前向传播:输入层一>隐藏层一输出层
    # 这个只能是forward 因为这得自动调用其他调不出来 x是什么
    def forward(self, x):
        # 1.1 第一层隐藏层计算: 加权求和 + 激活函数
        # 分解版写法
        x = self.linear1(x)
        x = torch.sigmoid(x)

        # 合并版写法 以后都是这样写的
        x = torch.sigmoid(self.linear1(x))
        # 1.2 隐藏层计算: 加权求和和relu
        x = torch.relu(self.linear2(x))
        # 1.3 第三层 输出层计算: 加权求和和softmax
        # dim=-1这里表示什么? 老师说是按行计算
        x = torch.softmax(self.output(x), dim=-1)
        # 1.4 返回预测值
        return x

# todo 2.模型训练
# 注意和class平级
def train():
    # 1. 创建模型对象
    my_model = ModelDemo()
    print(f'my_model:{my_model}')
    # 2. 创建数据集样本,随机生成 为什么随机生成的数字都是01左右的 没有342 -233这种?
    data = torch.randn(size=(5, 3))
    print(f'data:{data}')
    print(f'shape:{data.shape}')
    print(f'data.requires_grad:{data.requires_grad}') #默认false

    # 3. 调用神经网络模型- 进行模型训练
    output = my_model(data)  #自动调用forward 进行前向
    print(f'output:{output}')
    print(f'output:{output.shape}')
    print(f'output.requires_grad:{output.requires_grad}')
    # 默认false

    # 4. 计算和打印模型参数
    summary(my_model, input_size=(5, 3))

    for name, param in my_model.named_parameters():
        print(f'name:{name}')
        print(f'param:{param} \n')
# todo 3.测试
if __name__ == '__main__':
    train()

