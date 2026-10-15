"""
numpy对象->张量Tensor->数据集对象TensorDataset>数据加载器DataLoader

"""
#导入相关模块

import torch

from torch.utils.data import TensorDataset #构造数据集对象

from torch.utils.data import DataLoader #数据加载器

from torch import nn #nn模块中有平方损失函数和假设函数

from torch import optim #optim模块中有优化器函数

from sklearn.datasets import make_regression #创建线性回归模型数据集

import matplotlib.pyplot as plt #可视化

plt.rcParams['font.sans-serif']= ['SimHei'] # 用来正常显示中文标签 如果是mac换成什么consola 你看看老师说的啥意思

plt.rcParams['axes.unicode_minus']= False # 用来正常显示负号

# 1.定义函数,创建线性回归样本数据
def create_dataset():
    x, y, coef = make_regression(
        n_samples=100, #100条样本(100个样本点)
        n_features=1, #1个特征(1个特征点)
        noise= True, #噪声, 噪声越大样本点越散,噪声越小,
        coef=True, #是否返回系数,默认是false, 返回值是None
        bias=14.5, #偏置 这个是可以随便创建吗ai
        random_state=3 #随机种子, 随机种子相同, 输出数据相同

    )
    print(type(x))
    # 2.把上述的数据封装成张量对象
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    # 3.返回结果
    return x, y, coef

# 2.定义函数,表示模型训练
# 构建优化函数的其他方法ai也可以多扩展一下
def train(x, y, coef):
    # 1.创建数据集对象,把tensor-数据集对象-数据加载器
    dataset = TensorDataset(x, y)

    # 2.创建数据加载器对象
    # 参1:数据集对象, 参2:批次大小, 参3:是否打乱数据(训练集打乱 测试不打乱 为啥ai讲讲)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3.创建初始的线性回归模型 注意大写
    # 参1:输入特征维度，参2:输出特征维度。
    model = nn.Linear(1,1)

    # 4.创建损失函数对象
    criterion = nn.MSELoss()

    # 5.创建优化器对象
    # 参1:模型参数，参2:学习率.
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # 6.具体的训练过程
    # 6.1 定义变量，分别表示:训练轮数，每轮的(平均)损失值，训练总损失值，训练的样本数. 设置0.0便于获得除法
    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0
    # 6.2开始训练,按论训练
    for epoch in range(epochs):
        # 6.3每轮是分批次训练的, 所以从数据加载器中获取批次数据
        for train_x, train_y in dataloader: ## 7批 (16, 16, 16, 16, 16, 16, 4)
            # 6.3.1 模型预测
            y_pred = model(train_x)
            # 6.3.2 计算每批的平均损失
            loss = criterion(y_pred, train_y.reshape(-1, 1)) #-1自动计算 这是啥意思
            # 6.3.3 计算总损失和样本批次数 每轮的平均损失
            total_loss += loss.item()
            total_sample += 1
            # 6.3.4 梯度清零+反向传播+梯度更新
            optimizer.zero_grad() #梯度清零 这个操作在干嘛 老师说已经内部进行非0判断
            loss.backward()  #反向传播,计算损失
            optimizer.step()  #更新参数
            # 6.4 把本轮的平均损失值,添加到列表中
        loss_list.append(total_loss / total_sample)
        print(f'轮数:{epoch + 1}, 平均损失值:{total_loss / total_sample}')
        # 重置总损失和样本数，为下一轮做准备
        total_loss, total_sample = 0.0, 0

    # 7. 打印最终的训练结果
    print(f'{epochs}轮的平均损失分别为:{loss_list}')
    print(f'模型参数, 权重:{model.weight}, 偏置:{model.bias}')
    # 8.绘制损失曲线
    plt.plot(range(epochs), loss_list)
    plt.title('损失值曲线变化图')
    plt.grid()
    plt.show()
    # 9.绘制预测值和真实值的关系:
    # 9.1绘制样本点分布情况.
    # plt.scatter(x, y)
    plt.scatter(x.numpy(), y.numpy())
    # 9.2绘制训练模型的预测值
    # x:100个样本点的特征 我这样照着老师的写ai说不行 我也运行不了
    # y_pred = torch.tensor(data= [v * model.weight +model.bias for v in x])
    y_pred = model(x) #这样写也不对
    y_pred = model(x).detach().numpy()
    # 9.3计算真实值
    # y_true = torch.tensor(data= [ y * coef + 14.5 for  v in x])
    y_true = x * coef + 14.5 #这样写也不对
    y_true = (x * coef + 14.5).detach().numpy()
    # 9.4绘制预测值 和 真实值的 折线图.

    plt.plot(x, y_pred, color = 'red', label = '预测')

    plt.plot( x, y_true, color = 'green', label = '真实')
    # 9.5图例，网格. legend干啥的
    plt.legend()
    plt.grid()
    # 9.6 显示图像.
    plt.show()
if __name__ == '__main__':
    # 这里要让xycoef接收函数值 结果一个x对应一个y
    x, y, coef = create_dataset()
    # 为什么这里不传真实值也可以
    train(x, y, coef)
    print(f'x:{x}, y:{y}, coef:{coef}')