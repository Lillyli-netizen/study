import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

_, axes = plt.subplots(1, 2)   # 1 行 2 个子图: 左函数图像, 右导数图像

# ---- 左图: 函数图像 ----
x = torch.linspace(-20, 20, 1000)      # 从 -20 到 20 均匀取 1000 个点
y = torch.sigmoid(x)                   # 换 torch.tanh / torch.relu 即画另外两个
axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title('Sigmoid 函数图像')

# ---- 右图: 导数图像 (借助自动微分求导) ----
x = torch.linspace(-20, 20, 1000, requires_grad=True)   # 开启梯度追踪
torch.sigmoid(x).sum().backward()      # sum 后反向传播
axes[1].plot(x.detach(), x.grad)       # 横轴 x 的数值, 纵轴 x.grad (每个点的导数)
axes[1].grid()
axes[1].set_title('Sigmoid 导数图像')

plt.show()
