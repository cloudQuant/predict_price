import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
step_size = 4  # 步长大小
N = 850        # 总样本数
forecast_start = 720  # 预测开始的索引位置
# 创建一个时间数组 t
t = np.arange(0, N)
# 生成带有正弦波和随机噪声的序列数据 x
x = np.sin(0.03 * t) + 1.2 * np.random.rand(N) + t / 300
# 将数据转换为Pandas DataFrame格式
df = pd.DataFrame(x, columns=['Value'])
# 为了方便后续操作，可以将时间 t 也加入到 DataFrame 中
df['Time'] = t
# 如果只需要原始数据 x 而不需要时间 t，则上面的 'Time' 列可以省略
# 绘制整个数据序列
plt.plot(df['Time'], df['Value'])
# 在预测开始点处绘制一条垂直线
plt.axvline(x=df['Time'][forecast_start], c="r", label="预测开始点")
# 添加图例
plt.legend()
# 显示图形
plt.show()

# 将数据转换为序列和标签，给定序列长度

def create_labels(data, step):

    X = np.array([data[i:i+step] for i in range(len(data) - step)])

    y = np.array(data[step:])

    return X, y

# 准备训练和测试数据

values = df['Value'].values  # 只需要值列，因为我们已经有了时间t作为索引

train, test = values[:forecast_start], values[forecast_start:N]

# 生成序列数据

trainX, trainY = create_labels(train, step_size)

testX, testY = create_labels(test, step_size)

# 将数据重新整形以匹配LSTM的输入

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# 将数据转换为PyTorch张量

trainX_tens = torch.tensor(trainX, dtype=torch.float32)

trainY_tens = torch.tensor(trainY, dtype=torch.float32)

testX_tens = torch.tensor(testX, dtype=torch.float32)

testY_tens = torch.tensor(testY, dtype=torch.float32)

# 为训练集创建DataLoader

train_dataset = torch.utils.data.TensorDataset(trainX_tens, trainY_tens)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)  # 通常训练时会打乱数据


# 定义GRU模型


class GRUModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)

        out = self.fc(out[:, -1, :])  # 提取最后一个时间步的输出

        return out


input_size = step_size  # 输入尺寸设置为步长大小

hidden_size = 128  # 隐藏层尺寸为128

output_size = 1  # 输出尺寸为1

epochs = 100  # 训练轮次为100

learning_rate = 0.0001  # 学习率设置为0.0001

# 实例化GRU模型


model = GRUModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# 定义损失函数和优化器


criterion = nn.MSELoss()  # 使用均方误差损失函数

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器，并设置学习率

# 遍历所有训练轮次

for epoch in range(epochs):

    model.train()  # 设置模型为训练模式

    # 遍历数据加载器中的每一个批次数据

    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()  # 清空之前所有优化参数的梯度

        output = model(batch_X)  # 前向传播，得到模型预测结果

        # 计算当前小批量数据的模型预测值与真实标签之间的损失

        loss = criterion(output, batch_Y)

        # 反向传播，计算损失关于模型参数的梯度

        loss.backward()

        # 使用指定的优化算法根据计算得到的梯度更新模型参数

        optimizer.step()

        # 每10个轮次打印一次损失值

    if (epoch + 1) % 10 == 0:
        print(f'轮次 [{epoch + 1}/{epochs}], 损失值: {loss.item():.4f}')

with torch.no_grad():
    model.eval()  # 设置模型为评估模式

    testPredict = model(testX_tens)  # 使用模型对测试数据进行预测

    # 绘制结果

index = range(len(testY))

plt.plot(index, testY, label="真实值")

plt.plot(index, testPredict.numpy(), label="预测值")

plt.legend()  # 显示图例

plt.show()  # 显示图形