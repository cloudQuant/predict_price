from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.layers import Activation, Dense, Dropout, GRU

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.metrics import mean_absolute_error

# 读取CSV文件中的数据

data = pd.read_csv("/content/BTC.csv")

# 选取除“Price”和“Vol.”列外的前六列

data = data.iloc[:, 0:6]

# 将“Price”列作为目标变量y

y = data.loc[:, ['Price']]

# 从原始数据中删除“Price”和“Vol.”列

data = data.drop(['Price', 'Vol.'], axis='columns')

# 打印处理后的数据的前五行

print(data.head(5))

# 打印目标变量y的前五行

print(y.head(5))

data = data.set_index('Date')  # 将'Date'列设置为数据框的索引

data.index = pd.to_datetime(data.index, unit='ns')  # 将索引转换为datetime64[ns]类型

print(data.index)  # 打印转换后的日期索引

aim = 'Price'  # 设定目标变量为'Price'

X_train = data[300:]  # 将数据框从第300行开始到末尾作为训练集特征

X_test = data[:300]   # 将数据框从开始到第300行（不包括第300行）作为测试集特征

y_train = y[300:]     # 将目标变量从第300行开始到末尾作为训练集标签

y_test = y[:300]      # 将目标变量从开始到第300行（不包括第300行）作为测试集标签

print(y_test)         # 打印测试集标签

# line_plot(y_train[aim], y_test[aim], 'training', 'test', title='')

def normalise_zero_base(continuous): return continuous / continuous.iloc[0] - 1

def normalise_min_max(continuous): return (continuous - continuous.min()) / (data.max() - continuous.min())

X_train = normalise_zero_base(X_train)

X_test = normalise_zero_base(X_test)

y_train = normalise_zero_base(y_train)

y_test = normalise_zero_base(y_test)

import numpy as np

X_train = np.expand_dims(X_train, axis=1)

X_test = np.expand_dims(X_test, axis=1)

from tensorflow import keras  # 从 TensorFlow 导入 Keras 模块

# 定义 GRU 模型

gruMODEL = keras.Sequential()  # 创建一个顺序模型

# 添加第一个 GRU 层，并应用了 Dropout 正则化

gruMODEL.add(keras.layers.GRU(

    units=1024,  # 隐藏层单元数为 1024

    input_shape=(1, 3),  # 输入形状为 (时间步长, 特征数)，这里时间步长为 1，特征数为 3

    activation='PReLU',  # 激活函数为 PReLU（参数化修正线性单元）

    recurrent_activation="sigmoid",  # 循环层激活函数为 sigmoid

    use_bias=True,  # 使用偏置项

    kernel_initializer="glorot_uniform",  # 权重初始化方法为 Glorot 均匀分布

    recurrent_initializer="orthogonal",  # 循环层权重初始化方法为正交矩阵

    bias_initializer="zeros",  # 偏置项初始化方法为 0

    # 以下均为正则化和约束选项，此处未使用

    kernel_regularizer=None,

    recurrent_regularizer=None,

    bias_regularizer=None,

    activity_regularizer=None,

    kernel_constraint=None,

    recurrent_constraint=None,

    bias_constraint=None,

    dropout=0.0,  # 在输入层上丢弃的输入单元比例，此处未使用

    recurrent_dropout=0.0,  # 在循环层上丢弃的单元比例，此处未使用

    return_sequences=False,  # 是否返回完整序列的输出，这里只返回最后一个时间步的输出

    return_state=False,  # 是否返回最后一个时间步的状态，此处不返回

    go_backwards=False,  # 是否反向处理输入序列

    stateful=False,  # 如果为 True，则批次中的每个样本的索引 i 的状态将被用作下一个批次中索引 i 的样本的初始状态

    unroll=False,  # 如果为 True，则网络将展开，否则使用符号循环

    time_major=False,  # 输入和输出的形状格式，False 表示 (batch, time, features)，True 表示 (time, batch, features)

    reset_after=True  # 是否在每个时间步重置门控单元

))

# 添加 Dropout 层，丢弃率为 0.9

gruMODEL.add(keras.layers.Dropout(0.9))

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import GRU, Dropout

# 初始化 GRU 模型

gruMODEL = Sequential()

# 第一层 GRU，并添加 Dropout 正则化

gruMODEL.add(GRU(

    units=1024,  # 隐藏单元数量

    input_shape=(1, 3),  # 输入数据的形状

    activation='PReLU',  # 激活函数

    recurrent_activation="sigmoid",  # 循环层激活函数

    use_bias=True,  # 是否使用偏置项

    kernel_initializer="glorot_uniform",  # 权重初始化方法

    recurrent_initializer="orthogonal",  # 循环权重初始化方法

    bias_initializer="zeros",  # 偏置项初始化方法

    # 正则化、约束等参数均设置为 None，即不使用

    dropout=0.0,  # 在每个时间步的输入之间丢弃一部分单元

    recurrent_dropout=0.0,  # 在每个时间步的循环层之间丢弃一部分单元

    return_sequences=False,  # 是否返回输出序列的最后一个时间步

    return_state=False,  # 是否返回最后一个时间步的状态

    go_backwards=False,  # 是否反向处理输入序列

    stateful=False,  # 如果为 True，则批次中的每个样本将具有其自己的内部状态

    unroll=False,  # 如果为 True，则网络将展开其计算图，这可能会消耗更多的内存但会加速计算

    time_major=False,  # 输入数据的形状格式，此处为（batch, time, features）

    reset_after=True  # 在计算输出时，是否重置隐藏状态

))

gruMODEL.add(Dropout(0.9))  # 添加 Dropout 层，丢弃率为 0.9

# 第二层 GRU

gruMODEL.add(GRU(

    units=2048,  # 隐藏单元数量

    activation='PReLU',  # 激活函数

    recurrent_activation="sigmoid",  # 循环层激活函数

    use_bias=True,  # 是否使用偏置项

    kernel_initializer="glorot_uniform",  # 权重初始化方法

    recurrent_initializer="orthogonal",  # 循环权重初始化方法

    bias_initializer="zeros",  # 偏置项初始化方法

    # 正则化、约束等参数均设置为 None，即不使用

    dropout=0.0,  # 在每个时间步的输入之间丢弃一部分单元

    recurrent_dropout=0.0,  # 在每个时间步的循环层之间丢弃一部分单元

    return_sequences=False,  # 是否返回输出序列的最后一个时间步

    return_state=False,  # 是否返回最后一个时间步的状态

    go_backwards=False,  # 是否反向处理输入序列

    stateful=False,  # 如果为 True，则批次中的每个样本将具有其自己的内部状态

    unroll=False,  # 如果为 True，则网络将展开其计算图

    time_major=False,  # 输入数据的形状格式

    reset_after=True  # 在计算输出时，是否重置隐藏状态

))

gruMODEL.add(Dropout(0.8))  # 添加 Dropout 层，丢弃率为 0.8

# 第三层GRU层

gruMODEL.add(GRU(
units=4096, # 神经元数量
activation='PReLU', # 激活函数为PReLU
recurrent_activation="sigmoid", # 递归激活函数为sigmoid
use_bias=True, # 使用偏置项
kernel_initializer="glorot_uniform", # 权重初始化方法
recurrent_initializer="orthogonal", # 递归权重初始化方法
bias_initializer="zeros", # 偏置初始化方法
kernel_regularizer=None, # 权重正则化方法
recurrent_regularizer=None, # 递归权重正则化方法
bias_regularizer=None, # 偏置正则化方法
activity_regularizer=None, # 激活正则化方法
kernel_constraint=None, # 权重约束
recurrent_constraint=None, # 递归权重约束
bias_constraint=None, # 偏置约束
dropout=0.0, # dropout层比例
recurrent_dropout=0.0, # 递归dropout层比例
return_sequences=False, # 是否返回输出序列的最后一个输出
return_state=False, # 是否返回最后一个状态
go_backwards=False, # 是否反向计算
stateful=False, # 是否是有状态的网络
unroll=False, # 是否展开网络
time_major=False, # 输入张量的形状格式
reset_after=True)) # 是否在每个时间步之后重置门控单元
gruMODEL.add(Dropout(0.09)) # 添加dropout层，防止过拟合

# 第四层GRU层

gruMODEL.add(GRU(
units=512, # 神经元数量
activation='PReLU', # 激活函数为PReLU
recurrent_activation="sigmoid", # 递归激活函数为sigmoid
# ...（以下参数与第三层相同，省略）
reset_after=True))
gruMODEL.add(Dropout(0.09)) # 添加dropout层，防止过拟合

# 输出层

gruMODEL.add(Dense(units=1)) # 添加全连接层，输出单元数为1

# 编译RNN

gruMODEL.compile(optimizer="sgd", # 使用随机梯度下降优化器
loss='mean_squared_error') # 使用均方误差作为损失函数
gruMODEL.summary() # 打印模型概览信息