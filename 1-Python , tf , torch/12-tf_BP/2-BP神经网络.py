# -- encoding:utf-8 --
"""
Create by ibf on 2018/5/12
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
df = pd.read_csv('traffic_data.csv', encoding='utf-8')
# print(df.head())

# 2. 读取特征属性X和标签Y，都是DataFrame
x = df[['人口数', '机动车数', '公路面积']]
y = df[['客运量', '货运量']]

# 3. 模型预处理 - 归一化
# 因为x和y的数据取值范围太大了，所以做一个归一化操作(使用区间缩放法)
x_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))
x = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y)

# 方便后面和w进行矩阵的乘法
sample_in = x.T   # 使成为一列一个样本；神经网络输入的时候是按照样本属性输入的
sample_out = y.T

# 超参数
input_number = 3   # 输入的特征数目
hidden_unit_number = 8   # 隐层节点数
out_number = 2   # 输出的目标属性数目

max_epochs = 60000   # 迭代次数
learn_rate = 0.035   # 学习率
mse_final = 6.5e-4   # MSE阈值
sample_number = x.shape[0]   # 样本数量




# 单隐层网络参数，以随机数为初始参数
# 8*3的矩阵，与sample_in （3*None）矩阵进行点乘
W1 = 0.5 * np.random.rand(hidden_unit_number, input_number) - 0.1   # 理解为W1[i]对应隐层节点i
# 8*1的矩阵
b1 = 0.5 * np.random.rand(hidden_unit_number, 1) - 0.1
# 2*8的矩阵
W2 = 0.5 * np.random.rand(out_number, hidden_unit_number) - 0.1   # W2[i]对应输出节点i
# 2*1的矩阵
b2 = 0.5 * np.random.rand(out_number, 1) - 0.1

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


mse_history = []
# BP的计算
for i in range(max_epochs):
    # ①、FP过程
    # 隐藏层的输出
    hidden_out = sigmoid(np.dot(W1, sample_in).transpose() + b1.transpose()).transpose()   # 注意矩阵只能和行向量迭代相加 （None*8）+（1*8）矩阵 ★
    # 输出层的输出（为了简化我们的写法，输出层不进行sigmoid激活）
    network_out = (np.dot(W2, hidden_out).transpose() + b2.transpose()).transpose()   # 2*None矩阵

    # 计算错误率
    err = sample_out - network_out   # 注意矩阵只能和行向量相减
    mse = np.average(np.square(err))
    mse_history.append(mse)   # 用于绘制误差曲线图
    if mse < mse_final:   # 迭代停止条件
        break

    # ②、BP过程  d表示梯度
    dW2 = -np.dot(err, hidden_out.transpose())   # 前面没有使用sigmoid，所以这里也不用 2*8矩阵
    db2 = -np.dot(err, np.ones((sample_number, 1)))   # b不更新

    # E对隐层的偏导
    delta1 = -np.dot(W2.transpose(), err) * hidden_out * (1 - hidden_out)   # 权重值是矩阵，所以放前面乘了 8*None矩阵
    dW1 = np.dot(delta1, sample_in.transpose())   # 8*3矩阵
    db1 = np.dot(delta1, np.ones((sample_number, 1)))   # b不更新

    W2 -= learn_rate * dW2   # 更新参数
    b2 -= learn_rate * db2
    W1 -= learn_rate * dW1
    b1 -= learn_rate * db1

# # 误差曲线图
# mse_history10 = np.log10(mse_history)   # 减小图示范围
# min_mse = min(mse_history10)
# plt.plot(mse_history10)   # MSE变化曲线
# plt.plot([0, len(mse_history10)], [min_mse, min_mse])   # MSE最小值线
# ax = plt.gca()
# ax.set_yticks([-2, -1, 0, 1, 2, min_mse])   # 刻度线
# ax.set_xlabel('iteration')
# ax.set_ylabel('MSE')
# ax.set_title('Log10 MSE History')
# plt.show()

# 预测图和实际图
# 隐藏层和输出层输出
hidden_out = sigmoid((np.dot(W1, sample_in).transpose() + b1.transpose())).transpose()   # 使用迭代后的W1
network_out = (np.dot(W2, hidden_out).transpose() + b2.transpose()).transpose()

# -》.inverse_transform()：训练反转（进行归一化的反转），获取原预测数据   ★
network_out = y_scaler.inverse_transform(network_out.T)   # network_out.T还原为一行一个样本 None*2矩阵
sample_out = y_scaler.inverse_transform(y)   # 原数据

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))   # 子图
# 图一：客流模拟 [0]
line1, = axes[0].plot(network_out[:, 0], 'k', marker='o')   # 预测图
line2, = axes[0].plot(sample_out[:, 0], 'r', markeredgecolor='b', marker='*', markersize=9)   # 实际图
axes[0].legend((line1, line2), ('预测值', '实际值'), loc='upper left')
axes[0].set_title('客流模拟')
# 图二：货流模拟 [1]
line3, = axes[1].plot(network_out[:, 1], 'k', marker='o')
line4, = axes[1].plot(sample_out[:, 1], 'r', markeredgecolor='b', marker='*', markersize=9)
axes[1].legend((line3, line4), ('预测值', '实际值'), loc='upper left')
axes[1].set_title('货流模拟')
plt.show()
