# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/19
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
# path = '../datas/household_power_consumption_200.txt'
path = '../datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';')
# print(df.head(2))
# df.info()

# 2. 获取功率值作为作为特征属性X，电流作为目标属性Y
X = df.iloc[:, 2:4]
# print(X.head(2))
Y = df.iloc[:, 5]

# 3. 获取训练数据和测试数据
n = int(X.shape[0] * 0.8)
train_x = np.array(X[:n])
test_x = np.array(X[n:])
train_y = np.array(Y[:n])
test_y = np.array(Y[n:])
print("总样本数目:{}, 训练数据样本数目:{}, 测试数据样本数目:{}".format(X.shape, train_x.shape, test_x.shape))

# 4. 训练模型
# a. 训练数据转换为矩阵形式
x = np.mat(train_x)
y = np.mat(train_y).reshape(-1, 1)
# b. 训练模型参数θ值
theta = (x.T * x).I * x.T * y
print(theta.shape)
print("求解出来的theta值:{}".format(theta))

# 5. 模型效果评估
y_hat = np.mat(test_x) * theta
# 画图看一下效果
t = np.arange(len(test_x))
plt.figure(facecolor='w')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测值')
plt.plot(t, test_y, 'r-', linewidth=2, label=u'真实值')
plt.legend(loc='lower right')
plt.title('线性回归')
plt.show()

# 6. 模型的存储, 这里可以直接将θ值保存到数据库中，然后再需要的程序中，再讲θ加载到程序中
# TODO: 模拟一下加载的操作
theta1 = theta[0]
theta2 = theta[1]

# 产生预测值
global_active_power = 4.216
global_reactive_power = 0.418
print("当前的输入的特征属性值为:{}---------{}".format(global_active_power, global_reactive_power))
print("预测值为:{}".format(global_active_power * theta1 + global_reactive_power * theta2))
