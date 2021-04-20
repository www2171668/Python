# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/19
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 一、加载数据
path = '../datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';')

# 二、数据的清洗
df.replace('?', np.nan, inplace=True)
df.dropna(axis=0, how='any', inplace=True)

# 三、基于业务提取最原始的特征属性X和目标属性Y
X = df[['Global_active_power', 'Global_reactive_power']]
Y = df['Global_intensity']

# 四、数据的划分(将数据划分为训练集和测试集)
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练集样本形状:{}, 测试集样本形状:{}".format(x_train.shape, x_test.shape))

# 六、算法模型的选择/算法模型对象的构建
lr = LinearRegression(fit_intercept=True)

# 构建集成算法对象
"""
base_estimator=None,： 给定Bagging中的子模型对象
n_estimators=10, ：给定Bagging中的子模型的数目
max_samples=1.0, : 给定Bagging中每个子模型在训练的时候使用多少原始数据
max_features=1.0, : 给定Bagging中每个子模型在训练的时候使用多少特征属性
bootstrap=True, : 给定Bagging子模型的训练数据集采用什么方式产生，设置为True表示进行有放回的重采用产生，设置为False表示直接不放回的随机抽取产生。
bootstrap_features=False : 给定Bagging子模型中训练过程中使用的特征属性如何产生，设置为True表示有放回的重采样，设置为False表示直接不放回的随机抽取特征。
"""
algo = BaggingRegressor(base_estimator=lr, n_estimators=10)

# 七、算法模型的训练
algo.fit(x_train, y_train)

# 7.1 查看训练好的模型参数
# 获取所有的子模型
estimators_ = algo.estimators_
for i, estimator in enumerate(estimators_):
    print("{}：线性回归的各个特征属性对应的权重参数θ:{}".format(i, estimator.coef_))
    print("{}：线性回归的截距项的值:{}".format(i, estimator.intercept_))

# 八、模型效果评估
y_hat = algo.predict(x_test)
print("在训练集上的模型效果(回归算法中为R2):{}".format(algo.score(x_train, y_train)))
print("在测试集上的模型效果(回归算法中为R2):{}".format(algo.score(x_test, y_test)))
print("在测试集上的MSE的值:{}".format(mean_squared_error(y_true=y_test, y_pred=y_hat)))
# 画图看一下效果
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测值')
plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实值')
plt.legend(loc='lower right')
plt.title('线性回归')
plt.show()
