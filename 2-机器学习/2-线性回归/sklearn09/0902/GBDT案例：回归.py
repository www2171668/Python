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
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor

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
X = df[['Global_active_power', 'Global_reactive_power', 'Global_intensity']]
Y = df['Voltage']

# 四、数据的划分(将数据划分为训练集和测试集)
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)

# 六、算法模型的选择/算法模型对象的构建
# algo = LinearRegression(fit_intercept=True)
algo = GradientBoostingRegressor(n_estimators=300)
# algo = BaggingRegressor(base_estimator=algo, n_estimators=50)

# 七、算法模型的训练
algo.fit(x_train, y_train)

# 八、模型效果评估
y_hat = algo.predict(x_test)
print("在训练集上的模型效果(回归算法中为R2):{}".format(algo.score(x_train, y_train)))
print("在测试集上的模型效果(回归算法中为R2):{}".format(algo.score(x_test, y_test)))
print("在测试集上的MSE的值:{}".format(mean_squared_error(y_true=y_test, y_pred=y_hat)))

