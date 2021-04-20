# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/19
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.externals import joblib

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 加载模型
ss = joblib.load('./model/ss.pkl')
algo = joblib.load('./model/lr.pkl')

# 2. 使用加载模型对数据做一个预测操作
# a. 加载数据
path = '../datas/household_power_consumption_201.txt'
df = pd.read_csv(path, sep=';')
# b. 数据的清洗
df.replace('?', np.nan, inplace=True)
df.dropna(axis=0, how='any', inplace=True)
# c. 得到需要预测的x
x = df[['Global_active_power', 'Global_reactive_power']]
y = df['Global_intensity']

# d. 使用模型对x做一个预测的操作
y_predict = algo.predict(ss.transform(x))
print("预测结果:")
print(y_predict)

# e. 看一下预测效果
print("预测的效果:{}".format(algo.score(ss.transform(x), y)))
# 画图看一下效果
t = np.arange(len(x))
plt.figure(facecolor='w')
plt.plot(t, y_predict, 'g-', linewidth=2, label=u'预测值')
plt.plot(t, y, 'r-', linewidth=2, label=u'真实值')
plt.legend(loc='lower right')
plt.title('线性回归')
plt.show()
