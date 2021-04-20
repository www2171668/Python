# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/25
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


def date_format(dt):
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


# 加载数据
path = '../datas/household_power_consumption_200.txt'  ## 200行数据
path = '../datas/household_power_consumption_1000.txt'  ## 1000行数据
df = pd.read_csv(path, sep=';', low_memory=False)

# 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率
names2 = df.columns
names = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
         'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# 异常数据处理(异常数据过滤)
new_df = df.replace('?', np.nan)
datas = new_df.dropna(axis=0, how='any')  # 只要有数据为空，就进行删除操作

# 获取x和y变量, 并将时间转换为数值型连续变量
X = datas[names[0:2]]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas[names[4]]
X = X.astype(np.float)
Y = Y.astype(np.float)

# 对数据集进行测试集合训练集划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 1. 做一个维度多项式扩展的操作
# degree：给定模型做几阶的多项式扩展，也就是转换后的最高次项是多少
poly = PolynomialFeatures(degree=3)
# fit_transform：首先使用给定的数据集进行模型训练，找出模型的转换函数，然后使用找出的转换函数对给定的X数据做一个转换操作
X_train = poly.fit_transform(X_train, Y_train)
X_test = poly.transform(X_test)

# 2. 做一个线性回归
#  fit_intercept：是否训练模型的截距项，默认为True，表示训练；如果设置为False，表示不训练。
algo = LinearRegression(fit_intercept=True)

# 七、算法模型的训练
algo.fit(X_train, Y_train)

# 7.1 查看训练好的模型参数
print("线性回归的各个特征属性对应的权重参数θ:{}".format(algo.coef_))
print("线性回归的截距项的值:{}".format(algo.intercept_))

# 八、模型效果评估
y_hat = algo.predict(X_test)
print("在训练集上的模型效果(回归算法中为R2):{}".format(algo.score(X_train, Y_train)))
print("在测试集上的模型效果(回归算法中为R2):{}".format(algo.score(X_test, Y_test)))
print("在测试集上的MSE的值:{}".format(mean_squared_error(y_true=Y_test, y_pred=y_hat)))
# 画图看一下效果
t = np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测值')
plt.plot(t, Y_test, 'r-', linewidth=2, label=u'真实值')
plt.legend(loc='lower right')
plt.title('线性回归')
plt.show()
