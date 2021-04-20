# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/19
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 一、加载数据
path = '../datas/household_power_consumption_201.txt'
path = '../datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';')
# df.info()

# 二、数据的清洗
"""
可选操作
比如异常数据的处理，缺省数据的填充....
"""
# inplace: 该参数的作用是给定，是否是在原始的DataFrame上做替换，设置为true表示在原始DataFrame上做替换，直接修改原始的DataFrame的数据格式；默认是False
# df = df.replace('?', np.nan)
df.replace('?', np.nan, inplace=True)
# 删除所有具有非数字的样本
# how: 指定如何删除数据，如果设置为any，默认值, 那么只要行或者列中存在nan值，那么就进行删除操作；如果设置为all，那边要求行或者列中的所有值均为nan的时候，才可以进行删除操作
df.dropna(axis=0, how='any', inplace=True)

# 三、基于业务提取最原始的特征属性X和目标属性Y
X = df[['Global_active_power', 'Global_reactive_power']]
# X.info()
Y = df['Global_intensity']
# print("原始X形状:{}， 原始Y形状:{}".format(X.shape, Y.shape))

# 四、数据的划分(将数据划分为训练集和测试集)
# train_size：给定划分中的训练集占比，一般情况下，我们训练集和测试集的占比一般为8:2，如果样本比较多的时候，可以考虑7:3，如果样本比较少的时候，可以考虑9:1
# test_size: 给定划分中的测试集的占比，其中train_size和test_size只能有且给定其中的一个。
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
# print("训练集样本形状:{}, 测试集样本形状:{}".format(x_train.shape, x_test.shape))

# 五、特征工程
"""
可选操作
主要就是一些：哑编码、TF-IDF、连续数据的离散化、标准化、归一化、特征选择、降维....
NOTE: 所有的特征工程其实和算法模型训练一样，特征工程的意思就是对特征属性做一个数据的转换操作，所以需要获取一个对应数据转换函数---> 需要从训练数据中训练出来
"""
"""
NOTE:
sklearn中，所有特征工程、算法模型的API基本类似，主要是以下几个API：
fit: 模型/函数训练，基于给定的训练集数据，训练模型对象
transform: 使用训练好的模型对给定的数据集做一个对应的数据转换(使用训练好的函数转换数据)，一般出现在特征工程的相关的类中间
predict：使用构建好的算法模型对样本数据做一个预测的操作，其实内部也就是使用训练好的函数转换特征属性到目标属性，一般出现在算法模型的相关的类中间
fit_transform: 也就是fit和transform的结合，底层先使用给定的训练数据训练模型，找出对应的转换函数或者模型对象，然后使用找到的函数或者模型对训练数据做一个转换操作
"""
# 做一下数据的标准化操作
"""
实现方式一：
# a. 创建标准化的对象
ss = StandardScaler()
# b. 训练标准化的转换函数
ss.fit(x_train, y_train)
# c. 使用训练好的标注化模型对数据做一个转换操作
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

"""
# 实现方式二：
# a. 创建标准化的对象
ss = StandardScaler()
# b. 训练标准化的转换函数并使用训练好的函数对训练数据做一个转换操作
x_train = ss.fit_transform(x_train, y_train)
# c. 使用训练好的标注化模型对数据做一个转换操作
x_test = ss.transform(x_test)

# 六、算法模型的选择/算法模型对象的构建
# fit_intercept：是否训练模型的截距项，默认为True，表示训练；如果设置为False，表示不训练。
algo = LinearRegression(fit_intercept=True)

# 七、算法模型的训练
algo.fit(x_train, y_train)

# 7.1 查看训练好的模型参数
print("线性回归的各个特征属性对应的权重参数θ:{}".format(algo.coef_))
print("线性回归的截距项的值:{}".format(algo.intercept_))

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

# 九、模型保存
"""
两种保存方式：
1. 直接将模型输出为二进制的磁盘文件
2. 直接将预测结果y_hat输出到数据库
"""
# 查看python源码的API说明的方式：键盘按着ctrl键，鼠标放到需要查看的API上面，然后左键点击，进入源码即可查看
joblib.dump(ss, './model/ss.pkl')
joblib.dump(algo, './model/lr.pkl')

