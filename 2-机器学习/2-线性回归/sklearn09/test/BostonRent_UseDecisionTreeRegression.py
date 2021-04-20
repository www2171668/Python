# -- encoding:utf-8 --
"""
    作业二:
        使用决策树回归算法实现波士顿房租
    Create by 迪力木拉提·吾麦尔 on '2018/8/27 17:24'
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

# 设置字符集
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 加载数据

"""
    names属性:
        "CRIM":城镇人均犯罪率
        "ZN":住宅用地比例超过2.5万平方英尺
        "INDUS":城镇非零售业亩比重
        "CHAS":查尔斯河哑变量（如果，河道边界河流 =1，否则0）
        "NOX":氧化氮浓度（每百万份）
        "RM":平均每个住宅的房间数
        "AGE":1940以前建成单位占有比例
        "DIS":波士顿五个就业中心的加权距离
        "RAD":径向公路可达性指标
        "TAX":10000美元的全价物业税税率
        "PTRATIO":城镇学生与教师比率
        "B":1000（BK - 0.63）^ 2，其中BK是城镇黑人的比例
        "LSTAT": 人口比例低于百分比
        "MEDV":1000美元自有住房的中值

"""

names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
path = './data/boston_housing.data'
df = pd.read_csv(path, sep='\s+', header=None, names=names)
target = pd.DataFrame(load_boston().target)
# print(df.info)
# print(target.info)

# 数据清晰
df.replace('?', np.nan, inplace=True)
df.dropna(axis=0, how='any', inplace=True)
target.replace('?', np.nan, inplace=True)
target.dropna(axis=0, how='any', inplace=True)

# 提取最原始的特征属性X和目标属性Y
X = df.astype(float)
Y = target.astype(float)
# print(X.shape)
# print(Y.shape)

# 数据划分
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=99)
print("训练集样本形状:{}, 测试集样本形状:{}".format(x_train.shape, x_test.shape))

# 特征工程
# ss = StandardScaler()
# ss = MinMaxScaler()
ss = StandardScaler()
x_train = ss.fit_transform(x_train, y_train)
x_test = ss.transform(x_test)

# 算法模型构建
# algorithm = DecisionTreeRegressor(max_depth=20,criterion='mae',splitter='random',min_samples_split=5)
algorithm = DecisionTreeRegressor()

# 算法模型训练
algorithm.fit(x_train, y_train)
# # 查看训练好的模型参数
y_hat = algorithm.predict(x_test)
print("在训练集上的模型效果(回归算法中为R²):{}".format(algorithm.score(x_train, y_train)))
print("在测试集上的模型效果(回归算法中为R²):{}".format(algorithm.score(x_test, y_test)))
print("在测试集上MSE的值:{}".format(mean_squared_error(y_true=y_test, y_pred=y_hat)))
print(algorithm.feature_importances_)

# 画图
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'g-', linewidth=2, label=u'真实值')
plt.plot(t, y_hat, 'r-', linewidth=2, label=u'预测值')
plt.legend(loc='lower right')
plt.title(u"使用决策树回归算法预测波士顿房租")
plt.show()

# 保存模型
joblib.dump(algorithm, './models/treeAlgorithm.model')
print("It's Done!!!")
