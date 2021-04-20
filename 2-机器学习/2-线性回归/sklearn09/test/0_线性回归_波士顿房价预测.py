#!/usr/bin/env python
# -*- encoding:utf-8 -*-

"""
@Author  :   Q.W.Wang
@Software:   PyCharm
@File    :   0_线性回归_波士顿房价预测.py
@Time    :   2018-08-29 23:21
"""

"""
采用线性回归模型进行波士顿房价预测
"""

# 1、导包
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV, LassoCV
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# 2、解决中文显示与pycharm中显示问题
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
np.set_printoptions(linewidth=1000, edgeitems=1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 3、导入数据
# path = r'../../../datas/0_regression/boston_housing.data'
path = r'boston_housing.data'
df = pd.read_csv(path, header=None, )
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
new_df = np.empty((df.size, 14))  # 构建一个数组进行存放处理好的数据
for i, d in enumerate(df.values):
    d = map(float, filter(lambda x: x != '', d[0].split(' ')))
    new_df[i] = list(d)
# print(new_df.shape)
X, Y = np.split(new_df, (13,), axis=1)
# print('Y->',type(Y))
# print('Y->',Y.shape)
# print('X->',type(X))
# print('X->',X.shape)

# 4、测试机与训练集进行数据划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1201)
# 5、数据的标准化处理
ss = StandardScaler()
X_train = ss.fit_transform(X_train, Y_train)
X_test = ss.transform(X_test)
# joblib.dump(ss,r"../../../model/0_regression/波士顿房价预测_ss.model")

# print(Y_train)

# 5、模型构建
models = [
    Pipeline(
        [  ## 线性回归
            ('Poly', PolynomialFeatures()),
            ('Linear', LinearRegression())
        ]
    ),
    Pipeline(
        [  ## Lasso回归
            ('Poly', PolynomialFeatures()),
            ('Linear', LassoCV(alphas=np.logspace(-3, 1, 20)))
        ]
    ),
    Pipeline(
        [
            ## Ridge回归
            ('Poly', PolynomialFeatures()),
            ('Linear', RidgeCV(alphas=np.logspace(-3, 1, 10)))
        ]
    ),
    Pipeline(
        [
            ## ElasticNet回归：l1_ratio -> L1-norm占比，alphas为超参
            ('Poly', PolynomialFeatures()),
            ('Linear', ElasticNetCV(l1_ratio=np.logspace(-3, 1, 2), alphas=np.logspace(-3, 1, 2)))
        ]
    )
]
## 设置模型参数
paramters = {
    "Poly__degree": [3, 2, 1],
    "Poly__interaction_only": [False, True],
    "Poly__include_bias": [True, False],
    "Linear__fit_intercept": [True, False]
}

# 模型训练求解出最优解
title = ['Lr', 'Lasso', 'Rigde', 'ElasticNet']
fig = plt.figure(figsize=(20, 10), facecolor='w')
for i in range(3, 4):  # len(models)
    print("{}回归".format(title[i]))
    model = GridSearchCV(models[i], param_grid=paramters, cv=5, n_jobs=1)
    # 进行网格预测
    model.fit(X_train, Y_train)
    Y_Hat = model.predict(X_test)
    print("%s算法:最优参数:" % title[i], model.best_params_)
    print("%s算法:R值=%.3f" % (title[i], model.best_score_))
    print("{}回归的正确率:{}".format(title[i], str(model.score(X_test, Y_test))))
    sub = plt.subplot(len(models), 1, i + 1)
    t = np.arange(len(X_test))
    # lw是给定线条大小，所以必须是数字
    sub.plot(t, Y_test, ls='-', lw=2, c='r')
    sub.plot(t, Y_Hat, ls='-', lw=2, c='#%06x' % int(int('0xff0000', 16) + 255 * 2 * i))
plt.show()
