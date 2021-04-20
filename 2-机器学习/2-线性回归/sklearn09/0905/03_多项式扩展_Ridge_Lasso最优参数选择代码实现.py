# -- encoding:utf-8 --
"""
Create by ibf on 2018/9/5
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# 0. 读取数据形成DataFrame并且分割出x和y
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
file_path = '../datas/boston_housing.data'
df = pd.read_csv(filepath_or_buffer=file_path, header=None, names=names, sep='\\s+')

# 1. 数据分割(提取x和y)
x = df[names[:13]]
y = df[names[13]]
y = y.ravel()
print("样本数据量:%d，特征个数:%d" % x.shape)
print("目标属性样本数据量:%d" % y.shape[0])

# 2. 数据的划分
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=28)
print("训练数据特征属性形状:{}, 测试数据特征形状:{}".format(x_train.shape, x_test.shape))

# 3. 构建管道流的基础算法对象
models = [
    Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', Ridge())
    ]),
    Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', Lasso())
    ])
]

# 4. 构建可选的参数列表
alphas = np.logspace(-3, 1, 20)
parameters = {
    'poly__degree': [2, 3, 4],
    'poly__interaction_only': [True, False],
    'poly__include_bias': [True, False],
    'linear__alpha': alphas
}

# 5. 使用网格交叉验证进行参数选择
algo_names = ['Ridge', 'Lasso']
for t in range(2):
    model = models[t]
    algo = GridSearchCV(model, param_grid=parameters, cv=3)
    # 模型训练
    algo.fit(x_train, y_train)
    # 模型效果输出
    print("{}算法的最优参数为:{}".format(algo_names[t], algo.best_params_))
    print("{}算法的内置的评估值为:{}".format(algo_names[t], algo.best_score_))
    # 使用模型得到预测值
    y_pred = algo.predict(x_test)
    print("{}算法在测试集上的MSE评估值为:{}".format(algo_names[t], mean_squared_error(y_test, y_pred)))
    print("{}算法在测试集上的R2评估值为:{}".format(algo_names[t], r2_score(y_test, y_pred)))
