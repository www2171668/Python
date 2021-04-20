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

# 3. 构建多项式扩展的对象
poly = PolynomialFeatures(degree=3, include_bias=True, interaction_only=True)
x_train = poly.fit_transform(x_train, y_train)
x_test = poly.transform(x_test)
print("使用多项式扩展后的训练数据形状:{}".format(x_train.shape))

# 4. 构建Lasso对象
lasso = Lasso(alpha=3.5938)
lasso.fit(x_train, y_train)

# 5. 提取参数值不为0的特征信息
coef_ = lasso.coef_
x_train = x_train[:, coef_ != 0]
x_test = x_test[:, coef_ != 0]
print("提取重要特征后的训练数据大小:{}".format(x_train.shape))

# 6. 使用提取特征之后的数据做一个Ridge变化
algo = Ridge(alpha=10.0)
algo.fit(x_train, y_train)

# 7. 模型效果评估
train_pred = algo.predict(x_train)
test_pred = algo.predict(x_test)
print("训练数据的MSE评估指标:{}".format(mean_squared_error(y_train, train_pred)))
print("测试数据的MSE评估指标:{}".format(mean_squared_error(y_test, test_pred)))
print("训练数据的R2评估指标:{}".format(r2_score(y_train, train_pred)))
print("测试数据的R2评估指标:{}".format(r2_score(y_test, test_pred)))
