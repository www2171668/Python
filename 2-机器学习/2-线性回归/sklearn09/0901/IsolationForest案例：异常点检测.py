# -- encoding:utf-8 --
"""
Create by ibf on 2018/9/1
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(28)

# 产生模拟数据
x = 0.3 * np.random.randn(100, 2)
x_train = np.vstack((x + 2, x - 2))
x = 0.3 * np.random.randn(20, 2)
x_test = np.vstack((x + 2, x - 2))
x_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# 构建训练模型
algo = IsolationForest(n_estimators=3, random_state=28)
algo.fit(x_train)

# 获取预测值（范围为1表示正常样本，返回-1表示异常样本）
y_pred_train = algo.predict(x_train)
print(y_pred_train)
y_pred_test = algo.predict(x_test)
print(y_pred_test)
y_pred_outliers = algo.predict(x_outliers)
print(y_pred_outliers)

# 返回决策函数，模型情况下，在sklearn中，当决策函数值大于0的时候，表示属于正常样本，否则属于异常样本
print(algo.decision_function(x_outliers))
print(algo.decision_function(x_test))

# 获取所有的子模型
from sklearn import tree
import pydotplus

estimators_ = algo.estimators_
for i, treemodel in enumerate(estimators_):
    dot_data = tree.export_graphviz(decision_tree=treemodel, out_file=None,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('isolation_{}.png'.format(i))
    np.hstack()

# 画图
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
z = algo.decision_function(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

# contourf画等高线的区域图
plt.contourf(xx, yy, z, cmap=plt.cm.Blues_r)
plt.scatter(x_train[:, 0], x_train[:, 1], c='b')
plt.scatter(x_test[:, 0], x_test[:, 1], c='g')
plt.scatter(x_outliers[:, 0], x_outliers[:, 1], c='r')
plt.show()
