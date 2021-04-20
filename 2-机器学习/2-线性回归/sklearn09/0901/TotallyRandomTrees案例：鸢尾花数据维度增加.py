# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/19
"""

import pandas as pd
import numpy as np
import matplotlib as mpl

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomTreesEmbedding

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 一、加载数据
names = ['A', 'B', 'C', 'D', 'label']
path = '../datas/iris.data'
df = pd.read_csv(path, sep=',', header=None, names=names)

# 二、数据的清洗
df.replace('?', np.nan, inplace=True)
df.dropna(axis=0, how='any', inplace=True)

# 三、基于业务提取最原始的特征属性X和目标属性Y
X = df[names[:2]]
Y = df[names[-1]]
Y_label_values = np.unique(Y)
Y_label_values.sort()
random_index = np.random.permutation(len(Y))[:5]
for i in range(len(Y_label_values)):
    Y[Y == Y_label_values[i]] = i
Y = Y.astype(np.float)

# 四、数据的划分(将数据划分为训练集和测试集)
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)

# 六、算法模型的选择/算法模型对象的构建
algo = RandomTreesEmbedding(n_estimators=3, max_depth=5)

# 七、算法模型的训练
algo.fit(x_train, y_train)

# 获取所有的子模型（中间的所有决策树）
from sklearn import tree
import pydotplus

estimators_ = algo.estimators_
for i, treemodel in enumerate(estimators_):
    dot_data = tree.export_graphviz(decision_tree=treemodel, out_file=None,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('trt_tree_{}.png'.format(i))

# 数据维度扩展
x_train2 = algo.transform(x_train)
x_test2 = algo.transform(x_test)
print("扩展前训练数据大小:{}，扩展后数据大小:{}".format(x_train.shape, x_train2.shape))
print("扩展前测试数据大小:{}，扩展后数据大小:{}".format(x_test.shape, x_test2.shape))
