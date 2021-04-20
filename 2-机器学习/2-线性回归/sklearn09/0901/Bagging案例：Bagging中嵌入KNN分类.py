# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/19
"""

import pandas as pd
import numpy as np
import matplotlib as mpl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.ensemble import BaggingClassifier

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 一、加载数据
names = ['A', 'B', 'C', 'D', 'label']
path = '../datas/iris.data'
df = pd.read_csv(path, sep=',', header=None, names=names)
# df.info()
# print(df.head(2))

# 二、数据的清洗
df.replace('?', np.nan, inplace=True)
df.dropna(axis=0, how='any', inplace=True)

# 三、基于业务提取最原始的特征属性X和目标属性Y
X = df[names[:-1]]
Y = df[names[-1]]
Y_label_values = np.unique(Y)
Y_label_values.sort()
print("Y的取值可能:{}".format(Y_label_values))
random_index = np.random.permutation(len(Y))[:5]
print("随机的索引:{}".format(random_index))
print("原始随机的5个Y值:{}".format(np.array(Y[random_index])))
for i in range(len(Y_label_values)):
    Y[Y == Y_label_values[i]] = i
Y = Y.astype(np.float)
print("做完标签值转换后的随机的5个Y值:{}".format(np.array(Y[random_index])))

# 四、数据的划分(将数据划分为训练集和测试集)
print("原始X形状:{}， 原始Y形状:{}".format(X.shape, Y.shape))
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)
print("训练集样本形状:{}, 测试集样本形状:{}".format(x_train.shape, x_test.shape))

# 六、算法模型的选择/算法模型对象的构建
knn = KNeighborsClassifier(n_neighbors=10)

# 构建Bagging的集成分类算法对象
algo = BaggingClassifier(base_estimator=knn, n_estimators=10)

# 七、算法模型的训练
algo.fit(x_train, y_train)

# 八、模型效果评估
y_hat = algo.predict(x_test)
print("在训练集上的模型效果(分类算法中为准确率):{}".format(algo.score(x_train, y_train)))
print("在测试集上的模型效果(分类算法中为准确率):{}".format(algo.score(x_test, y_test)))
print("在测试集上的召回率的值:{}".format(recall_score(y_true=y_test, y_pred=y_hat, average='micro')))

# 九、输出分类中特有的一些API
print("=" * 100)
y_predict = algo.predict(x_test)
print("预测值:\n{}".format(y_predict))
print("预测的实际类别:\n{}".format([np.array(Y_label_values)[int(i)] for i in y_predict]))
print("=" * 100)
print("属于各个类别的概率值:\n{}".format(algo.predict_proba(x_test)))
print("=" * 100)
