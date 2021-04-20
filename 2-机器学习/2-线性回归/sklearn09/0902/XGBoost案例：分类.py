# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/19
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score

warnings.filterwarnings('ignore')

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
X = df[names[:-1]]
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
algo = GradientBoostingClassifier(n_estimators=100)

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
