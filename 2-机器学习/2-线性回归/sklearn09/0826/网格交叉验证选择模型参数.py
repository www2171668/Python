# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/19
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

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
for i in range(len(Y_label_values)):
    Y[Y == Y_label_values[i]] = i
Y = Y.astype(np.float)

# 四、数据的划分(将数据划分为训练集和测试集)
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=28)

# 五、特征工程
ss = StandardScaler()
x_train = ss.fit_transform(x_train, y_train)
x_test = ss.transform(x_test)

# 六、算法模型的选择/算法模型对象的构建
knn = KNeighborsClassifier(algorithm='kd_tree')

# 七、使用网格交叉验证的方式来进行模型参数选择(选择在训练数据集上效果最好的参数)
"""
def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score=True):
        estimator: 给定需要进行参数选择的sklearn的模型对象(特征工程、算法模型、管道流....)
        param_grid：给定的一个estimator参数选择的一个字典，key为estimator中的模型参数字符串，vlaue为该参数的取值列表
        cv：给定做几折交叉验证
        scoring：给定在做模型参数选择的时候，衡量模型效果的指标，默认为estimator自带的score算法；可选值参考：http://scikit-learn.org/0.18/modules/model_evaluation.html#model-evaluation
"""
param_grid = {
    'n_neighbors': [3, 5, 10, 20],
    'weights': ['uniform', 'distance'],
    'leaf_size': [10, 30, 50]
}
# algo = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='accuracy', cv=5)
algo = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)
algo.fit(x_train, y_train)

# 八、输出最好的参数、最好的模型
best_param = algo.best_params_
print("最好的参数列表:{}".format(best_param))
best_knn = algo.best_estimator_
print("最优模型:{}".format(best_knn))
print("使用最优模型来预测:{}".format(best_knn.predict(ss.transform([[4.6, 3.4, 1.4, 0.3]]))))
print("使用GridSearchCV预测:{}".format(algo.predict(ss.transform([[4.6, 3.4, 1.4, 0.3]]))))

# 九、模型效果评估
y_hat = algo.predict(x_test)
print("在训练集上的模型效果(分类算法中为准确率):{}".format(algo.score(x_train, y_train)))
print("在测试集上的模型效果(分类算法中为准确率):{}".format(algo.score(x_test, y_test)))
print("预测值:\n{}".format(y_hat))
print("预测的实际类别:\n{}".format([np.array(Y_label_values)[int(i)] for i in y_hat]))

# 十、模型保存
# 保存y标签和name之间的映射关系
y_index_label_dict = dict(zip(range(len(Y_label_values)), Y_label_values))
pickle.dump(obj=y_index_label_dict, file=open('./model/y_label_value.pkl', 'wb'))
joblib.dump(ss, './model/ss.pkl')
# 方式一：直接保存GridSearchCV对象
joblib.dump(algo, './model/gcv.pkl')
# 方式二：保存GridSearchCV中的最优模型
joblib.dump(algo.best_estimator_, './model/knn.pkl')

print("Done!!!")
