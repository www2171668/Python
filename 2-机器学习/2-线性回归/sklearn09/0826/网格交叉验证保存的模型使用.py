# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/26
"""

import numpy as np
import pickle
from sklearn.externals import joblib

# 1. 加载模型
ss = joblib.load('./model/ss.pkl')
gcv = joblib.load('./model/gcv.pkl')
knn = joblib.load('./model/knn.pkl')
y_label_index_2_label_dict = pickle.load(open('./model/y_label_value.pkl', 'rb'))


def predict1(x):
    return gcv.predict(ss.transform(x))


def predict2(x):
    return knn.predict(ss.transform(x))


if __name__ == '__main__':
    # 假定现在接收到一个输入数据x
    x = [[5.0, 3.4, 1.5, 0.2], [6.6, 3.0, 4.4, 1.4]]
    # 传入对应的API中获取得到预测值
    y_predict1 = predict1(x)
    print(y_predict1)
    y_predict2 = predict2(x).astype(np.int)
    print(y_predict2)
    # 结果返回
    result_label = []
    print(y_label_index_2_label_dict)
    for index in y_predict2:
        label = y_label_index_2_label_dict[int(index)]
        result_label.append(label)
    result = list(zip(y_predict2, result_label))
    print(result)
