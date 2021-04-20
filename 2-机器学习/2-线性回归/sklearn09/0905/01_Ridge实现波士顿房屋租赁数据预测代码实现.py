# -- encoding:utf-8 --
"""
Create by ibf on 2018/9/5
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn.utils import shuffle


def not_empty(s):
    return s != ''


if __name__ == '__main__':
    # flag为True表示模型训练，为False表示加载模型对数据进行预测
    flag = True

    # 1. 读取数据形成DataFrame并且分割出x和y
    file_path = '../datas/boston_housing.data'
    df = pd.read_csv(filepath_or_buffer=file_path, header=None)
    # 提取数据
    data = np.empty(shape=(len(df), 14))
    for i, d in enumerate(df.values):
        # 对df中的每行数据做遍历，d即为每行数据
        d = list(map(float, filter(not_empty, d[0].split(' '))))
        data[i] = d
    print(data.shape)

    if flag:
        # 1. 数据分割
        x, y = np.split(data, (13,), axis=1)
        y = y.ravel()
        print("样本数据量:%d，特征个数:%d" % x.shape)
        print("目标属性样本数据量:%d" % y.shape[0])

        # 2. 数据的划分
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=28)
        print("训练数据特征属性形状:{}, 测试数据特征形状:{}".format(x_train.shape, x_test.shape))

        # 3. 模型的构建
        algo = Ridge(alpha=1.0)

        # 4. 模型训练
        algo.fit(x_train, y_train)

        # 5. 模型效果评估
        train_pred = algo.predict(x_train)
        test_pred = algo.predict(x_test)
        print("训练数据的MSE评估指标:{}".format(mean_squared_error(y_train, train_pred)))
        print("测试数据的MSE评估指标:{}".format(mean_squared_error(y_test, test_pred)))
        print("训练数据的R2评估指标:{}".format(r2_score(y_train, train_pred)))
        print("测试数据的R2评估指标:{}".format(r2_score(y_test, test_pred)))

        # 6. 模型保存
        joblib.dump(algo, './models/01_ridge.m')
    else:
        # 1. 加载模型
        algo = joblib.load('./models/01_ridge.m')
        # 2. 使用模型对输入的数据做一个预测，并将预测值输出
        data = shuffle(data)
        data = data[:10]
        y_pred = algo.predict(data[:, :13])
        print("预测的R2指标为:{}".format(r2_score(data[:, 13], y_pred)))
        print("实际值:\n{}".format(data[:, 13]))
        print("预测值:\n{}".format(y_pred))
