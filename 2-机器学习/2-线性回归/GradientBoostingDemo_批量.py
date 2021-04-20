# -- encoding:utf-8 --
"""
实现基于批量梯度下降的线性回归算法
Create by ibf on 2018/9/6
"""

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


def predict(x, theta, intercept=0.0):   # 对于样本x产生预测值
    """
    :param x: 单个样本x
    """
    result = 0.0
    # 1. x和theta相乘
    n = len(x)   # 特征数量
    for i in range(n):
        result += x[i] * theta[i]
    # 2. 加上截距项
    result += intercept
    return result


def predict_X(X, theta, intercept=0.0):
    # 返回所有数据的预测值集合
    Y = []
    for x in X:
        Y.append(predict(x, theta, intercept))
    return Y


def fit(X, Y, alpha=0.0001, max_iter=None, tol=1e-5, fit_intercept=True):
    """
    使用批量的梯度下降实现线性回归的参数求解，最终返回参数theta以及截距项intercept
    :param X:  输入的特征矩阵，格式是二维的矩阵形式, m*n, m表示样本数目，n表示每个样本的特征属性数目
    :param Y:  输入的目标属性矩阵，格式是二维数组形式, m*1, m表示样本数目，1表示预测属性y
    :param alpha:  学习率
    :param max_iter:  最大迭代次数
    :param tol:  收敛值
    :param fit_intercept: 是否训练截距项
    :return:
    """
    # 1. 对输入数据做一个整理
    X = np.array(X)     # 转换为数组类型
    Y = np.array(Y)
    if max_iter is None:
        max_iter = 2000
    max_iter = max_iter if max_iter > 0 else 2000

    # 2. 获取样本的数目和维度信息
    m, n = np.shape(X)   # m为X的行数，b为X的列数

    # 3. 定义相关的变量
    theta = np.zeros(n)
    intercept = 0
    # 定义一个存储所有样本的残差的数组，即y-h(x)的值
    diff = np.zeros(m)   # m 个样本空间
    # 定义上一个迭代的损失函数的值
    previous_j = 1 << 64   # 1 左移64位

    # 4. 开始模型迭代
    for i in range(max_iter):
        # a. 计算所有样本的预测值和实际值之间的差值
        for k in range(m):
            y_true = Y[k]   # 真实值 ；遍历Y一维list
            y_predict = predict(X[k], theta, intercept)   # 预测值 ；X[k] 第k个样本
            diff[k] = y_true - y_predict

        # b. 基于所有样本的预测值和实际值之间的差值更新每一个theta值
        for j in range(n):   # 共有n个样本值，也就有n个theta
            # --1. 累加所有样本的梯度值（gd）
            gd = 0
            for k in range(m):   # 按照批量梯度公式，一个一个样本
                gd += diff[k] * X[k][j]   # X[k][j] X的第K个样本的第J个特征值
            # --2. 基于梯度值更新模型参数
            theta[j] += alpha * gd   # 更新theta值

        # c. 基于所有样本的预测值和实际值之间的差值更新截距项
        if fit_intercept:
            # 截距项就是theta0，公式中的x0 = 1
            intercept += alpha * np.sum(diff)

        # d. 计算在当前模型参数的情况下，损失函数（sum_j）的值。损失函数对应回归算法的最小二乘法
        sum_j = 0.0
        for k in range(m):
            y_true = Y[k]
            y_predict = predict(X[k], theta, intercept)
            # OverflowError: math range error, 该异常的主要原因是由于参数更新的时候梯度过大，导致模型不收敛
            sum_j += math.pow(y_true - y_predict, 2)   # 目标函数θ标准求解公式
        sum_j /= m   # 也可以不除m

        # e. 比较上一次的损失函数值和当前损失函数值之间的差值是否小于参数tol，如果小于表示结束循环
        if np.abs(previous_j - sum_j) < tol:
            break

        previous_j = sum_j   # 更新损失函数值
    print("迭代{}次后，损失函数值为:{}".format(i, previous_j))   # —— .format 和 % 功能是类似的，用format时替代{}
    return theta, intercept, previous_j


if __name__ == '__main__':
    # 1. 构建模拟数据
    np.random.seed(28)
    N = 10
    x = np.linspace(start=0, stop=6, num=N) + np.random.randn(N)
    y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)
    x.shape = -1, 1
    y.shape = -1, 1

    # 2. 使用算法构建模型
    # a. 使用sklearn中自带的线性回归算法
    model = LinearRegression()
    model.fit(x, y)
    print("sklearn内置线性回归模型============")
    s1 = model.score(x, y)
    print("训练数据R^2:{}".format(s1))
    print("模型的参数值:{}".format(model.coef_))
    print("模型的截距项:{}".format(model.intercept_))

    # b. 使用自定义的梯度下降实现代码
    theta, intercept, sum_j = fit(x, y, alpha=0.0001, max_iter=2000)
    print("自定的基于梯度下降的线性回归模型============")
    s2 = r2_score(y, predict_X(x, theta, intercept))   # —— r2_score(y_true,y_pred) 和线性回归的socre(x,y)功能一样
    print("训练数据R^2:{}".format(s2))   # 导入y实际值，和预测值··
    print("模型的参数值:{}".format(theta))
    print("模型的截距项:{}".format(intercept))

    # 构建画图用的模拟数据
    x_hat = np.linspace(x.min(), x.max(), num=100)
    x_hat.shape = -1, 1
    y_hat = model.predict(x_hat)
    y_hat2 = predict_X(x_hat, theta, intercept)

    # 画图看一下
    plt.plot(x, y, 'ro', ms=10)
    plt.plot(x_hat, y_hat, color='#b624db', lw=2, label=u'sklearn线性模型，$R^2$:%.3f' % s1)
    plt.plot(x_hat, y_hat2, color='#0049b6', lw=2, label=u'自定义线性模型，$R^2$:%.3f' % s2)
    plt.legend(loc='upper left')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.suptitle(u'sklearn模型和自定义模型的效果比较', fontsize=20)
    plt.grid(True)
    plt.show()
