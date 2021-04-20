# -- encoding:utf-8 --
"""
Create by ibf on 2018/9/1
"""

import numpy as np


def entropy(t):
    """
    计算信息熵
    :param t:  是一个概率组成的集合
    :return:
    """
    return np.sum([-p * np.log2(p) for p in t if p != 0])


def h1(y=[1, 1, 1, -1, -1, -1, 1, 1, 1, -1], w=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    """
    计算一下数据的信息熵
    :return:
    """
    # 1. 计算类别为1的概率和
    p1 = np.sum(np.array(w)[np.array(y) == 1])
    # 2. 计算类别为-1的概率和
    p2 = 1 - p1
    # 计算信息熵
    return entropy([p1, p2])


def h2(split=3, y=[1, 1, 1, -1, -1, -1, 1, 1, 1, -1], w=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    y = np.array(y)
    w = np.array(w)

    # 1. 左侧的信息熵
    data_y = y[:split]
    data_w = w[:split]
    p10 = np.sum(data_w)
    p11 = np.sum(data_w[data_y == 1]) / p10
    p12 = 1 - p11
    p1 = entropy([p11, p12])

    # 2. 右侧的信息熵
    data_y = y[split:]
    data_w = w[split:]
    p20 = np.sum(data_w)
    p21 = np.sum(data_w[data_y == 1]) / p20
    p22 = 1 - p21
    p2 = entropy([p21, p22])

    # 3. 计算信息熵
    return p10 * p1 + p20 * p2


def h3(errs=[], w=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    # w更新
    w = np.array(w)
    # 1. 计算错误率
    e = np.sum([w[i] for i in errs])
    # 2. 计算alpha
    alpha = 0.5 * np.log2((1 - e) / e)
    # 3. 计算更新后的权重值
    accs = [i for i in range(len(w)) if i not in errs]
    w[accs] = w[accs] * (np.e ** (-alpha))
    w[errs] = w[errs] * (np.e ** alpha)
    w = w / np.sum(w)

    return e, alpha, w


def h4(errs=[], w=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    # w更新
    w = np.array(w)
    # 1. 计算错误率
    e = np.sum([w[i] for i in errs])
    # 2. 计算alpha
    alpha = 0.5 * np.log((1 - e) / e)
    # 3. 计算更新后的权重值
    accs = [i for i in range(len(w)) if i not in errs]
    w[accs] = w[accs] * (np.e ** (-alpha))
    w[errs] = w[errs] * (np.e ** alpha)
    w = w / np.sum(w)

    return e, alpha, w


def calc1():
    print("第一个子模型的选择")
    print("=" * 100)
    a = h1()
    print("原始数据的信息熵:{}".format(a))
    print("以2.5分割的信息增益:{}".format(a - h2(split=3)))
    print("以5.5分割的信息增益:{}".format(a - h2(split=6)))
    print("以8.5分割的信息增益:{}".format(a - h2(split=9)))
    e, a1, w = h3(errs=[6, 7, 8])
    print("第1次迭代后的错误率:{}".format(e))
    print("第1次的模型权重为:{}".format(a1))
    print("第1次更新后的样本权重为:{}".format(w))

    print("\n第二个子模型的选择")
    print("=" * 100)
    a = h1(w=w)
    print("总的信息熵:{}".format(a))
    print("以2.5分割的信息增益:{}".format(a - h2(split=3, w=w)))
    print("以5.5分割的信息增益:{}".format(a - h2(split=6, w=w)))
    print("以8.5分割的信息增益:{}".format(a - h2(split=9, w=w)))
    e, a2, w = h3(errs=[0, 1, 2, 9], w=w)
    print("第2次迭代后的错误率:{}".format(e))
    print("第2次的模型权重为:{}".format(a2))
    print("第2次更新后的样本权重为:{}".format(w))

    print("\n第三个子模型的选择")
    print("=" * 100)
    a = h1(w=w)
    print("总的信息熵:{}".format(a))
    print("以2.5分割的信息增益:{}".format(a - h2(split=3, w=w)))
    print("以5.5分割的信息增益:{}".format(a - h2(split=6, w=w)))
    print("以8.5分割的信息增益:{}".format(a - h2(split=9, w=w)))
    e, a3, w = h3(errs=[3, 4, 5], w=w)
    print("第3次迭代后的错误率:{}".format(e))
    print("第3次的模型权重为:{}".format(a3))
    print("第3次更新后的样本权重为:{}".format(w))

    print("\n第四个子模型的选择")
    print("=" * 100)
    a = h1(w=w)
    print("总的信息熵:{}".format(a))
    print("以2.5分割的信息增益:{}".format(a - h2(split=3, w=w)))
    print("以5.5分割的信息增益:{}".format(a - h2(split=6, w=w)))
    print("以8.5分割的信息增益:{}".format(a - h2(split=9, w=w)))
    e, a4, w = h3(errs=[6, 7, 8], w=w)
    print("第4次迭代后的错误率:{}".format(e))
    print("第4次的模型权重为:{}".format(a4))
    print("第4次更新后的样本权重为:{}".format(w))

    print("=" * 100)
    print(a1)
    print(a2)
    print(a3)
    print(a4)

    print("=" * 100)
    print("1:{}".format(a1 - a2 + a3 + a4))
    print("2:{}".format(-a1 - a2 + a3 - a4))
    print("3:{}".format(-a1 + a2 + a3 - a4))
    print("4:{}".format(-a1 + a2 - a3 - a4))


def calc2():
    print("第一个子模型的选择")
    print("=" * 100)
    a = h1()
    print("原始数据的信息熵:{}".format(a))
    print("以2.5分割的信息增益:{}".format(a - h2(split=3)))
    print("以5.5分割的信息增益:{}".format(a - h2(split=6)))
    print("以8.5分割的信息增益:{}".format(a - h2(split=9)))
    print("第一个子模型选择以2.5划分数据:")
    e, a1, w = h4(errs=[6, 7, 8])
    print("第1次迭代后的错误率:{}".format(e))
    print("第1次的模型权重为:{}".format(a1))
    print("第1次更新后的样本权重为:{}".format(w))

    print("\n第二个子模型的选择")
    print("=" * 100)
    a = h1(w=w)
    print("总的信息熵:{}".format(a))
    print("以2.5分割的信息增益:{}".format(a - h2(split=3, w=w)))
    print("以5.5分割的信息增益:{}".format(a - h2(split=6, w=w)))
    print("以8.5分割的信息增益:{}".format(a - h2(split=9, w=w)))
    print("第二个子模型选择以8.5划分数据:")
    e, a2, w = h4(errs=[3, 4, 5], w=w)
    print("第2次迭代后的错误率:{}".format(e))
    print("第2次的模型权重为:{}".format(a2))
    print("第2次更新后的样本权重为:{}".format(w))

    print("\n第三个子模型的选择")
    print("=" * 100)
    a = h1(w=w)
    print("总的信息熵:{}".format(a))
    print("以2.5分割的信息增益:{}".format(a - h2(split=3, w=w)))
    print("以5.5分割的信息增益:{}".format(a - h2(split=6, w=w)))
    print("以8.5分割的信息增益:{}".format(a - h2(split=9, w=w)))
    print("第三个子模型选择以5.5划分数据:")
    e, a3, w = h4(errs=[0, 1, 2, 9], w=w)
    print("第3次迭代后的错误率:{}".format(e))
    print("第3次的模型权重为:{}".format(a3))
    print("第3次更新后的样本权重为:{}".format(w))

    print("=" * 100)
    print(a1)
    print(a2)
    print(a3)

    print("=" * 100)
    print("1:{}".format(a1 + a2 - a3))
    print("2:{}".format(-a1 + a2 - a3))
    print("3:{}".format(-a1 + a2 + a3))
    print("4:{}".format(-a1 - a2 + a3))


# print("以2为底数的计算方式:")
# calc1()

print("以e为底数的计算方式:")
calc2()
