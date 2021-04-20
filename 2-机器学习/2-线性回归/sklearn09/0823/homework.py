# -- encoding:utf-8 --
"""
实现梯度下降
Create by ibf on 2018/8/23
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 随机数据的产生
np.random.seed(1)
n = 10000
# 这里产生均值为-1，标准差为10的服从正太分布的随机数据
b_values = np.random.normal(loc=-1.0, scale=1.0, size=n)
c_values = np.random.normal(loc=0.0, scale=1.0, size=n)


# 随机数可视化查看一下
# plt.figure(facecolor='w')
# plt.subplot(1, 2, 1)
# plt.hist(b_values, 1000, color='#FF0000')
# plt.subplot(1, 2, 2)
# plt.hist(c_values, 1000, color='#00FF00')
# plt.suptitle(u'随机数据可视化')
# plt.show()

def calc_min_value_with_one_sample(b, c, max_iter=1000, tol=0.00001, alpha=0.01):
    """
    计算当样本数目只有一组的时候，对应的最小的y以及x的值
    :param b:
    :param c:
    :param max_iter:
    :param tol:
    :param alpha:
    :return:
    """

    def f(x, b, c):
        # 原始函数
        return x ** 2 + b * x + c

    def h(x, b, c):
        # 原始函数对应的导函数
        return 2 * x + b

    step_channge = 1.0 + tol
    step = 0
    # 随机的给定一个初始的x值
    current_x = np.random.randint(low=-10, high=10)
    current_y = f(current_x, b, c)
    print("当前的样本数据为:")
    print("b={}".format(b))
    print("c={}".format(c))

    # 开始迭代
    while step_channge > tol and step < max_iter:
        # 1. 计算函数的梯度值
        current_df = h(current_x, b, c)
        # 2. 基于梯度下降更新模型参数(更新x)
        current_x = current_x - alpha * current_df
        # 3. 更新x所对应的y值
        tmp_y = f(current_x, b, c)
        # 4. 记录一下变化大小
        step_channge = np.abs(current_y - tmp_y)
        step += 1
        current_y = tmp_y
    print("最终更新的次数为:{}, 最终的变化率为:{}".format(step, step_channge))
    print("最终的结果为:{}---->{}".format(current_x, current_y))


def calc_min_value_with_two_sample(b1, b2, c1, c2, max_iter=1000, tol=0.00001, alpha=0.01):
    """
    计算当样本数目只有两组的时候，对应的最小的y以及x的值
    :param b1:
    :param b2:
    :param c1:
    :param c2:
    :param max_iter:
    :param tol:
    :param alpha:
    :return:
    """

    def f(x, b1, b2, c1, c2):
        # 原始函数
        y1 = x ** 2 + b1 * x + c1
        y2 = x ** 2 + b2 * x + c2
        return y1 + y2

    def h(x, b1, b2, c1, c2):
        # 原始函数对应的导函数
        y1 = 2 * x + b1
        y2 = 2 * x + b2
        return y1 + y2

    step_channge = 1.0 + tol
    step = 0
    # 随机的给定一个初始的x值
    current_x = np.random.randint(low=-10, high=10)
    current_y = f(current_x, b1, b2, c1, c2)
    print("当前的样本数据为:")
    print("b_values={}, b的均值为:{}".format([b1, b2], np.mean([b1, b2])))
    print("c_values={}, c的均值为:{}".format([c1, c2], np.mean([c1, c2])))

    # 开始迭代
    while step_channge > tol and step < max_iter:
        # 1. 计算函数的梯度值
        current_df = h(current_x, b1, b2, c1, c2)
        # 2. 基于梯度下降更新模型参数(更新x)
        current_x = current_x - alpha * current_df
        # 3. 更新x所对应的y值
        tmp_y = f(current_x, b1, b2, c1, c2)
        # 4. 记录一下变化大小
        step_channge = np.abs(current_y - tmp_y)
        step += 1
        current_y = tmp_y
    print("最终更新的次数为:{}, 最终的变化率为:{}".format(step, step_channge))
    print("最终的结果为:{}---->{}".format(current_x, current_y))


def calc_min_value_with_ten_sample(b_values, c_values, max_iter=1000, tol=0.00001, alpha=0.01):
    """
    计算当样本数目只有十组的时候，对应的最小的y以及x的值
    :param b_values:
    :param c_values:
    :param max_iter:
    :param tol:
    :param alpha:
    :return:
    """
    # 检查一下数据是否是10组数据, assert的功能就是检查代码的运行，如果assert后的表达式执行为False，那么代码报错；如果执行为True，那么直接运行接下来的代码
    assert len(b_values) == 10 and len(c_values) == 10

    def f(x, b_values, c_values):
        # 原始函数
        y1 = x ** 2 + b_values[0] * x + c_values[0]
        y2 = x ** 2 + b_values[1] * x + c_values[1]
        y3 = x ** 2 + b_values[2] * x + c_values[2]
        y4 = x ** 2 + b_values[3] * x + c_values[3]
        y5 = x ** 2 + b_values[4] * x + c_values[4]
        y6 = x ** 2 + b_values[5] * x + c_values[5]
        y7 = x ** 2 + b_values[6] * x + c_values[6]
        y8 = x ** 2 + b_values[7] * x + c_values[7]
        y9 = x ** 2 + b_values[8] * x + c_values[8]
        y10 = x ** 2 + b_values[9] * x + c_values[9]
        return y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10

    def h(x, b_values, c_values):
        # 原始函数对应的导函数
        y1 = 2 * x + b_values[0]
        y2 = 2 * x + b_values[1]
        y3 = 2 * x + b_values[2]
        y4 = 2 * x + b_values[3]
        y5 = 2 * x + b_values[4]
        y6 = 2 * x + b_values[5]
        y7 = 2 * x + b_values[6]
        y8 = 2 * x + b_values[7]
        y9 = 2 * x + b_values[8]
        y10 = 2 * x + b_values[9]
        return y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10

    step_channge = 1.0 + tol
    step = 0
    # 随机的给定一个初始的x值
    current_x = np.random.randint(low=-10, high=10)
    current_y = f(current_x, b_values, c_values)
    print("当前的样本数据为:")
    print("b_values={}, b的均值为:{}".format(b_values, np.mean(b_values)))
    print("c_values={}, c的均值为:{}".format(c_values, np.mean(c_values)))

    # 开始迭代
    while step_channge > tol and step < max_iter:
        # 1. 计算函数的梯度值
        current_df = h(current_x, b_values, c_values)
        # 2. 基于梯度下降更新模型参数(更新x)
        current_x = current_x - alpha * current_df
        # 3. 更新x所对应的y值
        tmp_y = f(current_x, b_values, c_values)
        # 4. 记录一下变化大小
        step_channge = np.abs(current_y - tmp_y)
        step += 1
        current_y = tmp_y
    print("最终更新的次数为:{}, 最终的变化率为:{}".format(step, step_channge))
    print("最终的结果为:{}---->{}".format(current_x, current_y))


def calc_min_value_with_n_sample(n, b_values, c_values, max_iter=1000, tol=0.0000001, alpha=0.01):
    """
    计算当样本数目只有十组的时候，对应的最小的y以及x的值
    :param b_values:
    :param c_values:
    :param max_iter:
    :param tol:
    :param alpha:
    :return:
    """
    # 检查一下数据是否是n组数据, assert的功能就是检查代码的运行，如果assert后的表达式执行为False，那么代码报错；如果执行为True，那么直接运行接下来的代码
    assert len(b_values) == n and len(c_values) == n

    def f(x, n, b_values, c_values):
        # 原始函数
        result = 0
        for i in range(n):
            # 当n足够大的时候，直接累加的这种方式可能会出现result溢出的情况，这里就为了解决这个文件，这里将累加更换为均值的操作
            y = x ** 2 + b_values[i] * x + c_values[i]
            result += y / n
        return result

    def h(x, n, b_values, c_values):
        # 原始函数对应的导函数
        result = 0
        for i in range(n):
            # 当n足够大的时候，直接累加的这种方式可能会出现result溢出的情况，这里就为了解决这个文件，这里将累加更换为均值的操作
            y = 2 * x + b_values[i]
            result += y / n
        return result

    step_channge = 1.0 + tol
    step = 0
    # 随机的给定一个初始的x值
    current_x = np.random.randint(low=-10, high=10)
    current_y = f(current_x, n, b_values, c_values)
    print("当前的样本数据为:")
    print("b的均值为:{}".format(np.mean(b_values)))
    print("c的均值为:{}".format(np.mean(c_values)))

    # 开始迭代
    while step_channge > tol and step < max_iter:
        # 1. 计算函数的梯度值
        current_df = h(current_x, n, b_values, c_values)
        # 2. 基于梯度下降更新模型参数(更新x)
        current_x = current_x - alpha * current_df
        # 3. 更新x所对应的y值
        tmp_y = f(current_x, n, b_values, c_values)
        # 4. 记录一下变化大小
        step_channge = np.abs(current_y - tmp_y)
        step += 1
        current_y = tmp_y
    print("最终更新的次数为:{}, 最终的变化率为:{}".format(step, step_channge))
    print("最终的结果为:{}---->{}".format(current_x, current_y))


# TODO:这里实现SGD的求解方式，并且比较一下BGD和SGD更新过程中的模型参数的变化趋势(x的变化情况)以及在所有样本数据集上函数y和的变化情况  ---> 周二讲

# 计算一个样本的情况
# calc_min_value_with_one_sample(b_values[0], c_values[0])
# 计算两个样本的情况
# calc_min_value_with_two_sample(b_values[0], b_values[1], c_values[0], c_values[1])
# 计算十个样本的情况
# calc_min_value_with_ten_sample(b_values, c_values)
# 计算100个样本的情况
calc_min_value_with_n_sample(n, b_values, c_values)
