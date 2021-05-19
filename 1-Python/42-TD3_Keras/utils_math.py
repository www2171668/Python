""""""
import tensorflow as tf
import numpy as np
from numpy import matlib as mb

import keras.backend as K

# %% 正态分布噪音
def add_normal(x_input, outshape, at_eps):
    epsilon = K.random_normal(shape=outshape, mean=0., stddev=1.)  # 正态分布
    x_out = x_input + at_eps * tf.multiply(epsilon, tf.abs(x_input))
    return x_out

# %% 熵与KL散度
def entropy(p):  # * 熵的计算方法：∑p*logp  加epsilon
    out = - K.sum(p * K.log(p + 1e-8))    # 没有axis时，K.sum将全部元素相加
    return out

def kl(p, q):  # * KL散度 DKL(p||q)
    KL_divergency = K.sum(p * K.log((p + 1e-8) / (q + 1e-8)))
    return KL_divergency

# %% 加权
def weighted_entropy(p, w_norm):
    out = - K.sum(w_norm * p * K.log(p + 1e-8))
    return out

def weighted_mean_entropy(p, w_norm):
    out = - K.mean(w_norm * p * K.log(p + 1e-8))
    return out

def weighted_mse(Q_target, Q_pred, w_norm):
    error = K.mean(w_norm * K.square(Q_target - Q_pred))
    return error

def weighted_mean(p, w_norm):
    p_weight = K.mean(tf.multiply(w_norm, p), axis=0)  # 求每一列的均值
    return p_weight

# %% 概率
def softmax(x):
    # * 求得 e(maxQ - 原Q表)
    _, col = x.shape
    x_max = np.reshape(np.max(x, axis=1), (-1, 1))  # 找到最大的 目标Q 值，并扩充维度
    e_x = np.exp(x - mb.repmat(x_max, 1, col))  # np.matlib.repmat(a,m,n) 对传入的矩阵进行扩展（复制），m是行的倍数，n是列的倍数
    # * 求softmax     ——   e(maxQ - 原Q表) / ∑e(maxQ - 原Q表)
    e_x_sum = np.reshape(np.sum(e_x, axis=1), (-1, 1))  # e(差值)累加
    e_x_sum = mb.repmat(e_x_sum, 1, col)  # 复制列
    out = e_x / e_x_sum
    return out

def p_sample(p):
    row, col = p.shape
    p_cumsum = np.cumsum(p, axis=1)  # np.cumsum() 累加。=1 按列累加.  最后一列全为1
    rand = mb.repmat(np.random.random((row, 1)), 1, col)  # 随机数，并复制列
    o_softmax = np.argmax(p_cumsum >= rand, axis=1)  # 与随机数进行对比，找到为True的编号  (0,1,1,0,...一类)
    return o_softmax

def weighted_mean_array(x, weights):
    weights_mean = np.mean(weights, axis=1)  # n=2时为0.5，n=4时为0.25
    x_weighted_mean = np.mean(np.multiply(x, weights), axis=1)
    mean_weighted = np.reshape(np.divide(x_weighted_mean, weights_mean), (-1, 1))  # 归一化，并扩充列
    return mean_weighted
