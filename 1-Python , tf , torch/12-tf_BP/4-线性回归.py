# -- encoding:utf-8 --
"""
Create by ibf on 2018/5/6
"""

import numpy as np
import tensorflow as tf

# TODO: 大家自己画一下图
'''
    1、构造一个数据
'''
np.random.seed(28)   # 定义随机数种子
N = 100
x = np.linspace(0, 6, N) + np.random.normal(loc=0.0, scale=2, size=N)   # loc：均值；sclae：标准差；size：数量
y = 14 * x - 7 + np.random.normal(loc=0.0, scale=5.0, size=N)

x.shape = -1, 1  # 转为一维列向量
y.shape = -1, 1

'''
    2、线性模型构建，定义一个变量w和变量b  y=wx+b
    -》tf.random_uniform：产生一个服从均匀分布(uniform)的随机数列
        shape: 数列形状
        minval：均匀分布中可能出现的最小值
        maxval: 均匀分布中可能出现的最大值
'''
w = tf.Variable(initial_value=tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0), name='w')
b = tf.Variable(initial_value=tf.zeros([1]), name='b')
# 构建当前BP过程产生的预测值
y_predict = w * x + b   # 每次一都用了100个样本

'''
    3、构建损失函数MSE
    -》tf.reduce_mean(*)：求均值
    
    -》tf.train.GradientDescentOptimizer(*)：以随机梯度下降（SGD）的方式优化损失函数
    -》*.minimize(loss,name)：在优化的过程中，让loss函数最小化
'''
loss = tf.reduce_mean(tf.square(y_predict - y), name='loss')
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss, name='train')   # 对损失函数进行梯度下降处理，得到最小的损失函数

# train也可以拆开写成下面类型
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
# train = optimizer.minimize(loss, name='train')


'''
    4、运行
'''
def print_info(r_w, r_b, r_loss):
    print("w={},b={},loss={}".format(r_w, r_b, r_loss))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     # 初始化

    r_w, r_b, r_loss = sess.run([w, b, loss])
    print_info(r_w, r_b, r_loss)    # 输出初始化的w、b、loss

    # 进行训练(n次)
    for step in range(3):
        # 模型训练
        sess.run(train)
        # 输出训练后的w、b、loss
        r_w, r_b, r_loss = sess.run([w, b, loss])
        print_info(r_w, r_b, r_loss)
