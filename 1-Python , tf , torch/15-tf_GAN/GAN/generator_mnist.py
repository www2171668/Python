# -- encoding:utf-8 --
"""
GAN生成手写数字图片
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import shutil
import numpy as np
from skimage.io import imsave

checkpoint_dir = "output"
is_restore = True   # 默认加载数据继续训练

image_height = 28   # 手写数字识别库的图片高度和宽度就是28
image_weight = 28
image_size = image_height * image_weight  # 图像大小，也是G网络输出层神经元数量

z_size = 100    # G网络输入层神经元数量
h1_size = 150    # G网络第一隐层神经元数量
h2_size = 300    # G网络第二隐层层神经元数量

batch_size = 256
max_epoch = 500   # 最大迭代次数


def build_generator(z_prior):   # 构建生成网络G
    """
    :param z_prior: 输入的噪音数据, 是一个n行z_size列的Tensor对象，n行表示n个样本输入，z_size表示每个输入的样本的维度大小
    """
    # 1. 输入 -> 第二层
    w1 = tf.Variable(tf.truncated_normal([z_size, h1_size], stddev=0.1), name='g_w1', dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h1_size]), name='g_b1', dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(z_prior, w1) + b1)

    # 2. 第二层 -> 第三层
    w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name='g_w2', dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h2_size]), name='g_b2', dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # 3. 第三层 -> 输出
    w3 = tf.Variable(tf.truncated_normal([h2_size, image_size], stddev=0.1), name='g_w3', dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([image_size]), name='g_b3', dtype=tf.float32)
    h3 = tf.nn.tanh(tf.matmul(h2, w3) + b3)   # 使输出为[-1,1]过程

    return h3, [w1, b1, w2, b2, w3, b3]   # 假样本 & 超参


def build_disciminator(x_data, x_generated, keep_prob):   # 构建判别模型D
    """
    :param x_data:  真实数据
    :param x_generated: 假数据
    :param keep_prob: 进行drop out的时候给定的保留率，防止过拟合
    """
    # 1. 合并训练数据（axis=0）
    x_input = tf.concat([x_data, x_generated], 0)

    # 2. 输入 -> 第二层
    w1 = tf.Variable(tf.truncated_normal([image_size, h2_size], stddev=0.1), name='d_w1', dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h2_size]), name='d_b1', dtype=tf.float32)
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_input, w1) + b1), keep_prob)

    # 3. 第二层 -> 第三层
    w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name='d_w2', dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h1_size]), name='d_b2', dtype=tf.float32)
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)

    # 3. 第三层 -> 输出层
    w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name='d_w3', dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name='d_b3', dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3

    # 对h3的结果进行sigmoid转换(logistic回归)
    # -》tf.slice()：对结果进行切片（截断），将真样本的输出D(x) 和 假样本的输出D(G(z)) 分离
    y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1]))   # -1表示全包含
    y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1]))

    return y_data, y_generated, [w1, b1, w2, b2, w3, b3]   # 输出D(x) 和 D(G(z)) & 超参


def save_image(x_generated_val, fname):   # 保存图像
    """
    :param x_generated_val: 生成图像数据
    :param fname: 数据保存地址
    """
    x_generated_val = 0.5 * x_generated_val.reshape((x_generated_val.shape[0], image_height, image_weight)) + 0.5   # 格式重置，数据还原
    # 多组图像的中间位置进行填充，方便进行查看
    grid_h = image_height * 8 + 5 * (8 - 1)   # 5表示两个图像的中间高度差
    grid_w = image_weight * 8 + 5 * (8 - 1)   # 5表示两个图像的中间宽度差
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)   # 实际图像
    for i, res in enumerate(x_generated_val):
        if i >= 8 * 8:   # 将x_generated_val的前64张图像进行8*8的拼接，转换为一张图像进行输出，方便查看
            break
        img = ((res) * 255).astype(np.uint8)
        row = (i // 8) * (image_height + 5)   # i // 8：除法取整（向下取整）
        col = (i % 8) * (image_weight + 5)
        img_grid[row:row + image_height, col:col + image_weight] = img

    # -》imsave()：保存图像
    imsave(fname, img_grid)


def train():   # 模型模型训练
    # 1. 加载数据(真实数据)
    mnist = input_data.read_data_sets('data/', one_hot=True)

    # 2. 定义网络数据占位符
    x_data = tf.placeholder(tf.float32, [batch_size, image_size], name='x_data')
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name='z_prior')   # z和x的shape[0]不一定要保持一致
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    global_step = tf.Variable(0, name='global_step', trainable=False)   # 全局迭代次数

    # 3. 构建生成模型
    x_generated, g_params = build_generator(z_prior)

    # 4. 构建判别模型
    y_data, y_generated, d_params = build_disciminator(x_data, x_generated, keep_prob)

    # 5. 构造D与G的损失函数
    # d_loss = tf.reduce_mean(tf.reduce_sum(-(1 * tf.log(y_data) + (1 - 0)*tf.log(1-y_generated))))   # 简写为下面
    d_loss = -(tf.log(y_data) + tf.log(1 - y_generated))
    g_loss = -tf.log(y_generated)

    # 6. 构造D与G的目标函数
    optimizer = tf.train.AdamOptimizer(0.0001)   # 使用AdamOptimizer优化器
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)   # var_list: 待更新参数列表
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    # 7. 模型训练

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())   # 初始化全局变量
        saver = tf.train.Saver()   # 模型存储初始化

        # 判断是否是第一次训练（判断是否加载模型继续训练），用于间断训练
        chkpt_fname = None   # 文件信息初始值
        if is_restore:    # 加载模型继续训练
            chkpt_fname = tf.train.latest_checkpoint(checkpoint_dir)   # 加载文件并获得文件信息
            if chkpt_fname:
                print("load model......")
                saver.restore(sess, chkpt_fname)   # 加载模型及其参数（必须在Session中使用）
        if not chkpt_fname:    # 第一次训练 —— 删除模型，重新训练；或没有找到对应的模型保存位置
            print("Create model checkpoint dir.....")
            if os.path.exists(checkpoint_dir):   # 判定路径是否存在
                shutil.rmtree(checkpoint_dir)   # 删除路径（嵌套结构）
            os.mkdir(checkpoint_dir)

        z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)   # 产生噪音数据。该数据只产生一次，仅用来作为迭代后生成模型的验证图像，不作为网络的数据
        steps = int(10000 / batch_size)
        for i in range(sess.run(global_step), max_epoch):
            for j in range(steps):
                print("Epoch:{}, Steps:{}".format(i, j))
                # 1. 获取batch_size个真实数据（标签都是1，所以不用获取）
                x_data_value, _ = mnist.train.next_batch(batch_size)
                x_data_value = 2 * x_data_value.astype(np.float32) - 1
                # 2. 获取训练数据（噪音）
                z_value = np.random.normal(0, 1, size=[batch_size, z_size]).astype(np.float32)

                # 3. 执行判断操作，训练D模型
                sess.run(d_trainer, feed_dict={x_data: x_data_value, z_prior: z_value, keep_prob: 0.7})
                # 4. 执行生成的训练，训练G模型
                sess.run(g_trainer, feed_dict={x_data: x_data_value, z_prior: z_value, keep_prob: 0.7})

            # 每完成一次迭代，保存生成模型的输出图像
            x_generated_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
            save_image(x_generated_val, "output_sample/random_sample{}.jpg".format(i))   # output_sample文件夹下的random_samplei.jpg图形

            # 每完成一次迭代，保存模型
            # -》os.path.join()：将多个路径组合后返回
            sess.run(tf.assign(global_step, i + 1))   # 更新迭代次数
            saver.save(sess, os.path.join(checkpoint_dir, 'model'), global_step=global_step)   # output文件夹下的model-global_step.data及其附属文件（.index和.meta）


def test():   # 模型测试
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name='z_prior')

    x_generated, _ = build_generator(z_prior)
    chkpt_fname = tf.train.latest_checkpoint(checkpoint_dir)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()

        if chkpt_fname:
            print("load model......")
            saver.restore(sess, chkpt_fname)
        z_sample = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        x_generated_val = sess.run(x_generated, feed_dict={z_prior: z_sample})
        save_image(x_generated_val, "output_sample/test.jpg")


if __name__ == '__main__':
    train()
    # test()
