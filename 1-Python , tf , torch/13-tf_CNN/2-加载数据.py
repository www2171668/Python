# -- encoding:utf-8 --
"""
注意：一般情况下，我们都是直接将网络结构翻译成为这个代码，最多稍微的修改一下网络中的参数（超参数、窗口大小、步长等信息）
https://deeplearnjs.org/demos/model-builder/
https://js.tensorflow.org/#getting-started
"""

import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
    1、加载数据
    ①、手写数字识别的数据集主要包含三个部分：训练集(5.5w, data.train)、测试集(1w, data.test)、验证集(0.5w, data.validation)
    ②、手写数字图片大小是28*28*1像素的图片(黑白)，也就是每个图片由784维的特征描述
'''
data = input_data.read_data_sets('data/', one_hot=True)

train_img = data.train.images   # 训练集像素数据
train_label = data.train.labels   # 训练集类别
test_img = data.test.images
test_label = data.test.labels
train_sample_number = data.train.num_examples   # 训练集样本数

'''
    2、设置超参和占位符数据
'''
learn_rate_base = 1.0   # 学习率，一般学习率设置的比较小
batch_size = 64   # SGG每次迭代的训练样本数量
display_step = 1   # 展示信息的间隔大小

input_dim = train_img.shape[1]   # 训练集样本维度 - 属性值 像素 784
n_classes = train_label.shape[1]   # 训练集样本标签维度 - 0-9 10个

print(train_img.shape)
print(input_dim)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   # 初始化

