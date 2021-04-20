# -- encoding:utf-8 --

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

'''
    1、加载数据 - 数字图形预测
    通常在多分类问题中都需要做哑编码，以配合softmax使用，以找到分类中概率最大的分类项
'''
mnist = input_data.read_data_sets('data/', one_hot=True)

'''
    2、构建神经网络(4层结构)   
    注意神经元的形状 [k层神经元数量,k+1层神经元数量]   ★★★ 
    偏置项的形状 [k+1层神经元数量]   ★★★ 
'''
n_input = 784  # 样本（图像）是28*28像素的
n_unit_hidden_1 = 256  # 第一层hidden中的神经元数目
n_unit_hidden_2 = 128  # 第二层hidden中的神经元数目
n_classes = 10  # 输出的类别数目

# 通过占位符表示x和y，使得训练集和测试集数据都可以用x和y表示。同时也要注意，凡是会用到数据的函数，都要feed_dict x和y
x = tf.placeholder(tf.float32, shape=[None, n_input], name='x')   # None：样本数不限制
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')

weights = {
    "w1": tf.Variable(tf.random_normal(shape=[n_input, n_unit_hidden_1], stddev=0.1)),   # stddev：标准差
    "w2": tf.Variable(tf.random_normal(shape=[n_unit_hidden_1, n_unit_hidden_2], stddev=0.1)),
    "out": tf.Variable(tf.random_normal(shape=[n_unit_hidden_2, n_classes], stddev=0.1))
}
biases = {
    "b1": tf.Variable(tf.random_normal(shape=[n_unit_hidden_1], stddev=0.1)),   # b是和下一个节点连接的
    "b2": tf.Variable(tf.random_normal(shape=[n_unit_hidden_2], stddev=0.1)),
    "out": tf.Variable(tf.random_normal(shape=[n_classes], stddev=0.1))
}

'''
    2、构建网络模型（BP神经网络） ★★★ 注意tf.matmul的顺序 y = aw + b
    -》tf.nn.sigmoid(*)：sigmoid激活函数
'''
# ①、BF过程
def multiplayer_perceptron(_X, _weights, _biases):
    # 输入 -> 第二层
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))   # 一个样本*一列权重
    # 第二层 -> 第三层
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, _weights['w2']), _biases['b2']))
    # 第三层 -> 输出
    return tf.matmul(layer2, _weights['out']) + _biases['out']

# 构建当前BP过程产生的预测值，返回是各个标签的概率值组成的数组
predict = multiplayer_perceptron(x, weights, biases)

# ②、FP过程
'''
    3、构建损失函数（MBGD迭代更新）
    -》tf.nn.softmax_cross_entropy_with_logits(logits,labels): 计算softmax中每个样本的交叉熵，logits指定预测值，labels指定实际值
        在图形中显示为一些slice操作
'''
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))   # 多类型分类问题
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

'''
    4、评估
    -》tf.argmax(x, axis=1)：获取行中最大x
    -》tf.cast(x,dtype)：将x数据转换为dtype所表示的类型
'''
correct_label = tf.equal(tf.argmax(predict, axis=1), tf.argmax(y, axis=1))   # 获取最大概率的预测类型。预测值和实际值相等时，predict_label为1，否则为0
corr_rate = tf.reduce_mean(tf.cast(correct_label, tf.float32))   # 以预测正确样本数量的均值为正确率
# 在这里将数据转为tensor类型的float，True为1，Fasle为0

'''
   5、模型训练
    -》mnist.train.next_batch(n)：从训练集里一次提取n张图片数据来训练，返回n张图片的像素点数据）和标签。返回值均为list类型值
'''
batch_size = 100  # MBGD中每次处理的图片数
display_step = 4  # 每4次迭代打印一次结果

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   # 初始化
    saver = tf.train.Saver()   # 模型保存、持久化

    epoch = 0   # 迭代次数
    while True:
        avg_loss = 0    # 每次迭代损失值清零

        total_batch = int(mnist.train.num_examples / batch_size)   # 批次数量 = 训练集样本数/单次处理样本数
        for i in range(total_batch):   # total_batch：模型训练需要执行的总批次
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feeds = {x: batch_xs, y: batch_ys}
            sess.run(train, feed_dict=feeds)   # ①、模型训练
            # ★★★ 程序运到run时，虽然feed_dict会传值x给网络，但是程序并不会重读网络结构，当时初始化的参数（权重和偏置项）不会再次重置
            # 新的参数只会在GradientDescentOptimizer()中进行迭代更新

            avg_loss += sess.run(loss, feed_dict=feeds)   # ②、对损失函数值进行累加。这一步不对网络参数进行更新，只是方便计算损失值 ★

        # 计算平均损失值，用所有样本（MBGD）的平均损失表示每个样本的损失值
        avg_loss = avg_loss / total_batch

        # 计算准确率
        if (epoch + 1) % display_step == 0:
            print("批次: %03d 损失函数值: %.9f" % (epoch, avg_loss))

            feeds = {x: mnist.train.images, y: mnist.train.labels}   # 传入训练集数据
            train_acc = sess.run(corr_rate, feed_dict=feeds)
            print("训练集准确率: %.3f" % train_acc)

            feeds = {x: mnist.test.images, y: mnist.test.labels}   # 传入测试集数据
            test_acc = sess.run(corr_rate, feed_dict=feeds)
            print("测试准确率: %.3f" % test_acc)

            if train_acc > 0.9 and test_acc > 0.9:   # 迭代停止条件
                saver.save(sess, './mn/model')   # 保存最后的结果
                break   # 跳出最外围循环，即while循环
                
        epoch += 1

    writer = tf.summary.FileWriter('./mn/graph', tf.get_default_graph())   # 模型可视化处理
    writer.close()

'''
    当代码运行起来以后，准确率大概在92%左右浮动。添加如下代码可以大致查看是什么图片让预测不准
'''

# for i in range(0, len(mnist.test.images)):  # for循环内指明一旦result为false，就表示出现了预测值和实际值不符合的图片，然后我们把值和图片分别打印出来看看
#   result = sess.run(predict_label, feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]])})
#   if not result:
#     print('预测值是：',sess.run(predict, feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]])}))
#     print('实际值是：',sess.run(y,feed_dict={x: np.array([mnist.test.images[i]]), y: np.array([mnist.test.labels[i]])}))
#     one_pic_arr = np.reshape(mnist.test.images[i], (28, 28))
#     pic_matrix = np.matrix(one_pic_arr, dtype="float")
#     plt.imshow(pic_matrix)
#     plt.show()
#     break



