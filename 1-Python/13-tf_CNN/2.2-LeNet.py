# -- encoding:utf-8 --
"""
注意：一般情况下，我们都是直接将网络结构翻译成为这个代码，最多稍微的修改一下网络中的参数（超参数、窗口大小、步长等信息）
https://deeplearnjs.org/demos/model-builder/
https://js.tensorflow.org/#getting-started
Le_Net用的最多
"""

import numpy as np
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
    1、加载数据
    ①、手写数字识别的数据集主要包含三个部分：训练集data.train(5.5w 样本数, 像素点)、测试集data.test(1w, )、验证集data.validation(0.5w, )
    ②、手写数字图片大小是28*28*1像素的图片(黑白)，也就是每个图片由784维的特征描述
'''

class LeNetCalss():
    def __init__(self):
        self.data = input_data.read_data_sets('data/', one_hot=True)
        
        self.train_img = self.data.train.images   # 训练集像素数据
        self.train_label = self.data.train.labels   # 训练集类别
        self.test_img = self.data.test.images
        self.test_label = self.data.test.labels
        self.train_sample_number = self.data.train.num_examples   # 训练集样本数
        
        '''
            2、设置超参和占位符数据
        '''
        self.learn_rate_base = 1.0   # 学习率，一般学习率设置的比较小
        self.batch_size = 64   # SGG每次迭代的训练样本数量
        self.bdisplay_step = 1   # 展示信息的间隔大小
        
        self.input_dim = self.train_img.shape[1]   # 训练集样本维度 - 属性值 像素 784
        self.n_classes = self.train_label.shape[1]   # 训练集样本标签维度 - 0-9 10个
        
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='self.x')   # 图像像素数据
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='self.y')   # 图像类型
        learn_rate = tf.placeholder(tf.float32, name='learn_rate')   # 控制学习率减幅
        
    def learn_rate_func(self,epoch):   # 根据给定的迭代批次，更新产生一个学习率的值
        return self.learn_rate_base * (0.9 ** int(epoch / 10))   # 每10次迭代，学习率减小一次

    '''
        3、构建网络结构
        使用tf.get_variable()和tf.variable_scope()来定义w和b，以避开多次命名的麻烦
        
        ①、卷积：
            -》tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", name=None) => 卷积的API
            input：4维格式的输入数据，具体格式和data_format有关；
                当data_format为NHWC时，input的格式为: [self.batch_size, height, weight, channels] => [批次中的图片数目，图片的高度，图片的宽度，图片的通道数]
                当data_format为NCHW时，input的格式为: [self.batch_size, channels, height, weight] => [批次中的图片数目，图片的通道数，图片的高度，图片的宽度]
            filter: 卷积核，4维数据，shape: [height, weight, in_channels, out_channels] => [窗口高度，窗口宽度，输入的channel通道数(上一层图片的深度)，输出的通道数(卷积核数目)]
    
            data_format: 表示输入的数据格式，格式共有两种：NHWC和NCHW，N=>batch样本数目，H=>Height, W=>Weight, C=>Channels；NHWC为默认格式
    
            strides：步长，4维的数据，格式和data_format格式匹配，表示的是在data_format每一维上的移动步长
                当格式为NHWC时，strides的格式为: [batch, in_height, in_weight, in_channels] => [样本上的移动大小，高度的移动大小，宽度的移动大小，深度的移动大小]
                当格式为NCHW时，strides的格式为: [batch,in_channels, in_height, in_weight]
                在样本batch和深度in_channels上的移动距离必须是1，高度和宽度一般取相同移动值；
            padding: 可选参数为"SAME", "VALID"。  对池化同样适用
                当值为SAME候，表示进行填充。如果步长为1，且padding为SAME的时候，经过卷积之后的图像大小不变；②、如果步长不为n，填充为n/2
                当值为VALID时候，表示多余的特征会丢弃；
    
            -》tf.nn.bias_add()：添加偏置项
    
        ②、激励ReLu
            -》tf.nn.relu()：表示max(fetures, 0)，即对于大于0的值不进行处理
            -》tf.nn.relu6()：表示min(max(fetures,0), 6)  存在上界的ReLu，即当输入的值大于6的时候，返回6
            
        ③、池化
            -》tf.nn.max_pool(value, ksize, strides, padding, data_format="NHWC", name=None)：和conv2一样，需要给定窗口大小和步长，
                max_pool()   最大池化      avg_pool()   平均池化
                
                value：输入的数据，[self.batch_size, height, weight, channels]格式
                ksize：指定窗口大小，[batch, in_height, in_weight, in_channels]，其中batch和in_channels必须为1
                strides：指定步长大小，[batch, in_height, in_weight, in_channels],其中batch和in_channels必须为1
    '''
    def Le_net(self):   #  注意variable_scope和get_variable的配合★★★
        # 0. 输入层（黑白图片）
        with tf.variable_scope('input0'):
            self.x = tf.placeholder("float",[None, 28, 28, 1])
            # 将输入的x的格式转换为规定的格式:[None, self.input_dim] -> [Batch, height, weight, channels]
        # 1. 卷积层
        with tf.variable_scope('conv1'):
            net = tf.nn.conv2d(input=self.x, filter=tf.get_variable('w', [5, 5, 1, 20]), strides=[1, 1, 1, 1], padding='SAME')   # (?, 28, 28, 20)   VALID(?, 24, 24, 20)
            net = tf.nn.bias_add(net, tf.get_variable('b', [20]))   # 添加偏置项，(20,1)
            net = tf.nn.relu(net)
        # 2. 池化
        with tf.variable_scope('pool2'):
            net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   # (?, 14, 14, 20)   VALID(?, 12, 12, 20)
        # 3. 卷积
        with tf.variable_scope('conv3'):
            net = tf.nn.conv2d(input=net, filter=tf.get_variable('w', [5, 5, 20, 50]), strides=[1, 1, 1, 1], padding='SAME')   # (?, 14, 14, 50)   VALID(?, 8, 8, 50)
            net = tf.nn.bias_add(net, tf.get_variable('b', [50]))
            net = tf.nn.relu(net)
        # 4. 池化
        with tf.variable_scope('pool4'):
            net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   # (?, 7, 7, 50)   VALID(?, 4, 4, 50)
        # 5. 全连接
        with tf.variable_scope('fc5'):
            net = tf.reshape(net, shape=[-1, 7 * 7 * 50])   # 全连接需要使用2维数据的格式 - net 每行为一个样本，行数据描述了这个样本的特征（像素）
            net = tf.add(tf.matmul(net, tf.get_variable('w', [7 * 7 * 50, 500])), tf.get_variable('b', [500]))   # self.x*W+b  形成500个神经元  [7 * 7 * 50, 500]->(?, 500)
            net = tf.nn.relu(net)
        # 6. 全连接
        with tf.variable_scope('fc6'):
            net = tf.add(tf.matmul(net, tf.get_variable('w', [500, self.n_classes])), tf.get_variable('b', [self.n_classes]))   # (?, 500)->(?, 10)
            predict = tf.nn.softmax(net)   # 多分类问题
    
        return predict
    
    '''
        4、构建损失函数（Adam迭代更新）
        -》tf.train.AdamOptimizer(learning_rate).minimize(loss)：一个寻找全局最优点的优化算法，引入了二次方梯度校正
            AdamOptimizer通过使用动量（参数的移动平均数）来改善传统梯度下降，促进超参数动态调整。
            通过创建标签错误率的摘要标量来跟踪丢失和错误率
        
            相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
    '''
    def control(self):
        self.learn_rate = tf.placeholder("float", None)

        predict = self.Le_net()
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=self.y))
        train = tf.train.AdamOptimizer(learning_rate= self.learn_rate).minimize(loss)

        '''
           5、模型评估
        '''
        correct_label = tf.equal(tf.argmax(predict, axis=1), tf.argmax(self.y, axis=1))
        acc = tf.reduce_mean(tf.cast(correct_label, tf.float32))

        '''
            6、模型训练
        '''
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())   # 初始化
            saver = tf.train.Saver()   # 模型保存

            epoch = 0   # 迭代次数
            while True:
                avg_loss = 0
                total_batch = int(self.train_sample_number / self.batch_size)   # 计算出总的批次
                batch_xs = []
                batch_ys = []

                for i in range(total_batch):
                    batch_xs, batch_ys = self.data.train.next_batch(self.batch_size)
                    batch_xs = np.reshape(batch_xs,(-1, 28, 28, 1))

                    feeds = {self.x: batch_xs, self.y: batch_ys, self.learn_rate: self.learn_rate_func(epoch)}
                    sess.run(train, feed_dict=feeds)   # 模型训练 ★
                    avg_loss += sess.run(loss, feed_dict=feeds)   # 对损失函数值进行累加 .
                    # ★★★ —— 与2.1的区别，从run(loss)可以看出，这里的self.Le_net()中不传参，但是对于网络来说，self.x是必不可少的；
                    # 在deed_dict中传入self.x即起到了调用网络的作用，predict获得了网络的return

                # 计算平均损失值，用所有样本（SGD）的平均损失表示每个样本的损失值
                avg_loss = avg_loss / total_batch

                # 计算准确率
                if (epoch + 1) % self.bdisplay_step == 0:
                    print("批次: %03d 损失函数值: %.9f" % (epoch, avg_loss))

                    feeds = {self.x: batch_xs, self.y: batch_ys, self.learn_rate: self.learn_rate_func(epoch)}
                    train_acc = sess.run(acc, feed_dict=feeds)
                    print("训练集准确率: %.3f" % train_acc)

                    self.test_img = np.reshape(self.test_img,(-1, 28, 28, 1))
                    feeds = {self.x: self.test_img, self.y: self.test_label, self.learn_rate: self.learn_rate_func(epoch)}
                    test_acc = sess.run(acc, feed_dict=feeds)
                    print("测试准确率: %.3f" % test_acc)

                    if train_acc > 0.9 and test_acc > 0.9:   # 迭代停止条件
                        saver.save(sess, './self.data/model')
                        break
                epoch += 1

            # 模型可视化输出
            writer = tf.summary.FileWriter('./self.data/graph', tf.get_default_graph())
            writer.close()

def main():
    L = LeNetCalss()
    L.control()

if __name__ == '__main__':
    main()
