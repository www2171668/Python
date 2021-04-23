# -- encoding:utf-8 --
"""
17中花数据分类，是VGG网络初赛时候的数据集，现在网上没有下载；现在唯一一份数据集在tflearn这个框架中默认自带
tflearn这个框架起始是在tensorflow基础上的一个封装
tflearn安装：pip install tflearn
"""

from tflearn.datasets import oxflower17   # 导入花的图像包
import tensorflow as tf

'''
    1、加载数据
'''
X, Y = oxflower17.load_data(dirname="17flowers", one_hot=True)
print(X.shape)  # [sample_number,224,224,3]
print(Y.shape)  # [sample_number,17]

'''
    2、设置超参和占位符数据
'''
learn_rate = 0.1
batch_size = 16
total_sample_number = X.shape[0]   # 样本数量

x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x')   # 输入四维样本集，在input时不用转格式
y = tf.placeholder(tf.float32, shape=[None, 17], name='y')

# 将tf.get_variable写成一个方法，简化后面的写法。这样写和直接调用底层的tf.get_variable()方法是一样的
def get_variable(name, shape=None, dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.1)):
    return tf.get_variable(name, shape, dtype, initializer)


'''
    3、构建网络结构
    -》tf.nn.lrn(input, depth_radius, bias, alpha, beta=, name) ：局部响应归一化，即对卷积核的输出值做归一化
        depth_radius：对应ppt公式上的n
        bias：对应ppt公式上的k
        alpha：对应ppt公式上的α
        beta：对应ppt公式上的β
'''
def vgg_network(x, y):
    # 设置卷积核数量
    net1_kernal_num = 32
    net3_kernal_num = 64
    net5_kernal_num_1 = 128
    net5_kernal_num_2 = 128
    net7_kernal_num_1 = 256
    net7_kernal_num_2 = 256
    net9_kernal_num_1 = 256
    net9_kernal_num_2 = 256
    net11_unit_num = 1000
    net12_unit_num = 1000
    net13_unit_num = 17   # 17分类问题

    # 1. cov3-64
    with tf.variable_scope('net1'):
        net = tf.nn.conv2d(input=x, filter=get_variable('w', [3, 3, 3, net1_kernal_num]), strides=[1, 1, 1, 1],padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [net1_kernal_num]))
        net = tf.nn.relu(net)
        net = tf.nn.lrn(net)
    # 2. maxpool
    with tf.variable_scope('net2'):
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 3. conv3-128
    with tf.variable_scope('net3'):
        net = tf.nn.conv2d(input=net, filter=get_variable('w', [3, 3, net1_kernal_num, net3_kernal_num]),strides=[1, 1, 1, 1],padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [net3_kernal_num]))
        net = tf.nn.relu(net)
    # 4. maxpool
    with tf.variable_scope('net4'):
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 5. conv3-256 conv3-256
    with tf.variable_scope('net5'):   # 范围定义net5中有两个卷积操作
        net = tf.nn.conv2d(input=net, filter=get_variable('w1', [3, 3, net3_kernal_num, net5_kernal_num_1]),strides=[1, 1, 1, 1],padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b1', [net5_kernal_num_1]))
        net = tf.nn.relu(net)

        net = tf.nn.conv2d(input=net, filter=get_variable('w2', [3, 3, net5_kernal_num_1, net5_kernal_num_2]),strides=[1, 1, 1, 1],padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b2', [net5_kernal_num_2]))
        net = tf.nn.relu(net)
    # 6. maxpool
    with tf.variable_scope('net6'):
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 7. conv3-512 conv3-512
    with tf.variable_scope('net7'):
        net = tf.nn.conv2d(input=net, filter=get_variable('w1', [3, 3, net5_kernal_num_2, net7_kernal_num_1]),strides=[1, 1, 1, 1],padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b1', [net7_kernal_num_1]))
        net = tf.nn.relu(net)

        net = tf.nn.conv2d(input=net, filter=get_variable('w2', [3, 3, net7_kernal_num_1, net7_kernal_num_2]),strides=[1, 1, 1, 1],padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b2', [net7_kernal_num_2]))
        net = tf.nn.relu(net)
    # 8. maxpool
    with tf.variable_scope('net8'):
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 9. conv3-512 conv3-512
    with tf.variable_scope('net9'):
        net = tf.nn.conv2d(input=net, filter=get_variable('w1', [3, 3, net7_kernal_num_2, net9_kernal_num_1]),strides=[1, 1, 1, 1],padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b1', [net9_kernal_num_1]))
        net = tf.nn.relu(net)

        net = tf.nn.conv2d(input=net, filter=get_variable('w2', [3, 3, net9_kernal_num_1, net9_kernal_num_2]),strides=[1, 1, 1, 1],padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b2', [net9_kernal_num_2]))
        net = tf.nn.relu(net)
    # 10. maxpool
    with tf.variable_scope('net10'):
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 11. fc
    with tf.variable_scope('net11'):
        # 将四维的数据转换为两维的数据
        shape = net.get_shape()   # 得到当前网络层数据的shape值 [NHWC]
        feature_number = shape[1] * shape[2] * shape[3]
        net = tf.reshape(net, shape=[-1, feature_number])

        net = tf.add(tf.matmul(net, get_variable('w', [feature_number, net11_unit_num])),get_variable('b', [net11_unit_num]))
    # 12. fc
    with tf.variable_scope('net12'):
        net = tf.add(tf.matmul(net, get_variable('w', [net11_unit_num, net12_unit_num])),get_variable('b', [net12_unit_num]))
    # 13. fc
    with tf.variable_scope('net13'):
        net = tf.add(tf.matmul(net, get_variable('w', [net12_unit_num, net13_unit_num])),get_variable('b', [net13_unit_num]))

    # 14. softmax
    with tf.variable_scope('net14'):
        # softmax
        predict = tf.nn.softmax(net)

    return predict


'''
    4、构建损失函数
'''
predict = vgg_network(x, y)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
train = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)

'''
   5、模型评估
'''
correct_label = tf.equal(tf.argmax(y, axis=1), tf.argmax(predict, axis=1))
acc = tf.reduce_mean(tf.cast(correct_label, tf.float32))

'''
    6、模型训练
'''
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(10):      # 迭代达到一定数量时停止
        total_batch = int(total_sample_number / batch_size) - 5   # 用最后5个批次的数据作为测试集

        for step in range(total_batch):
            # 对当前批次的数据进行训练
            train_x = X[step * batch_size:step * batch_size + batch_size]   # 批次开始：批次结束
            train_y = Y[step * batch_size:step * batch_size + batch_size]
            sess.run(train, feed_dict={x: train_x, y: train_y})   # 每一次更新时，参数都会更新，所以即使每次输入的训练集一样，参数也会得到更新

            # # 每更新10个批次，就输出一次
            # if step % 10 == 0:
            #     Batch_loss, accuracy = sess.run([loss, acc], feed_dict={x: train_x, y: train_y})   # 等号左右两边不能重名，否则会报float错
            #     print('迭代次数:{}, 批次：{}, 训练集损失值：{}, 训练集准确率：{}'.format(epoch, step, Batch_loss, accuracy))

        if epoch % 2 == 0:
            test_loss, accuracy = sess.run([loss, acc], feed_dict={x: train_x, y: train_y})
            print('步骤：{}，训练集损失值：{}, 训练集准确率：{}'.format(epoch, test_loss, accuracy))

            test_x,test_y = X[step * batch_size:],Y[step * batch_size:]   # 获取测试集数据
            print(sess.run(test_x))
            train_loss, accuracy = sess.run([loss, acc], feed_dict={x: test_x, y: test_y})
            print('步骤：{}，测试集损失值：{}，测试集准确率：{}'.format(epoch, train_loss, accuracy))

print("End！！")
