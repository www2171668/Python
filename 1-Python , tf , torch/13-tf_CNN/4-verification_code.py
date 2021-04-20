# -- encoding:utf-8 --
"""
验证码识别（假定验证码中只有：数字、大小写字母，验证码的数目是4个，eg: Gx3f）
过程：
1. 使用训练集进行网络训练
训练集数据：使用代码随机的生成一批验证码数据，最好是先随机出10w张验证码的图片，然后利用这10w张图片来训练；否则收敛会特别慢，而且有可能不收敛
训练：直接将验证码输入（输入Gx3f），神经网络的最后一层是4个节点，每个节点输出对应的值为(第一个节点输出：G，第二个节点输出：x，第三个节点输出：3，第四个节点输出：f)
2. 使用测试集对训练好的网络进行测试
3. 当测试的正确率大于75%的时候，模型保存
4. 加载模型，对验证码进行识别
"""

import numpy as np
from captcha.image import ImageCaptcha   # python自带的验证码库
import random
from PIL import Image
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt

'''
    1、生成数据
    -》random.choice(*)：从*（可迭代对象）中随机选择一个返回值
    
    -》ImageCaptcha()：图像初始化
'''
code_char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 'q', 'a', 'z', 'w', 's', 'x', 'e', 'd', 'c', 'r',
                 'f', 'v', 't', 'g', 'b', 'y', 'h', 'n', 'u', 'j',
                 'm', 'i', 'k', 'o', 'l', 'p', 'Q', 'A', 'Z', 'W',
                 'S', 'X', 'E', 'D', 'C', 'R', 'F', 'V', 'T', 'G',
                 'B', 'Y', 'H', 'N', 'U', 'J', 'M', 'I', 'K', 'O',
                 'L', 'P']

def random_code_text(code_size=4):   # 1、随机生成验证码的字符
    code_text = []
    for i in range(code_size):    # code_size:验证码长度
        c = random.choice(code_char_set)   # 返回一个char
        code_text.append(c)
    return code_text   # [*,*,*,*]

def generate_code_image(code_size=4):   # 2、生成验证码的Image对象
    code_text = random_code_text(code_size)   # 获取验证码

    image = ImageCaptcha()   # ①、图像初始化
    code_text = ''.join(code_text)   # ②、把list类型的验证码转字符串类型 ★
    captcha = image.generate(code_text)   # ③、将字符串转换为验证码流 ★
    code_image = np.array(Image.open(captcha))    # ④、使用等距API法，将验证码流转数组图片格式[sample_number,h,w,c] ★

    # 保存验证码图片
    # image.write(code_text, 'captcha/' + code_text + '.jpg')   # 用code_text.jpg为图片命名

    return code_text, code_image   # 返回字符串类型验证码和图片类型验证码


code_char_set_size = len(code_char_set)   # 位置信息（字符编号）
code_char_to_number_dict = dict(zip(code_char_set, range(code_char_set_size)))   # key为char，value为数字编号
code_number_to_char_dict = dict(zip(range(code_char_set_size), code_char_set))   # key为数字编号，value为char

def text_to_vec(code,code_size =4):   # 3、字符串哑编码化
    vec = np.zeros((code_size, code_char_set_size))   # [4, 哑编码]
    k = 0
    for i in code:   # 遍历4个字符
        index = code_char_to_number_dict[i]   # 获取字符对应的编号
        vec[k][index] = 1   # 设置第k个验证码第index编号值为1
        k += 1
    return vec.ravel()   # 将4维行向量转一维行向量化

def random_next_batch(batch_size=64, code_size=4):   # 3、生成训练数据集：批量产生随机数列，并随机获取下一个批次的数据，

    batch_x = []
    batch_y = []

    for i in range(batch_size):   # MSGD法
        code, image = generate_code_image(code_size)   # 生成验证码
        code_number = text_to_vec(code)   # 字符串哑编码化
        batch_x.append(image)   # 图像数据
        batch_y.append(code_number)   # 图像

    return np.array(batch_x), np.array(batch_y)

'''
    2、设置全局参数
    -》dict(zip(key,value)) ：用现有list构造字典
'''
keep_prob = 0.75   # 用于控制神经元保留率的超餐，0.75表示75%的神经元保留下来，随机删除其中的25%的神经元(其实相当于将25%的神经元的输出值设置为0)
code_size = 4   # 验证码中的字符数目


'''
    3、构建CNN网络结构
    conv -> relu6 -> max_pool -> conv -> relu6 -> max_pool -> dropout -> conv -> relu6 -> max_pool -> full connection -> full connection
    
    -》tf.nn.dropout(*,keep_prob)：网络中进行Dropout时，神经元的保留率（有多少神经元被保留下来）
'''
def code_cnn(x):

    x_shape = x.get_shape()

    kernel_size_1 = 32   # 卷积核的数目
    kernel_size_2 = 64
    kernel_size_3 = 64
    unit_number_1 = 1024   # unit_number_k：全连接的输出神经元数目
    unit_number_2 = code_size * code_char_set_size   # 哑编码数量即为输出神经元数目

    with tf.variable_scope('net', initializer=tf.random_normal_initializer(0, 0.1), dtype=tf.float32):   # 给定网络初值
        with tf.variable_scope('conv1'):
            net = tf.nn.conv2d(input=x, filter = tf.get_variable('w', shape=[5, 5, x_shape[3], kernel_size_1]), strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, tf.get_variable('b',[kernel_size_1]))
        with tf.variable_scope('relu1'):
            net = tf.nn.relu6(net)
        with tf.variable_scope('max_pool1'):
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('conv2'):
            net = tf.nn.conv2d(net, filter=tf.get_variable('w', shape=[3, 3, kernel_size_1, kernel_size_2]), strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, tf.get_variable('b',shape=[kernel_size_2]))
        with tf.variable_scope('relu2'):
            net = tf.nn.relu6(net)
        with tf.variable_scope('max_pool2'):
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('dropout1'):
            # -》
            tf.nn.dropout(net, keep_prob=keep_prob)
        with tf.variable_scope('conv3'):
            w = tf.get_variable('w', shape=[3, 3, kernel_size_2, kernel_size_3])
            b = tf.get_variable('b', shape=[kernel_size_3])
            net = tf.nn.conv2d(net, w, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, b)
        with tf.variable_scope('relu3'):
            net = tf.nn.relu6(net)
        with tf.variable_scope('max_pool3'):
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('fc1'):
            net_shape = net.get_shape()
            net_sample_feature_number = net_shape[1] * net_shape[2] * net_shape[3]   # 改变形状
            net = tf.reshape(net, shape=[-1, net_sample_feature_number])
            w = tf.get_variable('w', shape=[net_sample_feature_number, unit_number_1])
            b = tf.get_variable('b', shape=[unit_number_1])
            net = tf.add(tf.matmul(net, w), b)
        with tf.variable_scope('fc2'):
            w = tf.get_variable('w', shape=[unit_number_1, unit_number_2])
            b = tf.get_variable('b', shape=[unit_number_2])
            net = tf.add(tf.matmul(net, w), b)

    return net


def train_code_cnn(model_path):
    ''''''
    '''
        1、设置模型的超参和占位符数据
    '''
    in_image_height = 60
    in_image_weight = 160
    x = tf.placeholder(tf.float32, shape=[None, in_image_height, in_image_weight, 1], name='x')   # 黑白图
    y = tf.placeholder(tf.float32, shape=[None, code_size * code_char_set_size], name='y')   # 哑编码类型

    '''
        2、引入网络模型 + 构建损失函数（验证码四个位置的值，相等的位置为False，不相等的为True）+ 评估
    '''
    predict = code_cnn(x)

    # ①、分类法：对所有验证码预测正确判定值做均值，均值越接近1，准确率越高
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
    train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    # 评估
    correct_label = tf.equal(tf.argmax(predict, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_label,tf.float32))

    # ②、回归法：
    predict = tf.reshape(predict, [-1, code_size, code_char_set_size])   # [batch4行，code_char_set_size列，一行一个char
    correct = tf.reshape(y, [-1, code_size, code_char_set_size])

    # 评估：得到预测值中的最大概率值的下标。注意是三维，所以axis=2
    correct_label = tf.equal(tf.argmax(predict, axis=2), tf.argmax(y, axis=2))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(correct_label), tf.float32))

    '''
        3、模型训练
    '''
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 1
        while True:
            # 1. 获取当前迭代的训练数据
            batch_x, batch_y = random_next_batch(batch_size=64, code_size=code_size)
            # 2. 对数据进行处理
            batch_x = tf.image.rgb_to_grayscale(batch_x)   # 彩色图片转黑白
            batch_x = tf.image.resize_images(batch_x, size=(in_image_height, in_image_weight),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)    # 图像大小重置
            # 3. 模型训练
            _, loss_, accuracy_ = sess.run([train, loss, accuracy], feed_dict={x: batch_x.eval(), y: batch_y})
            # 由于batch_x经过了图像转化，此时需要用.eval()转为数组类型 ★★
            # train是用来模型训练的，没有返回值，用 _ 占位 ★
            print("Step:{}, 损失值:{}, 训练集准确率:{}".format(step, loss_, accuracy_))

            if step % 10 == 0:
                # 获取当前迭代的测试数据。实际和获取训练集是一样的，只是后面的feed_dict传的值不同
                test_batch_x, test_batch_y = random_next_batch(batch_size=64, code_size=code_size)
                test_batch_x = tf.image.rgb_to_grayscale(test_batch_x)
                test_batch_x = tf.image.resize_images(test_batch_x, size=(in_image_height, in_image_weight),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                acc = sess.run(accuracy, feed_dict={x: test_batch_x.eval(), y: test_batch_y})
                print("测试集准确率:{}".format(acc))

                if acc > 0.7 and accuracy_ > 0.7:    # 如果模型准确率0.7，模型保存，然后退出
                    saver.save(sess, model_path, global_step=step)
                    break

            step += 1


if __name__ == '__main__':
    train_code_cnn('./model/code/capcha.model')
    # TODO: 作业，假定这个模型运行一段时间后，可以顺利的保存模型；那么自己加入一些代码，代码功能：使用保存好的模型对验证码做一个预测，最终返回值为验证码上的具体字符串； => 下周四，晚自习我讲一下怎么写
    # TODO: 作业，车牌照的识别，分成两个过程：1. 从车辆图片中提取出车牌照区域的值，然后2.对车牌照图片做一个预测； 简化：认为车牌只有蓝牌 ===> 下下周二或者周四，我带着大家写写（大家有一个礼拜的时间自己考虑一下怎么实现）

'''
    展示图像
    -》ax.text()：显示文字图片
'''
    # code,image = generate_code_image(4)
    #
    # ax = plt.figure()   # 背景
    # ax.text(0.1,0.9,code,ha='center',va='center')   # 0.1,0.9控制文本出现在图片的左上角
    # plt.imshow(image)
    # plt.show()
