
'''
    特征x1和x2, 5*x1-3x2 > 0是第一个类别，5*x1-3x2 <= 0是第二个类别，构建数据[x1,x2,y]
'''

import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer, OneHotEncoder

# 1. 模拟数据产生
# -》OneHotEncoder 添加列，与第一列成二值化关系
# -》Binarizer(threshold).fit_transform() 二值化，>0的=1，<=0的=0
# -》.toarray()：转ndarray
np.random.seed(28)
n = 100
x_data = np.random.normal(loc=0, scale=2, size=(n, 2))
y_data = np.dot(x_data, np.array([[5], [-3]]))  # -》np.dot 点乘
y_data = Binarizer(threshold=0).fit_transform(y_data)   # 二值化 - [n,1]数据
y_data = OneHotEncoder().fit_transform(y_data).toarray()   # 增加列 - [n,2]数据

# 构建用于分类图数据
t1 = np.linspace(-8, 10, 100)
t2 = np.linspace(-8, 10, 100)
xv, yv = np.meshgrid(t1, t2)
x_test = np.dstack((xv.flat, yv.flat))[0]

# plt.scatter(x_data[y_data[:, 0] == 0][:, 0], x_data[y_data[:, 0] == 0][:, 1], s=50, marker='+', c='red')
# plt.scatter(x_data[y_data[:, 0] == 1][:, 0], x_data[y_data[:, 0] == 1][:, 1], s=50, marker='x', c='blue')
# plt.show()

# 构建数据输入占位符x和y
# [None, 2]: None的意思表示维度未知(也就是可以传入任意样本数)
# x: 2表示变量有2个特征属性，即输入样本的维度数目
# y: 2表示样本样本的目标属性的类别数目(有多少个类别，这里就是几)
x = tf.placeholder(tf.float32, [None, 2], name='x')
y = tf.placeholder(tf.float32, [None, 2], name='y')

# 2. 构建softmax模型
# 构建权重w和偏置项b
# w第一个2表示变量有2个特征属性，即输入样本的维度数目
# w第二个2表示样本的目标属性的类别数目
# b中的2表示样本的目标属性的类别数目
w = tf.Variable(tf.zeros([2, 2]), name='w')
b = tf.Variable(tf.zeros([2]), name='b')
# act(Tensor类型)是通过softmax函数转换后的一个概率值(矩阵的形式)
act = tf.nn.softmax(tf.matmul(x, w) + b)  # x和w的位置不能变

# 3. 构建损失函数 - 交叉熵
# tf.reduce_sum: 求和，当参数为矩阵的时候，axis等于1的时候，对每行求解和 => 和numpy API中的axis参数意义一样
# tf.reduce_mean: 求均值，当不给定任何axis参数的时候，表示求解全部所有数据的均值，axis=1，按行遍历
loss = -tf.reduce_mean(tf.reduce_mean(y * tf.log(act), axis=1))   # y * tf.log(act) 数乘

# 4. 模型训练 - 使用梯度下降，最小化误差
# learning_rate: 要注意，不要过大，过大可能不收敛，也不要过小，过小收敛速度比较慢
# .minimize(loss)：对loss函数进行最小化（优化）
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# 5. 建立模型评估指标
# 得到预测的类别是那一个
# tf.argmax:对矩阵按行或列计算最大值对应的下标，和numpy中的一样。axis=1，按行遍历
# tf.argmax(act, axis=1)：获取模型预测样本的所属类别
# tf.equal:是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
pred = tf.equal(tf.argmax(act, axis=1), tf.argmax(y, axis=1))
# 计算正确率（True=1，False=0）
# -》tf.reduce_mean：求均值
# -》tf.cast：类型转换
acc = tf.reduce_mean(tf.cast(pred, tf.float32))

# 6. 设置迭代参数
# 总共训练迭代次数
training_epochs = 50
# 批次数量，每次用10个样本
num_batch = int(n / 10)
# 训练迭代次数（每5个批次打印一次数据）
display_step = 5

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        # 迭代训练
        avg_loss = 0   # 存储损失函数
        # -》np.random.permutation(n)：产生一个n的序列，并打乱数据顺序
        index = np.random.permutation(n)
        for i in range(num_batch):
            # 获取传入进行模型训练的数据对应索引
            indexs = index[i * 10:(i + 1) * 10]   # 每次训练的样本集序列
            # 构建传入的feed参数
            feeds = {x: x_data[indexs], y: y_data[indexs]}
            # 进行模型训练
            sess.run(train, feed_dict=feeds)
            # 可选：获取损失函数值
            avg_loss += sess.run(loss, feed_dict=feeds) / num_batch   # 均值

        # 满足5次的一个迭代，进行打印
        if epoch % display_step == 0:
            feeds_train = {x: x_data, y: y_data}
            train_acc = sess.run(acc, feed_dict=feeds_train)
            print("迭代次数: %03d/%03d 损失值: %.9f 训练集上准确率: %.3f" % (epoch, training_epochs, avg_loss, train_acc))

    # 对分类图的数据网格点进行预测
    # y_label: None*2的矩阵，概率值
    y_label = sess.run(act, feed_dict={x: x_test})
    # 根据softmax分类的模型理论，获取每个样本对应出现概率最大的(值最大的)
    # y_label：None*1的矩阵，标签类型
    y_label = np.argmax(y_label, axis=1)

print("模型训练完成")
# 画图展示一下
cm_light = mpl.colors.ListedColormap(['#bde1f5', '#f7cfc6'])
y_label = y_label.reshape(xv.shape)
plt.pcolormesh(xv, yv, y_label, cmap=cm_light)  # 预测值
plt.scatter(x_data[y_data[:, 0] == 0][:, 0], x_data[y_data[:, 0] == 0][:, 1], s=50, marker='+', c='red')
plt.scatter(x_data[y_data[:, 0] == 1][:, 0], x_data[y_data[:, 1] == 0][:, 1], s=50, marker='o', c='blue')
plt.show()
