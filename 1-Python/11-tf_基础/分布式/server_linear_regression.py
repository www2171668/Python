# -- encoding:utf-8 --
"""
Create by ibf on 2018/5/6
"""

import numpy as np
import tensorflow as tf

np.random.seed(28)

# TODO： 将这个代码整理成为单机运行的

# 1. 配置服务器相关信息
# 因为tensorflow底层代码中，默认就是使用ps和work分别表示两类不同的工作节点
# ps：变量/张量的初始化、存储相关节点
# work: 变量/张量的计算/运算的相关节点
ps_hosts = ['127.0.0.1:33331', '127.0.0.1:33332']
work_hosts = ['127.0.0.1:33333', '127.0.0.1:33334', '127.0.0.1:33335']
cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'work': work_hosts})

# 2. 定义一些运行参数(在运行该python文件的时候就可以指定这些参数了)
tf.app.flags.DEFINE_integer('task_index', default_value=0, docstring="Index of task within the job")
FLAGS = tf.app.flags.FLAGS


# 3. 构建运行方法
# -》tf.train.replica_device_setter
def main(_):
    # ①.图的构建
    with tf.device(
            tf.train.replica_device_setter(worker_device='/job:work/task:%d' % FLAGS.task_index, cluster=cluster)):   # 通过task_index明确程序运行位置
        # a. 构建一个样本的占位符信息，用于样本输入
        x_data = tf.placeholder(tf.float32, [10])   # 10 小批量梯度下降。1 随机梯度下降
        y_data = tf.placeholder(tf.float32, [10])

        # b. 构建模型
        w = tf.Variable(initial_value=tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0), name='w')
        b = tf.Variable(initial_value=tf.zeros([1]), name='b')
        # 构建一个预测值
        y_predict = w * x_data + b   # 使每次更新只使用了部分样本 - 用占位符来控制训练数据

        # c. 构建损失函数
        loss = tf.reduce_mean(tf.square(y_predict - y_data + 1e-8), name='loss')

        # d. 优化训练
        global_step = tf.Variable(0, name='global_step', trainable=False)   # 设置步数。trainable=False：不加入到优化中
        # 以随机梯度下降的方式优化损失函数
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)

        train = optimizer.minimize(loss, name='train', global_step=global_step)   # 在优化过程中，会对global_step进行累加操作

    # ②.图的运行
    hooks = [tf.train.StopAtStepHook(last_step=10000000)]   # 更新last_step步数后，结束更新
    with tf.train.MonitoredTrainingSession(   # 分布式条件下，用MonitoredTrainingSession，不用Session
        master='grpc://' + work_hosts[FLAGS.task_index],
        is_chief=(FLAGS.task_index == 0),  # 是否进行变量的初始化，设置为true，表示进行初始化
        checkpoint_dir='./tmp',   # 保存checkpoint文件信息
        save_checkpoint_secs=None,   # 保存checkpoint的间隔时间，即运行过程中的断点信息
        hooks=hooks  # 结束条件
    ) as mon_sess:
        while not mon_sess.should_stop():   # -》should_stop()：停止更新
            N = 10
            train_x = np.linspace(1, 6, N) + np.random.normal(loc=0.0, scale=2, size=N)   # 构建数据
            train_y = 14 * train_x - 7 + np.random.normal(loc=0.0, scale=5.0, size=N)
            _, step, loss_v, w_v, b_v = mon_sess.run([train, global_step, loss, w, b],
                                                     feed_dict={x_data: train_x, y_data: train_y})   # 传入训练集数据
            if step % 100 == 0:   # 每训练100次输出一次
                print('Step:{}, loss:{}, w:{}, b:{}'.format(step, loss_v, w_v, b_v))


if __name__ == '__main__':
    tf.app.run()
