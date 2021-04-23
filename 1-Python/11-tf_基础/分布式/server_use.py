# -- encoding:utf-8 --
"""
分布式集群使用(最简单的)
Create by ibf on 2018/5/6
"""

import tensorflow as tf
import numpy as np

'''
    -》/job：固定词
    -》ps：使用的是cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'work': work_hosts})中的ps value ps_hosts
    -》task:0：ps_host[0] - ps_hosts = ['127.0.0.1:33331', '127.0.0.1:33332']的第一个
'''

# 1. 构建图 - 使用第一个设备
with tf.device('/job:ps/task:0'):   # 指定位置'/job:ps/task:0/gpu:0'  也可以嵌套地添加with tf.device('gpu:0'):
    # 2. 构造数据
    x = tf.constant(np.random.rand(100).astype(np.float32))

# 3. 使用第二个设备
with tf.device('/job:work/task:1'):
    # 4. 构建数据
    y = x * 0.1 + 0.3

# 4. 运行 - 日志打印在target位置，在cmd中查看
with tf.Session(target='grpc://127.0.0.1:33331',   # -》target：指定Session运行设备，即设置Session主节点
                config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    print(sess.run(y))   # 在并行运行开始后，才会输出
