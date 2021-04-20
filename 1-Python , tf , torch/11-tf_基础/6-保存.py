""""""
import tensorflow as tf
import numpy as np

# %% 数据保存    tf.train.Saver()       Saver().save(Session,'path')        模型和数据ckpt文件类型
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
   sess.run(init)
   save_path = saver.save(sess, "my_net/save_net.ckpt")

# %% 数据读取   tf.train.Saver().restore(Session,'path')  saver.restore()   恢复数据，需要保证图的构建过程和之前的模型完全相同
# 重定义模型，无需初始化
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")     # * 数据、类型、name 都要和加载模型一一对应
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

saver = tf.train.Saver()        # 可以修改修改加载模型时的映射关系：saver = tf.train.Saver({"key": value, ...})   key：模型变量name，value - 当前变量name
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights:{},biases:{}".format(sess.run(W),sess.run(b)) )

# %% 图加载   tf.train.import_meta_graph('path').restore()
saver = tf.train.import_meta_graph('./model/model.ckpt.meta')  # './model/model.ckpt.meta'为图数据
with tf.Session() as sess:
    saver.restore(sess, './model/model.ckpt')
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
'''
    tf.get_default_graph().get_tensor_by_name("add:0")
            -》get_default_graph：获取默认图
            -》get_tensor_by_name：通过名称获取张量，name为操作名，add:0为第一个加法操作得值
'''
