""""""
import tensorflow as tf

# %% 直接赋值型。   数据存放类代码，和计算代码，放在Session内外都可以，且满足基本顺序即可
a = tf.placeholder(dtype=tf.float32, shape=[])

with tf.compat.v1.Session() as sess:
    c = a ** 2

    with tf.name_scope('fig1'):
        tf.summary.scalar('a', a)   # tag: fig1/a
    with tf.name_scope('fig2'):
        tf.summary.scalar('c', c)

    summary_vars = [a]
    smy = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("tjn", sess.graph)
    for i in range(5):
        x = i
        sumers = sess.run(smy, feed_dict={summary_vars[0]: x})
        writer.add_summary(summary=sumers, global_step=i)

# %% 多变量
# a = tf.placeholder(dtype=tf.float32, shape=[])
# b = tf.placeholder(dtype=tf.float32, shape=[])
#
# with tf.compat.v1.Session() as sess:
#     c = a ** 2
#
#     # 添加变量进去
#     tf.compat.v1.summary.scalar('a', a)
#     tf.compat.v1.summary.scalar('b', b)
#     tf.compat.v1.summary.scalar('c', c)
#
#     smy = tf.summary.merge_all()  # 将所有summary全部保存到磁盘，以便tensorboard显示
#
#     sess.run(tf.global_variables_initializer())
#     writer = tf.summary.FileWriter("tjn", sess.graph)
#     for i in range(5):
#         x = i
#         y = i ** 2
#         sumers = sess.run(smy, feed_dict={a: x, b: y})  # 赋值
#         writer.add_summary(summary=sumers, global_step=i)  # 把步骤都记录下来
