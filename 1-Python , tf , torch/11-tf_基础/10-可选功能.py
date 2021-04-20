""""""
import tensorflow as tf

#%% 构图  with graph.as_default():

# 使用新的构建的图
graph = tf.Graph()
with graph.as_default():
    # 此时在这个代码块中，使用的就是新的定义的图graph(相当于把默认图换成了graph)
    d = tf.constant(5.0, name='d')
    print("变量d是否在新图graph中:{}".format(d.graph is graph))

with tf.Graph().as_default() as g2:
    e = tf.constant(6.0)
    print("变量e是否在新图g2中：{}".format(e.graph is g2))

# 这段代码是错误的用法，记住：不能使用两个图中的变量进行操作，只能对同一个图中的变量对象（张量）进行操作(op)
# f = tf.add(d, e)


#%% CPU与GPU     tf.device('/cpu:0') / tf.device('/gpu:0'):表示使用第一个GPU运算，如果有的话
'''
    代码块中定义的操作，会在tf.device给定的设备上运行

    GPU有多个，如GPU:0，GPU:1...，CPU只有一个
    有一些操作，是不会再GPU上运行的（一定要注意）
    如果安装的是tensorflow cpu版本，不可以指定运行环境
'''
with tf.device('/cpu:0'):
    a = tf.Variable([1, 2, 3], dtype=tf.int32, name='name_a')
    b = tf.constant(2, dtype=tf.int32, name='name_b')
    c = tf.add(a, b, name='name_ab')

with tf.device('/gpu:0'):
    d = tf.Variable([2, 8, 13], dtype=tf.int32, name='name_d')
    e = tf.constant(2, dtype=tf.int32, name='name_e')
    f = d + e

g = c + f

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # 初始化
    tf.global_variables_initializer().run()
    print(g.eval())

'''
    输出
    name_d/read: (Identity): /job:localhost/replica:0/task:0/device:CPU:0
    add: (Add): /job:localhost/replica:0/task:0/device:GPU:0
    name_d/Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0
    init/NoOp_1: (NoOp): /job:localhost/replica:0/task:0/device:GPU:0
    name_a: (VariableV2): /job:localhost/replica:0/task:0/device:CPU:0
    name_a/read: (Identity): /job:localhost/replica:0/task:0/device:CPU:0
    name_ab: (Add): /job:localhost/replica:0/task:0/device:CPU:0
    add_1: (Add): /job:localhost/replica:0/task:0/device:GPU:0
    name_a/Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0
    init/NoOp: (NoOp): /job:localhost/replica:0/task:0/device:CPU:0
    init: (NoOp): /job:localhost/replica:0/task:0/device:GPU:0
    name_e: (Const): /job:localhost/replica:0/task:0/device:GPU:0
'''


