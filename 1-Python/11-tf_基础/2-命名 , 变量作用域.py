""""""
import tensorflow as tf

# %% 传统变量的定义    tf.Variable必须使用name=“”的形式，否则其名称为Variable
def my_func(x):
    w1 = tf.Variable(tf.random_normal([1]))[0]  # tf.random_normal([1]) 随机正态一维向量.使用向量时，用[0]来提取数
    b1 = tf.Variable(tf.random_normal([1]))[0]
    r1 = w1 * x + b1  # \ 输入层到隐含层

    w2 = tf.Variable(tf.random_normal([1]))[0]
    b2 = tf.Variable(tf.random_normal([1]))[0]
    r2 = w2 * r1 + b2  # \ 隐含层到输出层

    return r1, w1, b1, r2, w2, b2  # \输出当前层的结果+权重+偏置项

x = tf.constant(3, dtype=tf.float32)  # \ 输入值
r = my_func(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('result{}'.format(sess.run(r)))

# * 不带[0]：result(array([-2.9568975], dtype=float32), array([-0.851817], dtype=float32), array([-0.4014466], dtype=float32), array([-4.6082], dtype=float32), array([1.5168219], dtype=float32), array([-0.12311313], dtype=float32))
# * 带[0]：result(2.6382523, 0.64382935, 0.70676434, -0.1439712, 0.43426552, -1.2896732)

# %% 共享命名   tf.get_variable(name，shape，initializer)：通过给定名字创建或者返回一个对应变量
'''
    变量作用域使得变量在多次使用时，只使用一个变量值
    name和shape必不可少,name也常称为scope
    initializer：初始化器，定义该变量的值
        -》tf.constant_initializer(value) 初始化为给定的常数值value
        -》tf.random_uniform_initializer(a, b) 初始化为从a到b的均匀分布的随机值
        -》tf.random_normal_initializer(mean, stddev) 初始化为均值为mean、方差为stddev的服从高斯分布的随机值
    
        -》tf.orthogonal_initializer(gini=1.0) 初始化一个正交矩阵，gini参数作用是最终返回的矩阵是随机矩阵乘以gini的结果
        -》tf.identity_initializer(gini=1.0) 初始化一个单位矩阵，gini参数作用是最终返回的矩阵是随机矩阵乘以gini的结果
'''

# * tf.Variable输出的是值，如0.112，或向量[0.112]，如果是向量，可以用[0]来提取数
# * tf.get_variable输出的是值+类型定义，如[array([ 0.11187882], dtype=float32)]，需要用[0]来输出值，这和上面的[0]作用不同  ★★
def net(x):
    w = tf.get_variable(name='w', shape=[1], initializer=tf.random_normal_initializer())[0]  # * 用[0]来提取值
    # tf.get_variable()输出[array([ 0.10352619], dtype=float32)]。使用[0]后输出array([ 0.10352619])

    b = tf.get_variable(name='b', shape=[1], initializer=tf.random_normal_initializer())[0]
    r = w * x + b

    return r, w, b

# %% 变量作用域  with tf.variable_scope(name,initializer,reuse):定义变量作用域
'''
    initializer定义在tf.variable_scope上时，tf.variable_scope内tf.get_variable变量(未定义变量)会遵循该值
    
    reuse：决定采用何种方式来获取变量
        -》True:作用域是为重用变量所设置的，此时要求对应的变量必须存在，否则报错
        -》tf.AUTO_REUSE:如果变量存在就重用变量，如果不存在就创建新变量返回

    定义在variable_scope作用域中的变量和操作，会将variable_scope的名称作为前缀添加到 变量/操作名称 前
    tf使用'变量作用域/变量名称:0'的方式标记变量，如op2/b: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
    嵌套时为 func1/op1/a,有重名变量时，后接:0以区分
'''

def func(x):
    with tf.variable_scope('op1', reuse=tf.AUTO_REUSE):  # \定义变量作用域op1，运行输入层到隐含层
        r1 = net(x)
    with tf.variable_scope('op2', reuse=tf.AUTO_REUSE):  # \op2，运行隐含层到输出层
        r2 = net(r1[0])
    return r1, r2

x1 = tf.constant(3, dtype=tf.float32, name='x1')
r1 = func(x1)

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    print('result{}'.format(sess.run([r1])))

# * result[((-0.49506742, -0.22130431, 0.16884552), (-0.31943765, -0.56254494, -0.5979353))]

# %% 嵌套作用域  tf.name_scope(*)：封装变量，通常用在with tf.variable_scope()的作用域下
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    with tf.variable_scope('variable_scope1', initializer=tf.constant_initializer(4.0)) as name_scope1:
        with tf.name_scope('name_scope1'):
            v1 = tf.Variable(1.0, name="v1")
            w1 = tf.get_variable("w1", [1])  # \name_scope对其不起作用
            l = v1 + w1

            with tf.variable_scope(name_scope1):  # \ 使用已存在的作用域
                h = tf.get_variable('h', [1])
                g = v1 + w1 + l + h

    with tf.variable_scope('variable_scope2', initializer=tf.constant_initializer(2.0)):
        with tf.name_scope('name_scope2'):
            v2 = tf.Variable(1.0, "v2")
            w2 = tf.get_variable("w2", [1])[0]
            l2 = v2 + w2

    sess.run(tf.global_variables_initializer())
    print("{},{}".format(v1.name, v1.eval()))  # \ variable_scope1/name_scope1/v1:0,1.0  输出中的v1是op_name属性
    print(w1)  # \ <tf.Variable 'variable_scope1/w1:0' shape=(1,) dtype=float32_ref>  注意shape=(1,) dtype=float32_ref  <---①  ★
    # print("{},{}".format(w1.name, w1.eval()))  # \ variable_scope1/w1:0,[ 4.]
    # print("{},{}".format(l.name, l.eval()))  # \ variable_scope1/name_scope1/add:0,[ 5.]
    # print("{},{}".format(h.name, h.eval()))  # \ variable_scope1/h:0,[ 4.]
    # print("{},{}".format(g.name, g.eval()))  # \ variable_scope1/name_scope1/variable_scope1/add_2:0,[ 14.]   foo/bar是已有前缀，foo是当前前缀，add_2表示该变量作用域到目前为止，有两个+操作
    #
    # print("{},{}".format(v2.name, v2.eval()))  # \ variable_scope2/name_scope2/Variable:0,1.0
    # print(w2)  # \ Tensor("variable_scope2/name_scope2/strided_slice:0", shape=(), dtype=float32)  注意shape=(), dtype=float32  <---②
    # print("{},{}".format(w2.name, w2.eval()))  # \ variable_scope2/name_scope2/strided_slice:0,2.0  <---②
    # print("{},{}".format(l2.name, l2.eval()))  # \ variable_scope2/name_scope2/add:0,3.0
    #
