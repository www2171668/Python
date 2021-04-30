""""""
import tensorflow as tf

# %% tf.constant(数据，dtype，name)：声明tensor常量（不能直接用print输出结果的）  张量数据格式都是tf.
a = tf.constant([[1, 2], [3, 4]], dtype=tf.int32, name='a')  # * name属于op_name（操作名称）
b = tf.constant([5, 6, 7, 8], dtype=tf.int32, shape=[2, 2], name='b')  # \ sess run的输出结果是[[5 6] [7 8]]

# %% tensor计算    tf.multiply()： /  tf.matmul()：矩阵点乘和乘法        tf.add()： / tf.subtract()：加减法        tf.reduce_max(*,axis)：0表示遍历列，1表示遍历行
c = tf.matmul(a, b, name='matmul')  # 构建操作。在tf中，a和b的dtype的类型必须一模一样才可以进行运算  —— 这是一个tensor的张量
g = tf.add(a, c, name='add')
h = tf.subtract(b, a, name='b-a')

x = tf.constant([[1, 0, 3], [2, 1, 1]])
y = tf.reduce_max(x, 1)  # [3,2]
z = tf.reduce_sum(x, 1)  # [4 4]

# %% 会话创建方法一:     with tf.Session() as sess:
'''
    Session属性：config=tf.ConfigProto()的参数：
        -》allow_soft_placement：默认为False，不允许动态使用CPU和GPU；当安装方式为GPU时，建议设置为True，因为TensorFlow中的部分op只能在CPU上运行
        -》log_device_placement：默认为False，不打印日志

    graph：给定当前Session对应的图，默认为默认图；
    
    -》sess.run(*)和*.eval()：获取op值(只能接受一个值 - 单值或list\dict等)，返回numpy.ndarray数组值。相当于拿出这个值  ★★
    -》sess.run(fetches = [*])：获取op值，返回numpy.ndarray数组的list集合  ★★
'''
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # * 通过Session的run方法获取张量  ★
    print("sess run:{}".format(sess.run(b)))  # \[[5 6] [7 8]]     print("b eval:{}".format(b.eval()))
    print("sess run:{}".format(sess.run([c, g])))  # 输出两个值 。 或用 fetches=[c, r]

    print(sess.run(y))
    print(sess.run(z))

# %% 会话创建方法二：    tf.Session()：     sess.close() 会话关闭，也可以不加
sess2 = tf.Session()  # \构建并启动会话
print("value:\n{}".format(sess2.run([c, g])))
sess2.close()

# %% 变量型张量 tf.Variable(initial_value,dtype,trainable,validate_shape)
'''
    initial_value：初始数据。为[]时表示空值，使其为不固定形状的变量
    trainable=False / validate_shape=False：只有在定义一个不固定形状的变量时，才改为False ★
    tf.truncated_normal(shape, stddev = 0.01)：高斯初始化
'''
w1 = tf.Variable(initial_value=3.0, dtype=tf.float32)
b = tf.constant(value=2.0, dtype=tf.float32)
c = tf.add(w1, b)
w2 = tf.Variable(w1.initialized_value() * b, name='w2')  # * 存在变量依赖的变量的初始化需要加上initialized_value() ★

init_op = tf.global_variables_initializer()  # * tf.global_variables_initializer()：变量op的初始化

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    # * 在所有操作运行之前，运行init op进行变量初始化
    sess.run(init_op)
    print("result:{}".format(sess.run(c)))  # \result:5.0

# %% 占位符型张量:先运算，再赋值   tf.placeholder()：默认格式为张量      fetches：输出变量     feed_dict：字典型变量,用于在sesson.run()中对占位符张量赋值，并对run()中函数进行输出
# 案例1
m1 = tf.placeholder(dtype=tf.float32, shape=[2, 3], name='placeholder_1')
m2 = tf.placeholder(dtype=tf.float32, shape=[3, 2], name='placeholder_2')
m3 = tf.matmul(m1, m2)  # \ 先做运算

with tf.Session() as sess:
    print(sess.run(fetches=m3, feed_dict={m1: [[1, 2, 3], [4, 5, 6]], m2: [[9, 8], [7, 6], [5, 4]]}))  # * fetches     feed_dict

# 案例2
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))

# %% 变量更新  tf.assign(ref,value) / ref.assign(value)：所有变量的更新都必须通过assign      ref：待更新的变量 / value：更新操作
# 1. 定义变量
x = tf.Variable(0, dtype=tf.int32, name='v_x')
y = tf.Variable(0, dtype=tf.int32, name='v_y')

# 2. 更新变量
assign_op = tf.assign(ref=x, value=x + 1)  # 等价于tf.add(x,tf.constant(1))，在效果上是x=x+1     也可以写为 assign_op = x.assign(x + 1)

# 3. 变量初始化操作
init_op = tf.global_variables_initializer()

# 4. 运行
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(init_op)

    for i in range(5):
        # 方法一：更新器定义在运行前，在运行中执行
        sess.run(assign_op)  # \ 运行变量更新操作，x得到了更新。
        print(sess.run(x))  # \ 在更新之后获取x，即先+1，再获取x 1 2 3 4 5 6

        # 方法二：更新器定义和执行都在运行中进行
        # sess.run(y.assign(y + 1))
        # print(sess.run(y))  # \ 输出 1 2 3 4 5 6

'''
    ①上半部分
    v_x: (VariableV2): /job:localhost/replica:0/task:0/device:CPU:0

    v_x/read: (Identity): /job:localhost/replica:0/task:0/device:CPU:0
    add: (Add): /job:localhost/replica:0/task:0/device:GPU:0
    Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0
    v_x/Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0

    ②、Session部分
    init: (NoOp): /job:localhost/replica:0/task:0/device:GPU:0
    add/y: (Const): /job:localhost/replica:0/task:0/device:GPU:0
    v_x/initial_value: (Const): /job:localhost/replica:0/task:0/device:CPU:0
'''

# %% 不定型张量
# 1. 定义一个不定形状的变量
x = tf.Variable(
    initial_value=[],  # \给定一个空值，保证其为不固定形状的变量
    dtype=tf.float32,
    trainable=False,  # \只有在定义一个不固定形状的变量时，才改为False ★
    validate_shape=False  # \只有在定义一个不固定形状的变量时，才改为False
)

# 2. 变量更改
concat = tf.concat([x, [0.0, 0.0]], axis=0)  # * tf.concat():定义联合矩阵 axis=0按列遍历，横向叠加
assign_op = tf.assign(x, concat, validate_shape=False)  # \只有在定义一个不固定形状的变量时，才改为False

# 3. 变量初始化操作
x_init_op = tf.global_variables_initializer()

# 4. 运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    sess.run(x_init_op)

    for i in range(5):
        sess.run(assign_op)  # 执行更新操作
        print(sess.run(x))
# %% 案例：求阶乘
# 1. 定义一个变量和占位符
sum = tf.Variable(1, dtype=tf.int32)
i = tf.placeholder(dtype=tf.int32)

# 2. 更新操作
update_sum = sum * i
assign_op = tf.assign(sum, update_sum)

# 3. 变量初始化操作
init_op = tf.global_variables_initializer()

# 4. 运行
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    sess.run(init_op)

    for j in range(1, 6):
        sess.run(assign_op, feed_dict={i: j})  # 每次把j值赋予i，更新tep_sum
    print("5!={}".format(sess.run(sum)))
