"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.

The Cartpole example using distributed tensorflow + multiprocessing.

View more on my tutorial page: https://morvanzhou.github.io/

"""

import multiprocessing as mp   # 异步操作
import tensorflow as tf
import numpy as np
import gym, time
import matplotlib.pyplot as plt

UPDATE_GLOBAL_ITER = 10   # 全局网络更新频率
GAMMA = 0.9
ENTROPY_RATE = 0.001   # learning rate for 熵
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic

env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]   # 状态空间
N_A = env.action_space.n   # 动作空间


class ACNet():
    sess = None

    def __init__(self, scope, opt_a=None, opt_c=None, global_net=None):
        if scope == 'global_net':  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')   # 状态
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')   # 行动者动作
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')   # 目标V值（原文用G）

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)   # 调用网络
                td = tf.subtract(self.v_target, self.v, name='TD_error')

                with tf.name_scope('c_loss'):   # 评论家损失函数
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):   # 行动者目标函数
                    log_prob = tf.reduce_sum(
                        tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)   # logπ
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5), axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_RATE * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            self.global_step = tf.train.get_or_create_global_step()   # 全局步数
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_net.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_net.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = opt_a.apply_gradients(zip(self.a_grads, global_net.a_params), global_step=self.global_step)
                    self.update_c_op = opt_c.apply_gradients(zip(self.c_grads, global_net.c_params))

    '''
        创建网络
    '''
    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)   # 随机数字（这里没有用图片）
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')   # 行动者输出策略概率。softmax处理
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # 评论家输出状态值函数。线性处理

        '''
            tf.get_collection(key,scope)：获取名位‘key’的集合中的所有元素，返回一个列表，列表的顺序是按照变量放入集合中的先后顺序
            1.key：集合
            2.scope：可选参数，表示名称空间（名称域），如果指定，就返回名称空间中所有放入‘key’的变量列表，不指定则返回所有变量
        '''
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def choose_action(self, s):  # run by a local
        prob_weights = self.sess.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})   # np.newaxis：选取部分的数据增加一个维度。
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict=feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])


def work(job_name, task_index, global_ep, lock, r_queue, global_running_r):
    '''
    :param job_name: 工作组名称
    :param task_index: 工作组标签
    :param global_ep:
    :param lock:
    :param r_queue:
    :param global_running_r:
    '''
    # set work's ip:port（端口）   ps是参数服务器的缩写
    cluster = tf.train.ClusterSpec({
        "ps": ['localhost:2220', 'localhost:2221',],
        "worker": ['localhost:2222', 'localhost:2223', 'localhost:2224', 'localhost:2225',]
    })
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)
    if job_name == 'ps':
        print('Start Parameter Sever: ', task_index)
        server.join()
    else:
        t1 = time.time()   # 返回当前时间的时间戳
        env = gym.make('CartPole-v0').unwrapped
        print('Start Worker: ', task_index)
        '''
            -》tf.train.replica_device_setter：用于在同一个参数服务器(PS)(使用循环法)和一个worker上的计算密集型节点上自动分配变量.
            1.ps_tasks: Number of tasks in the ps job. Ignored if cluster is provided.
            2.ps_device: String. Device of the ps job. If empty no ps job is used. Defaults to ps.
            3.worker_device: String. Device of the worker job. If empty no worker job is used.
            4.merge_devices: Boolean. If True, merges or only sets a device if the device constraint is completely unset. merges device specification rather than overriding them.
            5.cluster: ClusterDef proto or ClusterSpec.
        '''
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):
            opt_a = tf.train.RMSPropOptimizer(LR_A, name='opt_a')   # 学习率
            opt_c = tf.train.RMSPropOptimizer(LR_C, name='opt_c')
            global_net = ACNet('global_net')   # 全局网络

        local_net = ACNet('local_ac%d' % task_index, opt_a, opt_c, global_net)   # 本地网络

        hooks = [tf.train.StopAtStepHook(last_step=100000)]   # set training steps
        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=True, hooks=hooks,) as sess:
            print('Start Worker Session: ', task_index)
            local_net.sess = sess
            total_step = 1   # 全局计数器
            buffer_s, buffer_a, buffer_r = [], [], []
            while (not sess.should_stop()) and (global_ep.value < 1000):   # 控制单条情节长度
                s = env.reset()   # 初始化情节
                ep_r = 0   # ep_r用以记录情节回报（无折扣）
                while True:
                    # if task_index:
                    #     env.render()
                    a = local_net.choose_action(s)
                    s_, r, done, info = env.step(a)
                    if done:    # s_为终止状态
                        r = -5.
                    ep_r += r
                    buffer_s.append(s)   # buffer缓冲
                    buffer_a.append(a)
                    buffer_r.append(r)

                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net(更新全局并分配到本地网络)
                        if done:
                            v_s_ = 0  # terminal
                        else:
                            v_s_ = sess.run(local_net.v, feed_dict={local_net.s: s_[np.newaxis, :]})[0, 0]

                        buffer_v_target = []
                        for r in buffer_r[::-1]:  # reverse buffer r
                            v_s_ = r + GAMMA * v_s_   # 目标值
                            buffer_v_target.append(v_s_)   # 目标值list
                        buffer_v_target.reverse()   # 反转顺序

                        buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                        feed_dict = {
                            local_net.s: buffer_s,
                            local_net.a_his: buffer_a,
                            local_net.v_target: buffer_v_target,
                        }
                        local_net.update_global(feed_dict)
                        buffer_s, buffer_a, buffer_r = [], [], []
                        local_net.pull_global()
                    s = s_   # 更新状态。因为没有经验池
                    total_step += 1

                    if done:   # 终止状态
                        if r_queue.empty():  # record running episode reward
                            global_running_r.value = ep_r
                        else:
                            global_running_r.value = .99 * global_running_r.value + 0.01 * ep_r
                        r_queue.put(global_running_r.value)   # -》Put：插入数据到队列中，可传参超时时长；有两个可选参数，blocked 和 timeout。 - blocked = True（默认值），timeout 为正

                        print(
                            "Task: %i" % task_index,
                            "| Ep: %i" % global_ep.value,
                            "| Ep_r: %i" % global_running_r.value,
                            "| Global_step: %i" % sess.run(local_net.global_step),
                        )
                        with lock:
                            global_ep.value += 1
                        break   # 离开循环

        print('Worker Done: ', task_index, time.time()-t1)   # 记录时间


if __name__ == "__main__":
    # use multiprocessing to create a local cluster with 2 parameter servers and 2 workers
    '''
    
    -》multiprocessing.Value(typecode_or_type, *args[, lock])：
    该方法返回从共享内存中分配的一个ctypes对象，类似Value('d', 0.0)即可
    1.typecode_or_type：定义返回类型，如int（i），double（d），char（c），float（f）等，
    2.*args：传递给ctypes的构造参数
    '''
    global_ep = mp.Value('i', 0)   # int
    lock = mp.Lock()   # 异步上锁
    r_queue = mp.Queue()   # 异步队列。-》 Queue 用来在多个进程间通信。Queue 有两个方法，get 和 put。
    global_running_r = mp.Value('d', 0)   # double

    jobs = [
        ('ps', 0), ('ps', 1),
        ('worker', 0), ('worker', 1), ('worker', 2), ('worker', 3)
    ]

    '''
        mp.Process([group [, target [, name [, args [, kwargs]]]]])
        1.group：指定进程组，大多数情况下用不到
        2.target：传递一个函数的引用，可以认为这个子进程就是执行这个函数的代码。这里传的是函数的引用，后面不能有小括号
        3.name：给进程设定一个名字，可以不设定
        4.args：给target指定的函数传递的参数，以元组的方式传递，这里必须是一个元组，只有一个参数时要注意别出错
        5.kwargs：给target指定的函数传递关键字参数，以字典的方式传递，这里必须是一个字典
    '''
    ps = [mp.Process(target=work, args=(j, i, global_ep, lock, r_queue, global_running_r), ) for j, i in jobs]

    '''
        Process的常用方法：
        1.start()：启动子进程实例（创建子进程）
        2.is_alive()：判断子进程是否还在活着
        3.join()：实现进程间的同步，等待所有进程退出
        4.terminate()：不管任务是否完成，立即终止子进程
        5.close()：阻止多余的进程涌入进程池 Pool 造成进程阻塞
    '''
    [p.start() for p in ps]   # 启动全部子进程
    [p.join() for p in ps[2:]]   # 从第三个子进程，也就是工作组worker处，开始执行

    ep_r = []
    while not r_queue.empty():   # 判断队列是否为空
        ep_r.append(r_queue.get())   # get：从队列中读取并删除一个元素。有两个参数可选，blocked 和 timeout 。- blocked = False （默认），timeout 正值
    plt.plot(np.arange(len(ep_r)), ep_r)
    plt.title('Distributed training')
    plt.xlabel('Step')
    plt.ylabel('Total moving reward')
    plt.show()



