"""
Policy Gradient, Reinforcement Learning.
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible   ★
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self._build_net()   # 构建网络

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    """
        -》tf.layers.dense：全连接层
            inputs：网络的输入
            units：输出的维度，改变inputs的最后一维
            activation: 激活函数，默认为None，不使用激活函数
            use_bias: 使用bias为True（默认使用），不用bias改成False即可
            trainable=True:表明该层的参数是否参与训练。如果为真则变量加入到图集合中
            
            kernel_initializer：卷积核的初始化器，默认为None
            bias_initializer：偏置项的初始化器，默认初始化为0，tf.zeros_initializer()
            
            kernel_regularizer：卷积核的正则化，默认为None
            bias_regularizer：偏置项的正则化，默认为None
            activity_regularizer：输出的正则化函数，默认为None，
    """
    def _build_net(self):
        self.observation = tf.placeholder(tf.float32, [None, self.n_features], name="observations")   # 注意形状
        self.actions = tf.placeholder(tf.int32, [None, ], name="actions_num")   # [None,]和[None]好像区别不大
        self.Vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        net = tf.layers.dense(
            inputs=self.observation,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        net = tf.layers.dense(
            inputs=net,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # 通过softmax转换，输出动作概率
        self.all_act_prob = tf.nn.softmax(net, name='act_prob')  # use softmax to convert to probability

        """ to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss) """
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=self.actions)   # this is negative log of chosen action
        # or in this way:
        # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.actions, self.n_actions), axis=1)   # ★

        loss = tf.reduce_mean(neg_log_prob * self.Vt)  # reward guided loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    # 策略
    def choose_action(self, observation):
        # 输出网络结果(softmax结果)
        all_act_prob = self.sess.run(self.all_act_prob, feed_dict={self.observation: observation[np.newaxis, :]})   # 在0维上增加一个维度[  [*]  ]，匹配网络输入数据格式
        # 依据p(softmax结果)的概率分布，从三个动作中随机选择一个 - 既保证了动作选择的随机性，又确保最优动作可以被大概率选中
        action = np.random.choice(range(3), p=all_act_prob.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        # 在一轮游戏中，持续存储所有s,a,r直到done
        self.ep_obs.append(s)
        self.ep_as.append(a)   # (None,)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()   # 以 一次游戏的标准化回报均值 作为该次游戏的 状态值函数

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.observation: np.vstack(self.ep_obs),  # shape=[None, n_observation]  叠加所有状态，一次性输入网络
             self.actions: np.array(self.ep_as),  # shape=[None, ]
             self.Vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data 一轮游戏结束，重置存储
        return discounted_ep_rs_norm   # 返回状态值函数

    def _discount_and_norm_rewards(self):
        # 在一轮游戏中，这些值都是固定的
        # discount episode rewards   # 折扣
        discounted_ep_rs = np.zeros_like(self.ep_rs)   # 以奖 励list形状构建 空折扣奖励list
        running_add = 0
        for t in range(len(self.ep_rs)-1 ,-1, -1):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add   # 存储每一个时刻(状态)的回报

        # normalize episode rewards   标准化
        discounted_ep_rs -= np.mean(discounted_ep_rs)   # 求回报均值
        discounted_ep_rs /= np.std(discounted_ep_rs)   # 标准化处理
        return discounted_ep_rs



