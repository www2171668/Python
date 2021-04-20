#coding=utf-8

import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import plotting

env = gym.envs.make("MountainCar-v0")
# env = env.unwrapped   # 取消对环境的监控机制

# print(env.action_space)   # 可能的动作   Discrete(3)
# print(env.observation_space)   # 可能的搜索状态   Box(2,)
# print(env.observation_space.high)   # [ 0.60000002  0.07      ]
# print(env.observation_space.low)   # [-1.20000005 -0.07      ]

observation_examples = np.array([env.observation_space.sample() for x in range(10000)])   # 采集10000条状态数据
scaler = preprocessing.StandardScaler()
scaler.fit_transform(observation_examples)   # 机器学习的方法 —— 模型学习(标准化预处理)并转换

"""
    FeatureUnion把若干个transformer objects组合成一个新的transformer，这个新的transformer组合了他们的输出，一个FeatureUnion对象接受一个transformer对象列表
    
    -》pipeline.FeatureUnion([])：合并多个转换器对象形成一个新的转换器，该转换器合并了他们的输出。
    对于转换数据，转换器可以并发使用，且输出的样本向量被连接成更大的向量。

    FeatureUnion 功能与 Pipeline 一样
    FeatureUnion通过一系列 (key, value) 键值对来构建的,其中的 key 给转换器指定的名字，value是一个评估器对象
"""
# 用于对状态进行特征抽取。使用不同的径向基函数覆盖状态空间的不同部分，通过featurizer.fit()学习经过归一化操作后的状态值
featurizer = pipeline.FeatureUnion([   # 管道
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),   # 使用RBF核函数(径向基函数)进行特征转换 —— 在QDN中换成为了神经网络
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(observation_examples)   # 及其悬系的方法 —— 模型训练

"""使用动作值函数近似预测Q函数"""
class Estimator():
    def __init__(self):
        self.action_models=[]
        for _ in range(env.action_space.n):   # 遍历所有动作。对于每个状态，都会把经过特征处理的状态值分配给各自对应的动作模型中
            """
                -》fit和partial_fit：在第一次训练时，fit和partial_fit训练模型的原理本质上是一样的
                但是fit就是一次性把模型训练完毕，而对于partial_fit的模型来说，新的数据可以在原模型基础上继续训练、更新模型，而不必重新训练
                
                partial_fit只针对onlin的算法，SGD就是online的
            """
            # 机器学习的方法 —— 模型训练 ([X_train, Y_train])，因为做了归一化，所以用[0]来作为标签
            model = SGDRegressor(learning_rate="constant")   # SGD回归
            model.partial_fit([self.feature_state(env.reset())],[0])   # 传入初始化的状态，形如[-0.52369602  0.        ]。

            self.action_models.append(model)   # 动作模型。在之后的模型训练中，设定不同的动作对应不同的模型[0,1,2]

    # 状态特征抽取函数
    def feature_state(self,s):
        # 先对当前状态进行标准化转化（所有状态都会经历这一步），然后对转化后的当前状态进行特征化处理
        # 由于featurizer.transform(scaler.transform([s]))是一个[1，None]的数据，所以用[0]来降维，得到[None, ]数据作为样本数据
        return featurizer.transform(scaler.transform([s]))[0]

    def predict_func(self,s,a=None):
        s=self.feature_state(s)   # 数据预处理（特征提取）
        if a:
            return self.action_models[a].predict([s])[0]
        else:
            return [self.action_models[m].predict([s])[0] for m in range(env.action_space.n)]   # 分别用3个模型对状态s进行预测，将结果值(状态值函数)存入列表

    def update(self,s,a,target):
        s=self.feature_state(s)   # 数据预处理（特征提取）
        self.action_models[a].partial_fit([s],[target])   # 机器学习的方法 —— 模型训练，根据目标[target]和当前的状态特征，更新对应的模型

def make_epsilon_greedy_policy(estimator,nA,epsilon):

    def epsilon_greedy_policy(observation):

        best_action = np.argmax(estimator.predict_func(observation))   # 贪婪策略，取最大状态值函数对应的动作
        A =np.ones(nA,dtype=np.float32)*epsilon/nA   # 平均概率
        A[best_action] += 1-epsilon   # 设定ε-贪婪策略
        return A

    return epsilon_greedy_policy

"""根据ε-贪婪策略找到能使小车更快到达终点的最优策略"""
def Q_learning_with_value_approximation(env,estimator,epoch_num
                                        ,discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):

    # stats = plotting.EpisodeStats(
    #     episode_lengths=np.zeros(epoch_num),
    #     episode_rewards=np.zeros(epoch_num))
    for i_epoch_num in range(epoch_num):   # epoch_num 总迭代次数
        decay_Parameter = epsilon*epsilon_decay**i_epoch_num
        policy = make_epsilon_greedy_policy(estimator, env.action_space.n, decay_Parameter)   # 得到基于ε-贪婪策略的策略
        state = env.reset()

        for it in itertools.count():   # 开始一轮游戏

            action_probs = policy(state)   # 获得基于ε-贪婪策略得到的动作空间概率
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)   # 以ε-贪婪策略选择动作

            env.render()    # 渲染环境
            next_state,reward,done,_=env.step(action)   # 至此得到SARSA
            q_values_next = estimator.predict_func(next_state)   # 根据时间差分更新方法，预测下一时间步的动作值
            td_target = reward + discount_factor * np.max(q_values_next)   # 使用时间差分目标作为预测结果更新函数近似器   ★★★
            estimator.update(state, action, td_target)

            # 更新统计信息（奖励 和 迭代次数）
            # stats.episode_rewards[i_epoch_num] += reward
            # stats.episode_lengths[i_epoch_num] = it
            print("\rStep {} @ Episode {}/{}".format(it, i_epoch_num + 1, epoch_num))

            if done:
                print(it)
                break
            state = next_state


estimator=Estimator()
Q_learning_with_value_approximation(env, estimator, 100, epsilon=0.0)
plotting.plot_cost_to_go_mountain_car(env, estimator)
env.close()









