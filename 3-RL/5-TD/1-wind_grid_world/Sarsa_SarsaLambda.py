# #有风格子世界

from random import random, choice
from Env import Environment


class Agent():
    def __init__(self):
        self.Q = {}  # {state:{action:q_value}} 初始化Q ★
        self.E = {}  # {state:{action:E_value}} 初始化Eligibility Trace（适合度轨迹），用于Sarsa(λ) ★
        self.actions = []  # 四个动作

    def sarsaLearning(self, env, gamma, alpha, max_episode_num):
        """"""
        """初始化Q"""
        env.loadActions(self)  # 加载动作，获得动作名称列表['left','rignt',...]
        env.initValue(self)  # 初始化Q和E

        num_episode = 1
        while num_episode <= max_episode_num:  # 重新开始，num_episode总迭代次数，max_episode_num最大迭代次数
            """初始化状态S和策略π和动作A"""
            s, _ = env.reset()  # 环境重置，回到起点 —— 获得初始状态，得到状态名称'X0-Y3'。 s是statename
            a = self.Policy(s, num_episode)  # 得到当前状态下的动作(最优或随机) - state(action_name)

            """一次游戏的开始"""
            time_in_episode = 0
            while not env.is_done():  # 判断是否到达终点
                n_s, n_r = env.step(a)  # 获得下一个位置和奖励

                # 初始化Q；如果Q中里没有state，则对其初始化（一开始Q表并没有初始化，在遍历state的时候再进行初始化）
                if not self.__isStateInQ(n_s):
                    self.initValue(n_s, randomized=True)  # 初始化Q[s][a]

                old_q = self.__getQValue(s, a)  # 得到Q(s,a)
                n_a = self.Policy(n_s, num_episode)  # 得到a'；num_episode用来控制ε大小
                q_next = self.__getQValue(n_s, n_a)  # 得到Q(s',a')

                target = n_r + gamma * q_next  # Ra + 折减系数*Q(s',a')
                # alpha = alpha / num_episode   # 可以通过这种方式设置动态alpha
                new_q = old_q + alpha * (target - old_q)  # 更新Q
                self.__setQValue(s, a, new_q)  # 存入Q表，覆盖旧Q(s,a)

                if num_episode == max_episode_num:
                    print("T:{0:<2}: S:{1}, A:{2:10}, S':{3}".format(time_in_episode, s, a, n_s))

                s = n_s  # 更新当前state
                a = n_a  # 更新当前action
                time_in_episode += 1
            print("Episode {0} takes time {1}".format(num_episode, time_in_episode))

            num_episode += 1
        return

    def sarsaLambdaLearning(self, env, lambda_, gamma, alpha, max_episode_num):
        env.loadActions(self)
        env.initValue(self)

        num_episode = 1
        while num_episode <= max_episode_num:
            s, _ = env.reset()
            a = self.Policy(s, num_episode)
            self.__resetEValue()  # 比SARSA(0)多了要初始化E

            time_in_episode = 0
            while not env.is_done():
                n_s, n_r = env.step(a)  # 进入一下个状态

                if not self.__isStateInQ(n_s):
                    self.initValue(n_s, randomized=True)

                n_a = self.Policy(n_s, num_episode)
                q = self.__getQValue(s, a)  # 取Q(s,a)
                q_next = self.__getQValue(n_s, n_a)
                delta = n_r + gamma * q_next - q

                e = self.__getEValue(s, a)  # 得到E值
                e = e + 1  # 一次游戏中，每经历一个(s,a)，就为这个(s,a)下的E(s,a) 增加1
                self.__setEValue(s, a, e)  # 存下当前(s,a)的E。因为走一步就会有一个E，所以要先把该(s,a)存入，后面的遍历才可以访问到该(s,a)

                # SARAS(λ)，遍历所有经历过的(s,a)
                # 回溯之前的Q。不讲究顺序，因为在一次游戏中，每前进一次，所有存在过的(s,a)都会被λ处理一次，出现的越早的(s,a)就会被alpha*e_value乘的越多，有值状态的更新就会表的越微小
                for s, action_e_dic in list(zip(self.E.keys(), self.E.values())):  # 遍历所有state
                    for action_name, e_value in list(
                            zip(action_e_dic.keys(), action_e_dic.values())):  # 遍历每个state下的action
                        old_q = self.__getQValue(s, action_name)
                        # alpha = alpha / num_episode
                        new_q = old_q + alpha * delta * e_value  # alpha 步长
                        new_e = gamma * lambda_ * e_value
                        self.__setQValue(s, action_name, new_q)  # 更新历史(s,a)的Q
                        self.__setEValue(s, action_name, new_e)  # 更新

                if num_episode == max_episode_num:
                    print("T:{0:<2}: S:{1}, A:{2:10}, S':{3}".format(time_in_episode, s, a, n_s))
                s = n_s
                a = n_a
                time_in_episode += 1

            print("Episode {0} takes time {1}".format(num_episode, time_in_episode))
            num_episode += 1

        return

    # return a possible action list for a given state
    # def possibleActionsForstate(self, state):
    #  actions = []
    #  # add your code here
    #  return actions

    # if a state exists in Q dictionary
    def __isStateInQ(self, state):
        # 判断空值。有值则返回值，无值则返回None - None is not None = Fasle
        return self.Q.get(state) is not None  # 因为是实例属性，所以要用self.进行引用

    def initValue(self, s, randomized=True):  # 初始化Q和E
        # Q[s]为空值时进入判断
        if not self.__isStateInQ(s):
            self.Q[s] = {}  # 初始化Q
            self.E[s] = {}  # 初始化E
            for a in self.actions:  # 遍历所有action_name
                self.Q[s][a] = random() / 10 if randomized is True else 0.0  # 初始化Q(s,a)；随机一个动作值函数。只有结束状态的Q(s,a) = 0
                self.E[s][a] = 0  # 起始E值都是0

    # 基于ε-柔性策略选择动作
    def Policy(self, s, episode_num):
        epsilon = 1.00 / episode_num  # 随着迭代次数增加，epsilon会越来越小
        rand_value = random()

        if rand_value > epsilon:
            a = max(self.Q[s], key=self.Q[s].get)  # 获取最大value值对应的key，即得到最大动作值函数对应的动作
        else:
            a = choice(list(self.actions))  # 随机选择动作
        return a

    """Q与E的获取与设置方法"""

    def __getQValue(self, s, a):  # ①
        return self.Q[s][a]  # argmax(q)

    def __setQValue(self, s, a, new_q):  # ②
        self.Q[s][a] = new_q

    def __getEValue(self, s, a):  # ①
        return self.E[s][a]

    def __setEValue(self, s, a, new_q):  # ②
        self.E[s][a] = new_q

    def __resetEValue(self):  # ③、重置E
        for action_Evalue in self.E.values():  # E是{*:{*:*}，...}结构
            for action in action_Evalue.keys():
                action_Evalue[action] = 0.00


# def sarsaLearningExample(agent, env):  # 参数初始化
#     # agent是类的实例化，使用agent.sarsaLearning来调用类中的方法 — sarsaLearning(self, env, gamma, alpha, max_episode_num):
#     agent.sarsaLearning(env=env, gamma=0.9, alpha=0.1, max_episode_num=1000)


def sarsaLambdaLearningExample(agent, env):
    agent.sarsaLambdaLearning(env=env, lambda_=0.1, gamma=0.9, alpha=0.1, max_episode_num=1000)  # alpha是λ


if __name__ == "__main__":
    # agent和windy_grid_env是全局变量
    agent = Agent()
    env = Environment()  # 引入环境env

    print("Learning...")
    # sarsaLearningExample(agent, env)   # 注意导入的都是类的实例化
    sarsaLambdaLearningExample(agent, env)
