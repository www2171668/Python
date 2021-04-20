from __future__ import print_function
from __future__ import division

import numpy as np
import random

from utils import draw_grid, draw_episode_steps


class TDAgent():
    def __init__(self, env, epsilon, gamma, alpha=0.05, lamda=0.7):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon  # explore & exploit
        self.init_episilon = epsilon
        self.lamda = lamda

        self.V = np.zeros(self.env.num_s)   # 状态值函数矩阵 - 没用上
        self.P = np.zeros((self.env.num_s, self.env.num_a))   # 策略，以数组方式存储π(s,a)
        self.Q = np.zeros((self.env.num_s, self.env.num_a))   # 一开始Q 和E 都是0
        self.E = np.zeros((self.env.num_s, self.env.num_a))   # Eligibility Trace（适合度轨迹）

        self.step_set = []  # store steps of each episode
        self.avg_step_set = []  # store average steps of each 100 episodes
        self.episode = 0
        self.step = 0
        self.max_episodes = 5000

        # initialize random policy  在异策略中这是行为策略b
        for s in range(self.env.num_s):   # 遍历状态位置(pos)
            poss_action = self.env.allow_actions(s)   # poss_action表示在状态s下可执行动作数组
            for a in poss_action:   # 遍历所有可执行动作
                self.P[s][a] = 1.0 / len(poss_action)   # 以均值初始化策略

    def predict(self, episode=1000):   # TD-Prediction —— TD预测方法。但是没用上
        for e in range(episode):   #
            curr_s = self.env.reset()  # new episode
            while not self.env.is_terminal(curr_s):  # for every time step
                a = self.select_action(curr_s, policy='greedy')
                r = self.env.rewards(curr_s, a)
                next_s = self.env.next_state(curr_s, a)
                self.V[curr_s] += self.alpha * (r+self.gamma*self.V[next_s] - self.V[curr_s])   # TD(0)预测
                curr_s = next_s   # 更新状态
        # result display
        draw_grid(self.env, self, p=True, v=True, r=True)

    """选择算法，进行迭代"""
    def control(self, method):
        assert method == 'qlearn' or method == 'sarsa'   # -》assert：检查条件，不符合就终止程序

        # ①、选择要执行的算法
        if method == 'qlearn':
            agent = Qlearn(self.env, self.epsilon, self.gamma)
        elif method == 'sarsa':
            agent = SARSA(self.env, self.epsilon, self.gamma)

        # ②、开始迭代
        while agent.episode < self.max_episodes:
            agent_action = agent.act()   #
            agent.learn(agent_action)

        # ①、resutl display
        draw_grid(self.env, agent, p=True, v=True, r=True)
        # ②、draw episode steps
        draw_episode_steps(agent.avg_step_set)

    """基于动作值函数，更新策略"""
    def update_policy(self):
        poss_action = self.env.allow_actions(self.curr_s)
        q_s_pa = self.Q[self.curr_s][poss_action]   # poss_action是一个列表，所以这个操作是一个多值索引，返回是一个列表
        q_maxs = [q for q in q_s_pa if q == max(q_s_pa)]   # 获得最大动作值函数列表
        # q_maxs = [np.max(self.Q[self.curr_s][poss_action])]

        # update probs
        for i, a in enumerate(poss_action):
            self.P[self.curr_s][a] = 1.0 / len(q_maxs) if q_s_pa[i] in q_maxs else 0.0   # 根据Q值更新策略。存在多个最大动作值函数时，策略的以均值更新

    """选择动作"""
    def select_action(self, state, policy='egreedy'):
        poss_action = self.env.allow_actions(state)  # possiable actions 形如[0,1,2,3]
        if policy == 'egreedy' and random.random() < self.epsilon:   # 基于ε-贪婪策略，以ε的概率进行随机选择
            a = random.choice(poss_action)   # 随机选择一个动作
        else:  # greedy action 选择最优动作
            pros = self.P[state][poss_action]  # probobilities for possiable actions 形如[0.25,0.25,0.25,0.25]
            best_a_idx = [i for i, p in enumerate(pros) if p == max(pros)]   # 遍历每个策略，输出最优动作的id集合
            a = poss_action[random.choice(best_a_idx)]   # 在最优动作的id集合中随机选出最优动作
        return a


class SARSA(TDAgent):
    def __init__(self, env, epsilon, gamma):
        super(SARSA, self).__init__(env, epsilon, gamma)
        # 初始化状态S和其他参数
        self.reset_episode()   # 实际操作时，由于构造函数中调用了reset_episode()，使得episode = 1为起始episode值而非0

    def act(self):   # TD(0)
        r = self.env.rewards(self.curr_s, self.curr_a)   # 返回当前状态的奖励。体现的是先拿奖励再进入下一个状态
        r -= 0.01  # a bit negative reward for every step   每走一步都会有一点消极的回报，使最后获得最短路径
        s = self.env.next_state(self.curr_s, self.curr_a)   # 获得下一个状态
        a = self.select_action(s, policy='egreedy')   # 基于ε-贪婪策略选择下一状态的动作
        return [self.curr_s, self.curr_a, r, s, a]   # 得到Sarsa

    def learn(self, exp):
        s, a, r, n_s, n_a = exp
        if self.env.is_terminal(s):   # 到达终点
            target = r   # 没有下一个状态了，只获得即时奖励
        else:
            target = r + self.gamma * self.Q[n_s][n_a]   # TD(0)计算公式
        error = target - self.Q[s][a]   # δt

        # SARSA(λ)
        self.E[s][a] += 1.0   # 对当前(s,a)自增1
        for _s in range(self.env.num_s):   # 遍历所有(s,a)
            for _a in range(self.env.num_a):
                self.Q[_s][_a] += self.alpha * error * self.E[_s][_a]   # SARSA(λ)更新公式
                self.E[_s][_a] *= self.lamda * self.gamma   # 对E更新

        # update policy
        self.update_policy()

        if self.env.is_terminal(s):   # 一次迭代结束后
            # self.V = np.sum(self.Q, axis=1)
            print('episode %d step: %d epsilon: %f' % (self.episode, self.step, self.epsilon))
            # print(self.E)

            self.reset_episode()   # 重置环境，并使episode+1
            self.E = np.zeros((self.env.num_s, self.env.num_a))   # 重置E
            self.epsilon -= self.init_episilon / 10000   # 随着迭代次数的增加，epsilon减少，这样使得ε-贪婪策略的贪婪程度增大，更容易取得最大值

            if self.episode % 100 == 0:   # record per 100 episode
                self.avg_step_set.append(np.sum(self.step_set[self.episode-100: self.episode])/100)
        else:  # shift state-action pair
            self.curr_s = n_s   # 更新状态
            self.curr_a = n_a   # 更细动作
            self.step += 1   # 一次游戏中的步数
            # 用的是实例属性，所以没有return

    def reset_episode(self):
        # start a new episode
        self.curr_s = self.env.reset()   # 返回起点状态
        self.curr_a = self.select_action(self.curr_s, policy='egreedy')
        self.episode += 1   # 进入下一次迭代(游戏)
        self.step_set.append(self.step)   # 存储此次迭代（游戏）的步数
        self.step = 0   # 一次游戏步数归零


class Qlearn(TDAgent):
    def __init__(self, env, epsilon, gamma):
        super(Qlearn, self).__init__(env, epsilon, gamma)
        self.reset_episode()

    def act(self):   # Qlearn没有E
        a = self.select_action(self.curr_s, policy='egreedy')
        s = self.env.next_state(self.curr_s, a)
        r = self.env.rewards(self.curr_s, a)
        r -= 0.01
        return [self.curr_s, a, r, s]

    def learn(self, exp):
        s, a, r, n_s = exp

        if self.env.is_terminal(s):
            target = r
        else:
            target = r + self.gamma * max(self.Q[n_s])   # 目标值，Off-policy使用的是max()
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])   # 核心公式

        self.update_policy()

        if self.env.is_terminal(s):
            self.V = np.sum(self.Q, axis=1)
            print('episode %d step: %d' % (self.episode, self.step))

            self.reset_episode()
            self.epsilon -= self.init_episilon / self.max_episodes
            if self.episode % 100 == 0:
                self.avg_step_set.append(np.sum(self.step_set[self.episode-100: self.episode])/100)
        else:
            self.curr_s = n_s   # 更新状态
            self.step += 1

    def reset_episode(self):
        self.curr_s = self.env.reset()
        self.episode += 1
        self.step_set.append(self.step)
        self.step = 0
