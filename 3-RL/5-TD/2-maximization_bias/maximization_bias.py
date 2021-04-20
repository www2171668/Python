''''''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy


STATE_A = 0   # 起始状态 A
STATE_B = 1   # 起始状态 B
STATE_TERMINAL = 2   # 结束状态（包括了A右和B左两个结束状态）

EPSILON = 0.1
ALPHA = 0.1   # step size
GAMMA = 1.0   # discount for max value

# 定义 A的动作，向左或向右
ACTION_A_RIGHT = 0
ACTION_A_LEFT = 1
# 定义 B的动作, 全部向左，但有10个动作可选
ACTIONS_B = range(0, 10)
ACTIONS = [[ACTION_A_RIGHT, ACTION_A_LEFT], ACTIONS_B]   # all possible actions   [[0, 1], range(0, 10)]

INITIAL_Q = [np.zeros(2), np.zeros(len(ACTIONS_B)), np.zeros(1)]   # 初始化动作值函数Q  [array([ 0.,  0.]), array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]), array([ 0.])]  可以看出到达终止状态的时候，还是有(s,a)存在，(s,a)中存q
TRANSITION = [[STATE_TERMINAL, STATE_B], [STATE_TERMINAL] * len(ACTIONS_B)]   # # 设置状态转换矩阵 [[2, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]  设定从一开始只能往一个方向走到底

# 基于ε-贪婪策略（行为策略）选择动作
def policy(s, Q):
    if np.random.binomial(1, EPSILON) == 1:   # 随机选择动作
        a = np.random.choice(ACTIONS[s])
    else:
        best_a = [a for a, q in enumerate(Q[s]) if q == np.max(Q[s])]
        a = np.random.choice(best_a)
    return a

# take action in state, return the reward
def step(state, action):
    if state == STATE_A:
        return 0
    else:
        return np.random.normal(-0.1, 1)   # 服从正态分布（-0.1,1）的奖励

# if there are two state action pair value array, use double Q-Learning, otherwise use normal Q-Learning
def q_learning(Q1, Q2=None):
    s = STATE_A   # 初始化状态

    left_count = 0   # 只记录从A向左走的次数
    while s != STATE_TERMINAL:
        if Q2 is None:   # Q-learning
            a = policy(s, Q1)
            # print(a)
        else:
            Q_integrate =[item1 + item2 for item1, item2 in zip(Q1, Q2)]   # 依次相加
            a = policy(s, Q_integrate)   # Double Q-learning：derive a action form Q1 and Q2

        if s == STATE_A and a == ACTION_A_LEFT:
            left_count += 1

        reward = step(s, a)
        n_s = TRANSITION[s][a]

        if Q2 is None:
            active_Q = Q1
            max_n_q = np.max(active_Q[n_s])
        else:
            if np.random.binomial(1, 0.5) == 1:
                active_Q = Q1
                target_Q = Q2
            else:
                active_Q = Q2
                target_Q = Q1
            best_a = [a for a, q in enumerate(active_Q[n_s]) if q == np.max(active_Q[n_s])]
            a = np.random.choice(best_a)
            max_n_q = target_Q[n_s][a]

        active_Q[s][a] += ALPHA * (reward + GAMMA * max_n_q - active_Q[s][a])   # 策略评估，更新Q
        s = n_s   # 更新状态

    return left_count   # 返回值 非0即1

def figure_6_7():
    # each independent run has 100 episodes
    episodes = 100
    runs = 1000

    left_counts = np.zeros((runs, episodes))
    # left_counts_Double_Q_Learning = np.zeros((runs, episodes))

    for run in tqdm(range(runs)):   # 用于对每一次迭代求均值
        Q = copy.deepcopy(INITIAL_Q)   # -》copy.deepcopy(*) 深拷贝。 重置Q
        Q1 = copy.deepcopy(INITIAL_Q)
        Q2 = copy.deepcopy(INITIAL_Q)
        for ep in range(0, episodes):   # 迭代开始
            left_counts[run, ep] = q_learning(Q)
            print('*'*10)
            # left_counts_Double_Q_Learning[run, ep] = q_learning(Q1, Q2)

    # 把每一episode中向左移动的次数求均值，以表示出 % left actions from A —— 在episode中从A向左移动的次数/总移动次数。 注意是| | | | 地求均值
    # episode越到后面Q越稳定，智能体越容易往右走，即越不容易出现最大化偏差问题
    # left_counts = left_counts.mean(axis=0)     # len(left_counts_q)=100
    left_counts_Double_Q_Learning = left_counts_Double_Q_Learning.mean(axis=0)

    plt.plot(left_counts, label='Q-Learning')
    # plt.plot(left_counts_Double_Q_Learning, label='Double Q-Learning')
    plt.plot(np.ones(episodes) * 0.05, label='Optimal')
    plt.xlabel('episodes')
    plt.ylabel('% left actions from A')
    plt.legend()

    plt.savefig('figure_6_7.png')
    plt.close()

if __name__ == '__main__':
    figure_6_7()
