import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

def create_random_policy(nA):
    """
    具有探索性

    Returns:
        A function that takes a state as input and returns a vector of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA   # 初始化动作概率 [0.5，0.5]
    def policy_fn():   # 有的程序这里写了observation，但其实和状态没关系
        return A   # 输出一个与动作空间大小相同的全1数组。在后续的随机选择策略时，会随机选择全1数组中的某一个
    return policy_fn

def create_greedy_policy(Q):
    """
    基于动作值函数使用贪婪策略，具备利用思想

    Returns:
        A function that takes an observation as input and returns a vector of action probabilities.
    """
    def policy_fn(s):
        A = np.zeros_like(Q[s], dtype=float)
        best_a = np.argmax(Q[s])
        A[best_a] = 1.0   # 贪婪策略
        return A   # [0,1] 或 [1,0]
    return policy_fn

"""异策略-加权重要性采样-蒙特卡洛控制"""
def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    找到最优贪婪策略（目标策略）

    Args:
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.

    Returns:
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns action probabilities. This is the optimal greedy policy.
    """

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))   # 累积重要性权重
    target_policy = create_greedy_policy(Q)   # 初始化目标策略

    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode. An episode is an array of (state, action, reward) tuples
        episode = []   # 重置轨迹信息
        state = env.reset()

        """①、采样"""
        for t in range(100):
            probs = behavior_policy()   # 按照行为策略进行采样，得到当前状态的动作概率[0.5，0.5]，保证了探索性
            action = np.random.choice(np.arange(len(probs)), p=probs)   # 随机选择动作
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))   # 形如：[((13, 6, False), 1, 0), ((15, 6, False), 0, -1)]   或   [((13, 6, False), 1, 0)]
            if done:
                break
            state = next_state

        G = 0.0   # Sum of discounted returns(未来折扣回报)
        W = 1.0   # The importance sampling ratio (the weights of the returns) 重要性权重参数

        """②、广义策略迭代"""
        for t in range(len(episode))[::-1]:   # 注意-1，用于从最后的一个时间步开始遍历
            s, a, reward = episode[t]   # 行为策略采样的 状态,动作,奖励

            G = reward + discount_factor * G   # Update the total reward since step t 获取当前时间步的未来折扣回报 —— 其实就是从首段开始遍历的 Σ奖励*折扣系数**i 的另一种计算形式
            C[s][a] += W   # Update weighted importance sampling formula denominator   更新累积权重

            # Update the action-value function using the incremental update formula (5.7)
            # This also improves our target policy which holds a reference to Q
            Q[s][a] += (W / C[s][a]) * (G - Q[s][a])   # 策略评估

            best_a = np.argmax(target_policy(s))   # 策略改进：目标策略采取的动作。此时的Q已经被更新，所以目标策略是提升过的

            print(a,"*",best_a)
            # If the action taken by the behavior policy is not the action taken by the target policy the probability will be 0 and we can break
            if a !=  best_a:   # 如果行为策略采样的动作不是目标策略采取的动作，则概率为0，退出本次经验轨迹的时间步的迭代  ★
                print('*'*10)
                break

            W = W * 1./behavior_policy()[a]   # 根据行为策略更新重要性权重参数

    return Q, target_policy


env = BlackjackEnv()
random_policy = create_random_policy(env.action_space.n)   # 初始化策略
Q, policy = mc_control_importance_sampling(env, num_episodes=100000, behavior_policy=random_policy)


V = defaultdict(float)
for state, action_values in Q.items():
    V[state] = np.max(action_values)


# plotting.plot_value_function(V, title="Optimal Value Function")
