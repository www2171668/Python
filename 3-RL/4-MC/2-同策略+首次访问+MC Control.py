import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

""" 
    ε-贪婪策略
    根据动作值函数，每次都将一个最优动作概率设置为0.95，另一个设置为0.05
"""
def make_epsilon_greedy_policy(Q, epsilon, nA ):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        epsilon: The probability to select a random action . float between 0 and 1.   探索概率
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument(参数) and returns the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(s):   # 输入状态
        A = np.ones(nA, dtype=float) * epsilon / nA   # 初始化动作概率，均为epsilon / nA = 0.05
        # 由于Q是defaultdic类型的，所以如果s不存在，则会增加一组{s:(0,0)}的数据；如果存在，则查找到相应动作概率tuple  ★
        best_a = np.argmax(Q[s])   # 对于新s来说，返回值都是(0，0)，所以随机选择一个；对于旧s，选取具有最大动作值函数的动作，即最优动作
        A[best_a] += (1.0 - epsilon)   # 以1-ε = 0.95 设定最大的动作概率
        return A   # 只有[ 0.05  0.95] 或 [ 0.95  0.05]两种类型
    return policy_fn


"""固定策略的非起始点探索蒙特卡洛控制算法"""
def on_policy_MC_control(env, num_episodes, discount_factor=1.0, epsilon=0.1):   # Monte Carlo Control using Epsilon-Greedy policies.
    """
    Finds an optimal epsilon-greedy policy.

    Args:
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        Q is a dictionary mapping state -> action values；Each value is a numpy array of length nA (see below)
        policy is a function that takes an observation as an argument and returns action probabilities
    """

    # Keeps track of sum and count of returns for each state to calculate an average.
    # We could use an array to save all returns (like in the book), but that's memory inefficient(内存效率很低).
    Returns_sum = defaultdict(float)
    Returns_count = defaultdict(float)

    # A nested dictionary(嵌套字典) that maps state -> (action-value1, action-value2).
    # {(state):array([第一个动作对应的动作值函数, 第二个动作对应的动作值函数]),...}   ★     如{(20, 5, False): array([ 0.,  0.])，...}
    # env.action_space.n：动作空间大小nA，在这里为2。 初始化值为{ }，每次Q的初始动作都是array([ 0.,  0.])
    # 由于Q定义为列表型字典，之后更新Q的时候都是深拷贝式更新，相当于在调用policy()时，make_epsilon_greedy_policy()中的Q会跟着更新   ★★★
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # 返回的是policy_fn(observation)方法 ★
    # 体现的是同策略思想，共用行为策略和目标策略
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)   # 初始化ε-贪婪策略
    for i_episode in range(1, num_episodes + 1):   # 开始迭代
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode. An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()

        """
            ①、采样
        """
        for t in range(100):   # 对本次episode进行采样。实际采样不到这么多次，很快就停了
            # 对make_epsilon_greedy_policy中的policy_fn进行调用   ★
            # 在调用policy_fn时，make_epsilon_greedy_policy(Q, *，*)也会被调用，所以深拷贝的Q值在一轮迭代更新后，make_epsilon_greedy_policy也会使用新的Q值，这样使得policy()发生变化
            probs = policy(state)   # 通过ε-贪婪策略获得动作概率 —— action_probability为[ 0.95  0.05] 或 [ 0.05  0.95]
            action = np.random.choice(np.arange(len(probs)), p=probs)   # 保证了探索性。返回的是动作编号，0或1
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))   # 存储采样。体现的是离线学习思想
            # print(episode)
            if done:
                break
            state = next_state

        """
            ②、广义策略迭代
        """
        # 去重 + 乱序。需要说明的是，在这个游戏中，一次episode是不会存在两个相同的(state,action)
        # 由于state是一个三元组(*，*，*)，所以用tuple来接收
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for s, a in sa_in_episode:
            sa_pair = (s, a)
            # print(sa_pair)
            # Find the first occurance of the (state, action) pair in the episode  找到的是(s,a)第一次的出现位置
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == s and x[1] == a)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])   # Σ奖励*折扣系数**i

            # Calculate average return for this state over all sampled episodes
            Returns_sum[sa_pair] += G   # 存储了每一组(s,a)的回报值
            Returns_count[sa_pair] += 1.0

            Q[s][a] = Returns_sum[sa_pair] / Returns_count[sa_pair]   # 这里并没有用增量法计算
        # The policy is improved implicitly by changing the Q dictionary   ★

    return Q, policy

env = BlackjackEnv()
Q, policy = on_policy_MC_control(env, num_episodes=10000, epsilon=0.1)

"""求状态动作值函数中每一个状态对应的最大值，依次作为状态值函数中该状态的价值期望"""
V = defaultdict(float)
#print(Q.items())
for state, action_values in Q.items():
    V[state] = np.max(action_values)   # 最优状态值函数 = 最优动作值函数。 形如{(13, 9, False): -0.34693877551020408, (17, 10, False): -0.4838709677419355,...}
# print(V.keys())
"""For plotting(画图): Create value function from action-value function by picking the best action at each state"""
plotting.plot_value_function(V, title="Optimal Value Function")