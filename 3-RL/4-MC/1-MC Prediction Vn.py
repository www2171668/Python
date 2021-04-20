# %matplotlib inline

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()  # 引入环境


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):  # 蒙特卡洛算法 Monte Carlo prediction algorithm
    """
    Calculates the value function for a given policy using sampling(抽样计算).

    Args:
        policy: A function that maps(映射) an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.   要采样的episode总数
        discount_factor: Gamma discount factor.

    Returns:
        A dictionary that maps from state -> value.  The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state to calculate an average.
    # We could use an array to save all returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)  # 回报值 G，dict
    returns_count = defaultdict(float)  # 回报数量 N，dict

    # The final value function
    V = defaultdict(float)  # 最终状态价值函数

    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:  # 每1000次打印一次
            # \r 表示将光标的位置回退到本行的开头位置；end=""使打印不换行；配合使用以达到多次打印覆盖上一次打印的效果
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()  # flush()刷新缓存区，实时打印

        # Generate an episode.   An episode is an array of (state, action, reward) tuples
        episode = []  # 多episodes。每一次迭代都会重置episode
        state = env.reset()  # 环境重置。返回一个随机observation(state)   state = (玩家总牌数，庄家第一张牌，是否可以使用A的11)

        for t in range(100):  # 随机地走，直到吸收状态或到100次(实际一般只有1-3次，哪有100次那么多)，记录下每一次行动
            action = policy(state)  # 策略
            next_state, reward, done, _ = env.step(action)  # 进入下一步。输入action，返回(observation, reward, done, info)
            episode.append((state, action,
                            reward))  # 如 [((15, 7, False), 1, 0), ((19, 7, False), 1, 0), ((20, 7, False), 0, 1)]   done=1时停止；由于游戏性质，多数情况下只有一个元组
            if done:  # 判断是否达到吸收状态
                break
            state = next_state  # 更新状态

        # Find all states the we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        states_in_episode = set([tuple(x[0]) for x in episode])  # 通过set对状态去重 + 打乱顺序
        # print(states_in_episode)
        for state in states_in_episode:  # 遍历每一个状态，这些状态都是唯一的
            # 首次访问：Find the first occurance of the state in the episode
            # 遍历原始episode中的每一个state，与当前循环的state进行对比，先记录下所有相同的状态编号，然后用next取第一个
            first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == state)
            # Sum up all rewards since the first occurance  计算首次访问的状态的回报（累计奖励）
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])

            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G  # 更新回报G 。在多个episodes中，针对每一个起始相同的state计算其回报
            returns_count[state] += 1.0  # 更新动作-状态对的计数器
            V[state] = returns_sum[state] / returns_count[state]  # 以回报的均值作为状态state对应的状态价值函数。这里没有用到增量计算法

    return V  # 所有episodes结束后，返回字典型V，表示为{state:状态价值函数,....}


def sample_policy(observation):  # 这里的sample_policy实际上是policy()方法，传入的observation实际上是policy(state)中的state
    """A policy that sticks if the player score is >= 20 and hits otherwise."""
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


V_10k = mc_prediction(sample_policy, env, num_episodes=10000)  # 总迭代次数为num_episodes次
print()
print(V_10k)  # {(16, 8, False): -0.65, (14, 4, False): -0.5119047619047619,...}
# plotting.plot_value_function(V_10k, title="10,000 Steps")

# V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
# plotting.plot_value_function(V_500k, title="500,000 Steps")
