"""
Policy Gradient, Reinforcement Learning.
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

RENDER = False  # rendering wastes time

env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible(可再利用的), general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)   # 可能的动作   Discrete(3)
print(env.action_space.n)   # 3
print(env.observation_space)   # 可能的搜索状态   Box(2,)
print(env.observation_space.shape[0])   # 2 状态 = (位置,速度)
print(env.observation_space.high)   # [ 0.60000002  0.07      ]
print(env.observation_space.low)   # [-1.20000005 -0.07      ]

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,   # 学习率
    reward_decay=0.995,   # 奖励衰减系数
    # output_graph=True,
)

for i_episode in range(1000):
    observation = env.reset()

    while True:
        if RENDER:
            env.render()

        """收集数据"""
        action = RL.choose_action(observation)
        observation_, reward, done, info = env.step(action)     # reward = -1 in all cases
        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)   # calculate running reward 当前伦次的回报
            if 'running_reward' not in globals():   # 第一轮迭代是没有running_reward在globals()中的
                running_reward = ep_rs_sum   # running_reward：表示回报
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01   #
            if running_reward > DISPLAY_REWARD_THRESHOLD:    # 单次游戏回报大于限定值，开始渲染游戏
                RENDER = True     # rendering

            print("episode:", i_episode, "  reward:", int(running_reward))

            Vt = RL.learn()  # train

            if i_episode == 30:
                plt.plot(Vt)  # plot the episode Vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()

            break

        observation = observation_   # 更新状态
