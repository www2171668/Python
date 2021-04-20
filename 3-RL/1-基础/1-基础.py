import numpy as np
import gym
import random

'''
    gym开源库：包含一个测试问题集，每个问题为一个环境env, 环境有共享的接口，允许用户设计通用的算法
    OpenAI gym服务：提供站点和API允许用户对训练的算法进行性能比较

    -》Env：gym的核心接口，包含以下核心方法：
        reset()：重置环境的状态，返回观察。
        step(action)：输入动作a，返回 observation, reward, done, info
        render(mode=’human’, close=False)：重绘环境的一帧
'''
# CartPole-v0
env = gym.make('Pendulum-v0')   # 创建环境（系统自带的Pendulum-v0摆钟）

env.reset()
while True:
    env.render()    # 渲染环境，用来显示环境中的物体图像，便于直观显示当前环境物体的状态
    a = random.random()   # 产生0-1的随机数。作为摆钟的力矩
    action = np.array([a])
    print('current torque',action)
    env.step(action)      # 与环境进行交互


