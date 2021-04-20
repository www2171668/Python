import gym
import time

print(gym.envs.registry.all())

"""
    -》gym.make(*)  加载环境
"""
env = gym.make('MsPacman-v0') #CartPole-v0  加载环境

print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)

"""
    -》env.reset()：环境初始化
    -》env.render()：显示环境中的物体图像，便于直观显示当前环境物体的状态。
    -》env.step()：描述智能体与环境的交互信息。其输入是动作a，输出是：下一步状态，立即回报，是否终止，调试项。
"""
for i_episode in range(10):   # 总回合数
    observation = env.reset()   # 环境的初始化
    time.sleep(0.1)
    for t in range(1000):   # 一个回合里，总的时间步
        env.render()  # 环境展示（渲染）
        #time.sleep(0.1)
        print(observation)   # 打印环境观测内容
        action = env.action_space.sample()  # 随机从动作空间中选取动作
        observation, reward, done, info = env.step(action)  # 根据动作获取下一步的信息
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

quit()
