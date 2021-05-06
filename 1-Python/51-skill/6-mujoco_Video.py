""""""
import gym
from gym import wrappers
import time

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, './video', video_callable=False  ,force=True)
for i_episode in range(5):
    # env = wrappers.Monitor(env, 'video', video_callable=lambda i_episode: i_episode % 5 == 0, force=True)
    observation = env.reset()
    for t in range(1000):
        print(observation)
        action = env.action_space.sample()
        s, r, done, info = env.step(action)
        if done:
            print("Episode finished after {} timestep".format(t + 1))
            break

# env.monitor.close()
env.close()
