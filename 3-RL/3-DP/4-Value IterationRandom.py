import numpy as np
import pprint
import sys
from gridworld import GridworldEnv


env = GridworldEnv()

def Calculate_Q(s, V , discount_factor = 1.0):
    Q = np.zeros((env.nS, env.nA))
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            # Bellman最优状态值函数求解方程。+= 就是在当前状态下，对所有动作的(即时奖励，长期奖励)效果进行求和
            Q[s][a] += prob * (reward + discount_factor * V[next_state])   # 之后使用np.max得到v*(s)
    return Q   # 返回状态s下的四种动作的值函数数组[*,*,*,*]

def value_iteration(env, theta=0.0001, discount_factor=1.0):   # 值迭代算法
    V = np.zeros(env.nS)   # 初始化状态值函数

    while True:
        delta = 0

        for s in range(env.nS):
            Q = Calculate_Q(s, V)
            v = np.max(Q[s])   # 得到最优状态值函数

            delta = max(delta, np.abs(v - V[s]))   # 计算状态值函数增量
            V[s] = v   # 更新状态值函数

        print(V.reshape(env.shape))

        if delta < theta:
            break   # 此时已经获得最优状态值函数，不再做调整

    policy = np.random.rand(env.nS, env.nA)   # 每一个状态都有一组随机策略[*,*,*,*]
    print(policy)
    for s in range(env.nS):
        Q = Calculate_Q(s, V)
        best_action = np.argmax(Q[s])   # 以可获得最大的值函数的动作为当前状态最优动作
        # policy[s, best_action] = 1.0

    return policy, V

policy, v = value_iteration(env)

# print("Policy Probability Distribution:")
# print(policy)
# print("*"*100)

# print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
# print(np.argmax(policy, axis=1).reshape(env.shape))   # env.shape就是(4*4)
# print("*"*100)


# 验证最优状态值函数
# expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
# np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
