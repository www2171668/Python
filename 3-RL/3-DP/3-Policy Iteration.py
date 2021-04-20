import numpy as np
import pprint
import sys
from gridworld import GridworldEnv  # 引入gridworld包

"""策略评价"""


def policy_eval(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Args:
        policy: [S, A] shaped matrix representing the policy.  π=[状态，动作]
        env: OpenAI env. env.P represents the transition probabilities of the environment.   状态转移概率矩阵
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        discount_factor: Gamma discount factor.
        theta: We stop evaluation once our value function change is less than theta for all states.

    Returns:
        Vector of length env.nS representing the value function.   以env.nS为形状的状态值函数矩阵
    """

    V = np.zeros(env.nS)  # 初始化状态值函数列表
    while True:
        delta = 0  # 定义最大差值，判断是否有进行更新

        for s in range(env.nS):  # 遍历所有状态 [0~15]
            v = 0  # 针对每个状态值函数进行计算

            for a, action_prob in enumerate(policy[s]):  # 遍历状态s下的4种动作，获得 动作 和 策略函数
                for prob, next_state, reward, done in env.P[s][a]:  # 通过(s,a)获取 状态转移概率,下一状态,奖励,是否结束
                    v += action_prob * prob * (
                                reward + discount_factor * V[next_state])  # action_prob * prob 就是 π(s,a)*P(s',r|s,a)

            delta = max(delta, np.abs(v - V[s]))  # 更新差值
            V[s] = v  # 存储(更新)每个状态下的状态值函数，即伪代码中的 v <- V(s)

        if delta < theta:  # 策略评估的迭代次数不能太多，否则状态值函数的数值会越来越大（即使算法仍然在收敛）
            break
    return V  # 一轮迭代结束后，状态值函数暂时固定


def Caculate_Q(s, V, discount_factor=1):
    """
    Returns:
        A vector of length env.nA containing the expected value of each action.
    """
    Q = np.zeros((env.nS, env.nA))
    for a in range(env.nA):  # 遍历所有动作
        for prob, next_state, reward, done in env.P[s][a]:
            Q[s][a] += prob * (reward + discount_factor * V[next_state])  # 计算当前状态s下的动作值函数列表 [q1,q2,q3,q4]
    return Q


def policy_improvement(env, V, policy, discount_factor=1.0):  # 策略改进
    """
    Returns:
        policy: the optimal policy, a matrix of shape [S, A] where each state s contains a valid probability distribution over actions.
        V: the value function for the optimal policy.
    """
    policy_stable = True  # Will be set to false if we make any changes to the policy
    for s in range(env.nS):
        old_a = np.argmax(policy[s])  # 记录当前策略在该状态s下所选择的动作 —— 选择概率最高的动作

        Q = Caculate_Q(s, V)  # 在当前状态和策略下，计算动作值函数 —— 要判断在状态s下选择其他动作的好坏，就需要获得状态s的动作值函数
        best_a = np.argmax(Q[s])  # 采用贪婪策略获得状态s的最优动作；如果往两个方向前进都可以得到最优解，会随机选其一

        if old_a != best_a:  # 判定策略是否稳定
            policy_stable = False  # 动作还在变化，不稳定状态

        policy[s] = np.eye(env.nA)[best_a]  # 基于贪婪法则更新策略，形如[0，0，1，0]； -》np.eye(*)：构建对角单位阵
        # 经过一次策略改进后的策略将不再拥有多个动作可供备选，取而代之的是在某种状态下的确定策略
    return policy, V, policy_stable


def Policy_Iterration(env, discount_factor=1.0):
    policy = np.ones([env.nS, env.nA]) / env.nA  # 初始化策略，初始策略函数为0.25   [[ 0.25  0.25  0.25  0.25],...]   16*4 矩阵
    while True:  # Evaluate the current policy
        V = policy_eval(env, policy,
                        discount_factor)  # 得到当前策略下的收敛状态值函数 —— 与Value_Iteration的不同点，多了policy_eval()函数。policy会在迭代中改变
        policy, V, policy_stable = policy_improvement(env, V, policy)

        print(V.reshape(env.shape))
        print(np.argmax(policy, axis=1).reshape(env.shape))  # 输出在每个状态上会采取的动作
        print("*" * 100)

        if policy_stable:  # # If the policy is stable we've found an optimal policy. Return it
            return policy, V


env = GridworldEnv()  # 引入环境
policy, V = Policy_Iterration(env)

# print("Policy Probability Distribution:")
# print(policy)   # 动作是从 低状态值 向 高状态值 的方向进行选择的
# print("*"*100)


# Test the value function
# expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
# np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
