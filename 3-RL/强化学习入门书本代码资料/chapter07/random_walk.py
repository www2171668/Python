''''''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


N_STATES = 19   # all states
GAMMA = 1   # discount
STATES = np.arange(1, N_STATES + 1)   # all states but terminal states [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
START_STATE = 10   # start from the middle state

# two terminal states
# an action leading to the left terminal state has reward -1
# an action leading to the right terminal state has reward 1
END_STATES = [0, N_STATES + 1]

# true state value from bellman equation
TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
TRUE_VALUE[0] = TRUE_VALUE[-1] = 0
# [ 0.  -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1  0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  0. ]

# n-steps TD method
# value: values for each state, will be updated
# n: of steps
# alpha: step-size
def temporal_difference(value, n, alpha):
    state = START_STATE   # initial starting state

    # arrays to store states and rewards for an episode
    # space isn't a major consideration, so I didn't use the mod trick
    states = [state]
    rewards = [0]

    time = 0
    T = float('inf')   # the length of this episode  -》float('inf')：无穷大
    while True:
        time += 1

        if time < T:
            # 设定(0.5,0.5)的随机策略
            if np.random.binomial(1, 0.5) == 1:
                next_state = state + 1
            else:
                next_state = state - 1

            if next_state == 0:   # 判断终止状态
                reward = -1
            elif next_state == 20:
                reward = 1
            else:
                reward = 0

            # store new state and new reward
            states.append(next_state)
            rewards.append(reward)

            if next_state in END_STATES:
                T = time

        # get the time of the state to update
        update_time = time - n
        if update_time >= 0:
            returns = 0.0   # 初始化回报
            # ①、calculate corresponding rewards
            for t in range(update_time + 1, min(update_time + n, T) + 1):   # 这个t在伪代码中用i表示
                returns += pow(GAMMA, t - update_time - 1) * rewards[t]
            # ②、dd state value to the return
            if update_time + n <= T:
                returns += pow(GAMMA, n) * value[states[(update_time + n)]]

            # ③、update the state value
            state_to_update = states[update_time]   # 获取当前正在被更新的状态
            if not state_to_update in END_STATES:
                value[state_to_update] += alpha * (returns - value[state_to_update])
        if update_time == T - 1:
            break
        state = next_state


def figure7_2():
    # 有点像机器学习里的管道
    steps = np.power(2, np.arange(0, 10))   # all possible steps
    alphas = np.arange(0, 1.1, 0.1)   # all possible alphas

    episodes = 10   # each run has 10 episodes
    runs = 10   # perform 100（10） independent runs

    # track the errors for each (step, alpha) combination
    errors = np.zeros((len(steps), len(alphas)))
    for run in tqdm(range(0, runs)):

        for step_ind, step in enumerate(steps):   # 管道
            for alpha_ind, alpha in enumerate(alphas):   # 管道
                # print('run:', run, 'step:', step, 'alpha:', alpha)

                value = np.zeros(N_STATES + 2)   # 初始化状态值函数
                for ep in range(0, episodes):   # 迭代开始
                    temporal_difference(value, step, alpha)
                    errors[step_ind, alpha_ind] += np.sqrt(np.sum(np.power(value - TRUE_VALUE, 2)) / N_STATES)   # calculate the RMS error

    errors /= episodes * runs   # take average

    for i in range(0, len(steps)):
        plt.plot(alphas, errors[i, :], label='n = %d' % (steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()

    plt.savefig('figure_7_2.png')
    plt.close()

if __name__ == '__main__':
    figure7_2()


