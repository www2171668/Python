import numpy as np

"""定义格子世界参数"""
world_h = 4
world_w = 4
length = world_h * world_w
gamma = 1
state = [i for i in range(length)]  # 状态（编号）
action = ['n', 'e', 's', 'w']  # 动作名称
ds_action = {'n': -world_w, 'e': 1, 's': world_w, 'w': -1}
value = [0 for i in range(length)]  # 初始化状态值函数，均为0.  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# 定义奖励
def reward(s):
    return 0 if s in [0, length - 1] else -1  # 左上角和右下角格子奖励为0，其他为-1
    # in表示0是[*，*，*]中的一个


# 在s状态下执行动作a，返回下一状态（编号）
def next_states(s, a):
    # 越过边界时pass
    if (s < world_w and a == 'n') \
            or (s % world_w == 0 and a == 'w') \
            or (s > length - world_w - 1 and a == 's') \
            or ((s + 1) % world_w == 0 and a == 'e'):  # (s % (world_w - 1) == 0 and a == 'e' and s != 0)
        next_state = s  # 表现为next_state不变
    else:
        next_state = s + ds_action[a]  # 进入下一个状态
    return next_state


# 在s状态下执行动作，返回所有可能的下一状态（编号）list
def getsuccessor(s):
    successor = []
    for a in action:  # 遍历四个动作
        next = next_states(s, a)  # 得到下一个状态（编号）
        successor.append(next)  # 以list保存当前状态s下执行四个动作的下一状态
    return successor


# 更新状态值函数
def value_update(s):  # 传入当前状态
    value_new = 0
    if s in [0, length - 1]:  # 若当前状态为吸入状态，则直接pass不做操作
        pass
    else:
        successor = getsuccessor(s)  # 得到所有可能的下一状态list
        rewards = reward(s)  # 得到当前状态的奖励
        for next_state in successor:  # 遍历所有可能的下一状态
            value_new += 0.25 * (rewards + gamma * value[next_state])  # 计算公式，得到当前状态的状态价值函数
            # 注意前面的0.25，该代码是第一次迭代时的固定策略π(a|s)

    return value_new


def initial_state():
    v = np.array(value).reshape(world_h, world_w)  # 调整初始化状态值函数矩阵
    print(v)


def main():
    max_iter = 201  # 最大迭代次数
    initial_state()

    iter = 1
    while iter < max_iter:
        for s in state:  # 遍历所有状态
            value[s] = value_update(s)  # 更新状态值函数

        v = np.array(value).reshape(world_h, world_w)  # 更新状态值函数矩阵

        if (iter <= 10) or (iter % 10 == 0):  # 前1次 + 每10次打印一次
            print('k=', iter)  # 打印迭代次数
            print(np.round(v, decimals=2))  # np.round() 返回浮点数的四舍五入值

        iter += 1


if __name__ == '__main__':
    main()
