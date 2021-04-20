import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table

"""定义格子世界参数"""
WORLD_SIZE = 4
ACTIONS = [np.array([0, -1]),  # j-1，即向左移动,left
           np.array([-1, 0]),  # up
           np.array([0, 1]),  # right
           np.array([1, 0])]  # down
ACTION_PROB = 0.25  # 策略


# Pss'a = 1

# 判断是否已经终点状态
def is_terminal(state):
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)  # 为终点状态时，返回True，否则为False


# 根据(s,a)，返回(r,s')
def step(s, a):
    s = np.array(s)  # list转numpy，用于和numpy类型的action相加
    next_s = (s + a).tolist()  # 得到下一个状态  numpy转list
    x, y = next_s

    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:  # 判断下一状态是否超出棋盘
        next_s = s.tolist()  # 超出棋盘时，将上一状态返还给当前状态

    reward = -1  # 奖励为-1
    return next_s, reward


"""
    ①、in_place=True时，在一轮迭代中，在遍历状态时，每计算完一个状态，就立刻更新该状态的状态值函数
    ②、in_place=False时，在一轮迭代中，一直使用上一轮结束时的状态值函数矩阵
        用in_place=True往往可以减少迭代次数，后面我们都是用in_plcae来处理动态规划问题
"""


def compute_V_sync():
    V = np.zeros((WORLD_SIZE, WORLD_SIZE))
    new_V = V.copy()
    iteration = 1
    while True:
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                if is_terminal([i, j]):
                    continue

                v = 0
                for a in ACTIONS:
                    (next_i, next_j), reward = step([i, j], a)
                    v += ACTION_PROB * (reward + V[
                        next_i, next_j])  # π(a|s)*(奖励 + 1*下一时刻的状态值函数) —— 在in_pacle和off_place中，区别就在这个下一时刻的状态值函数
                new_V[i, j] = v  # 根据in_plcae，每计算完一个状态，就立刻更新该状态的状态值函数

        if np.sum(np.abs(new_V - V)) < 1e-4:  # 判断新旧状态值函数的绝对值差值是否小于阈值
            V = new_V.copy()  # 更新状态值函数，并停止迭代
            break
        else:
            V = new_V.copy()  # 更新状态值函数
            iteration += 1

    return V, iteration


def compute_V_asycn():  # in-place   ★
    V = np.zeros((WORLD_SIZE, WORLD_SIZE))  # 初始化状态值函数矩阵
    iteration = 1  # 记录迭代次数
    while True:
        delta = 0  # 定义最大差值，判断是否有进行更新

        # 在一次迭代中遍历所有状态，用[i,j]表示所在状态，左上角为[0,0]状态
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                if is_terminal([i, j]):  # 判断当前状态是否为终点状态
                    continue  # 是终点状态时，跳入最外层循环（while），继续开始迭代

                v = 0
                for a in ACTIONS:
                    (next_i, next_j), reward = step([i, j], a)
                    v += ACTION_PROB * (reward + V[
                        next_i, next_j])  # π(a|s)*(奖励 + 1*下一时刻的状态值函数) —— 在in_pacle和off_place中，区别就在这个下一时刻的状态值函数

                delta = max(delta, np.abs(v - V[i, j]))  # 更新差值
                V[i, j] = v  # 根据in_plcae，每计算完一个状态，就立刻更新该状态的状态值函数

        if delta < 1e-4:  # 判断新旧状态值函数的绝对值差值是否小于阈值
            break

        iteration += 1

    return V, iteration


# 绘制数据表格
def draw_image(image):  # 传入的image其实是状态值函数矩阵(4*4)
    fig, ax = plt.subplots()
    ax.set_axis_off()  # 不显示坐标轴
    tb = Table(ax)  # 在坐标轴上绘制数据表格。bbox=[0, 0, 1, 1] 默认起点为(0,0)，单元格宽和高都为1

    nrows, ncols = image.shape  # WORLD_SIZE * WORLD_SIZE 形状
    width, height = 1 / ncols, 1.0 / nrows

    # ①、在Table中加入元素
    for (i, j), val in np.ndenumerate(image):  # val为[i，j]位置的状态值函数
        """"-》tb.add_cell()
            width和height：表示单元格大小，而Table(bbox)中的bbox会对width和height造成影响，一般不设置bbox也行
            loc：行性质
            facecolor：单元格底色，为'none'和'white'是一样的
            edgecolor：单元格边框色，为'none'表示无边框
            """
        tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    # ②、Row Labels...
    for i, label in enumerate(range(len(image))):  # len（image） = 4
        # -1表示在表格之外（左侧）
        tb.add_cell(i, -1, width, height, text=label + 1, loc='right', edgecolor='none', facecolor='none')
    # ③、Column Labels...
    for j, label in enumerate(range(len(image))):
        tb.add_cell(-1, j, width, height / 2, text=label + 1, loc='center', edgecolor='none',
                    facecolor='none')  # height/2只是用于减少单元格高度，使标签靠下

    ax.add_table(tb)  # -》ax.add_table(*) ：将tb数据表格应用于图形上


def main():
    sync_values, sync_iteration = compute_V_sync()
    asycn_values, asycn_iteration = compute_V_asycn()

    print(sync_values)
    draw_image(np.round(sync_values, decimals=2))  # 传入状态值矩阵（保留小数点后两位）
    print(asycn_values)
    draw_image(np.round(asycn_values, decimals=2))  # 传入状态值矩阵（保留小数点后两位）

    print('Synchronous: %d iterations' % (sync_iteration))
    print('In-place: %d iterations' % (asycn_iteration))

    """-》plt.savefig(地址/名称)：保存图片"""
    plt.savefig('./figure_4_1.png')  # 保存
    plt.close()


if __name__ == '__main__':
    main()
