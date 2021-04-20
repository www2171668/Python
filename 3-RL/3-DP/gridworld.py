import numpy as np
import sys
from gym.envs.toy_text import discrete
from io import StringIO


UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3   # 定义动作

class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal state at the top left or the bottom right corner.

    For example, a 4x4 grid looks as follows:
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T

    x is your position and T are the two terminal states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[4,4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape)   # -》np.prod()：计算所有元素的乘积。   16个状态
        nA = 4   # 4个动作

        MAX_Y = shape[0]   # 4
        MAX_X = shape[1]   # 4

        P = {}
        grid = np.arange(nS).reshape(shape)   # 绘制4*4网格，并辅以状态编号
        it = np.nditer(grid, flags=['multi_index'])   # 对网格进行遍历
        """
            -》np.nditer(a,flags,op_flags)：np自带的迭代器
                flags=['multi_index']：对a进行多重索引
                op_flags=['readwrite']：对a进行read（读取）和write（写入），即设定迭代器权限。
            
            -》it.multi_index：输出元素的索引index。
            -》it.iternext()：进入下一次迭代。如果不加这一句的话，输出的结果就一直都是(0, 0)。
            
            -》it.finished：Whether the iteration over the operands is finished or not.
            -》it.iterindex：An index which matches the order of iteration.
            -》it.multi_index：When the “multi_index” flag was used, this property provides access to the index.

        """
        while not it.finished:
            s = it.iterindex   # 返回索引   ★
            y, x = it.multi_index   # 返回数组位置索引  ★

            P[s] = {a : [] for a in range(nA)}   # 初始化P[s]

            is_done = lambda s: s == 0 or s == (nS - 1)   # 为True时：当前状态为吸收状态。 匿名函数
            reward = 0.0 if is_done(s) else -1.0   # 定义奖励值

            if is_done(s):
                P[s][UP]    = [(1.0, s, reward, True)]
                P[s][DOWN]  = [(1.0, s, reward, True)]
                P[s][LEFT]  = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
            else:   # Not a terminal state
                # 状态更新 + 碰壁处理。以左上角为[x=0,y=0]
                # 注意状态s上下移动时跨度为MAX_X和MAX_Y，而不是以[i,j]矩阵操作状态的
                ns_up    = s if y == 0           else s - MAX_X   # 当前状态为s，动作为UP时的下一个状态
                ns_down  = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left  = s if x == 0           else s - 1
                ns_right = s if x == (MAX_X - 1) else s + 1

                # [(状态转移概率，下一个状态点，奖励，是否结束),...] —— 1.0表示Pss'a是固定的
                P[s][UP]    = [(1.0, ns_up   , reward, is_done(ns_up))]
                P[s][DOWN]  = [(1.0, ns_down , reward, is_done(ns_down))]
                P[s][LEFT]  = [(1.0, ns_left , reward, is_done(ns_left))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]

                # 需要说明，从1格进入0格虽然id_done是True，但是奖励reward没有更新，还是-1
                # 体现了，Agent先得奖励再转移状态，所以A格向左转移时还是得到了-1奖励  ★

            it.iternext()   # 进入下一次迭代

        # Initial state distribution is uniform(均匀的)  初始状态分布均匀
        isd = np.ones(nS) / nS   # 初始时每一个状态都是1/16

        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)   # 继承父类discrete.DiscreteEnv，这是系统自带的强化学习环境

    def _render(self, mode='human', close=False):
        if close:
            return

        # StringIO常被用来作为字符串的缓存，StringIO的有些接口和文件操作是一致的，可用于文件操作或者StringIO操作
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip() 
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
