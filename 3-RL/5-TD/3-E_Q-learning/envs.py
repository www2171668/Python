from __future__ import print_function
from __future__ import division

import numpy as np


class GridWorld():

    def __init__(self):
        # 网络参数
        self.env_w = 10
        self.env_h = 10
        self.num_s = self.env_w * self.env_h
        self.num_a = 4   # 动作数

        r = np.zeros(self.num_s)
        w = np.zeros(self.num_s)

        self.target = np.array([27])   # 终点位置(pos)
        self.terminal = self.target   # 终点位置
        self.bomb = np.array([16, 25, 26, 28, 36, 40, 41, 48, 49, 64])   # 炸弹的位置
        self.wall = np.array([22, 32, 42, 52, 43, 45, 46, 47, 37])   # 墙的位置
        r[self.target] = 10   # 目标奖励
        r[self.bomb] = -1   # 炸弹奖励
        r[self.wall] = 0   # 墙奖励
        w[self.wall] = 1   # 对墙位置的w进行更新，以控制agent的行动

        # 重新赋值方便使用而已
        self.W = w
        self.R = r


    def rewards(self, s, a):
        return self.R[s]

    def allow_actions(self, s):
        # return allow actions in state s
        x = self.get_pos(s)[0]
        y = self.get_pos(s)[1]
        allow_a = np.array([], dtype='int')   # 存下当前s的可执行动作 0-up ，1-down，2-left，3-right。[0,1,2,3]

        if y > 0 and self.W[s-self.env_w] != 1:   # 在可以往上走的情况下，判定往上走是否会撞墙
            allow_a = np.append(allow_a, 0)   # 不会撞墙，则将动作0存入allow_a数组
        if y < self.env_h-1 and self.W[s+self.env_w] != 1:
            allow_a = np.append(allow_a, 1)
        if x > 0 and self.W[s-1] != 1:
            allow_a = np.append(allow_a, 2)
        if x < self.env_w-1 and self.W[s+1] != 1:
            allow_a = np.append(allow_a, 3)
        return allow_a

    def get_pos(self, s):
        '''transform to coordinate (x, y)'''
        x = s % self.env_h   # 假设 26%10 = 6
        y = s / self.env_w   # 假设 26/10 = 2
        return x, y

    def next_state(self, s, a):
        # return next state in state s taking action a in this definitized environment it returns a certain state ns
        ns = 0
        if a == 0:   # up
            ns = s - self.env_w
        if a == 1:   # down
            ns = s + self.env_w
        if a == 2:   # left
            ns = s - 1
        if a == 3:   # right
            ns = s + 1
        return ns

    def is_terminal(self, s):
        return True if s in self.terminal else False

    def reset(self):
        return 0  # initi state
