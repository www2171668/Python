""""""


class Environment():
    def __init__(self):

        self.startstate = state([0, 3])   # 设置起点   这里接收的是实例化后的类，不是一个是实质的数
        self.goalstate = state([7, 3])   # 设置终点

        self.agentstate = state(self.startstate.pos)   # 起始状态位置
        self.agent_reward = 0   # 初始奖励为0
        self.action_dict = {
            "left": (-1, 0),
            "right": (1, 0),
            "up": (0, 1),
            "down": (0, -1)  # ,
            #"left_up":(-1,1),   # 也可以增加这样的动作
            #"up_right":(1,1),
            #"right_down":(1,-1),
            #"down_left":(-1,-1)
        }

    def loadActions(self, agent):   # 加载动作
        for action_name in self.action_dict.keys():   # 遍历所有动作名称 "left"、"right"...
            agent.actions.append(action_name)   # 添加到actions列表中

    def reset(self):
        self.agentstate = state(self.startstate.pos)   # 回到起始状态。 startstate.pos = [0, 3]
        return self.agentstate.name, self.agent_reward   # 返回当前时刻的state和reward

    # only environment knows the start and goal state, so it is appropriate that QValue of the two states is set in this class
    # 初始化起始状态和结束状态的动作值函数
    def initValue(self, agent):
        agent.initValue(self.startstate.name, randomized=True)
        agent.initValue(self.goalstate.name, randomized=False)

    # update self.agentstate to a new state  ★
    def step(self, action):
        old_x, old_y = self.agentstate.pos[0], self.agentstate.pos[1]   # 获取当前state的位置
        new_x, new_y = old_x, old_y

        # windy effect  根据风向更新位置
        if new_x in [3, 4, 5, 8]:
            new_y += 1
        elif new_x in [6, 7]:
            new_y += 2

        # action   根据动作更新位置
        dx, dy = self.action_dict[action]   # 获得执行action动作后的行动方向 (*,*)
        new_x += dx
        new_y += dy

        # boundary restriction
        if new_x < 0:
            new_x = 0
        elif new_x >= 9:
            new_x = 9
        if new_y < 0:
            new_y = 0
        elif new_y >= 6:
            new_y = 6

        self.agentstate = state([new_x, new_y])   # 更新state。注意这里state都是用state(action.pos)表示的

        if self.is_done():
            self.agent_reward = 0   # 设置目标位置奖励为0
        else:
            self.agent_reward = -1   # 设置其他位置奖励为-1

        return self.agentstate.name, self.agent_reward   # 返回当前时刻的state和reward

    def is_done(self):
        return self.agentstate.equalTo(self.goalstate)


class state():
    def __init__(self, pos=[0, 3]):   # 初始化智能体位置
        self.pos = pos.copy()  # left bottom corner is (0,0)
        self.name = "X{0}-Y{1}".format(self.pos[0], self.pos[1])   # 点位置，如起点位置'X0-Y3'

    def equalTo(self, state):
        return self.pos[0] == state.pos[0] and self.pos[1] == state.pos[1]

