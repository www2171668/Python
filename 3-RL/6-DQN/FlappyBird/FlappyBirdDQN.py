''''''

import numpy as np
import cv2
import sys
sys.path.append("game/")   # 新增系统地址 game/ 。可以理解为，当前文件夹下，和game/文件夹下都可以直接用import name来获取包 ★
import wrapped_flappy_bird as game   # 读取环境
# from BrainDQN_NIPS import BrainDQN
from BrainDQN_Nature import BrainDQN


# preprocess raw image to 80*80 gray image
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)   # 转换图片大小，并灰度化
	_, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)   # 获得观察值
	return np.reshape(observation,(80,80,1))   # 增加维度(深度)，方便合并

"""主函数"""
def playFlappyBird():
	# Step 1: init BrainDQN
	actions = 2   # 两个动作
	brain = BrainDQN(actions)   # 调用类，实现 初始参数设置 + 网络构建 + 存储读取设置
	# Step 2: init Flappy Bird Game
	flappyBird = game.GameState()   # 读取环境
	# Step 3: play game
	# Step 3.1: obtain init state
	"""
		-》cv2.cvtColor(img, color_type)：设置图像格式   ★★★
			color_type参数的 输入 不管是cv2.COLOR_BGR2RGB、cv2.COLOR_BGR2GRAY，或是其他 颜色转换空间（color space conversion），均是 int 型数据
		
		-》cv2.threshold(src, thresh, maxval, type[, dst]) → retval, dst：将一幅灰度图二值化
			src：图片源，必须是灰度图
			thresh：阈值（起始值），将灰度图中灰度值小于thresh的点置0，灰度值大于thresh的点置255
			maxval：最大值
			type：在划分的时候使用的是什么的算法，常用值为0（cv2.THRESH_BINARY 图像二值化）

			ret:设定的thresh阈值
			dst：二值化的图像
	"""
	action0 = np.array([1,0])   # 初始化动作概率
	observation0, reward0, terminal = flappyBird.frame_step(action0)   # 获得 当前观测值(图像) + 即时奖励 + 是否结束(done)

	# 图像预处理
	observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)   # 图像大小重设 + 灰化 BGR->GRAY
	_, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)   # 灰度处理
	brain.setInitState(observation0)   # 图像叠加处理

	# Step 3.2: run the game
	while 1!= 0:
		action = brain.getAction()   # 基于Q网络，通过ε-贪婪策略获得动作概率。  因为self.currentState的定义和更新都在Brain_Nature中，所以这一片都没有state传值
		nextObservation,reward,terminal = flappyBird.frame_step(action)   # 通过动作概率获得 下一时刻观测值(图像) + 即时奖励 + 是否结束(done)
		nextObservation = preprocess(nextObservation)   # 图像预处理
		brain.setPerception(action,reward,nextObservation,terminal)   # 输入 当前动作 + 即时奖励 + 下一状态 + 是否结束(done)

def main():
	playFlappyBird()

if __name__ == '__main__':
	main()
