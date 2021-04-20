''''''

import tensorflow as tf 
import numpy as np 
import random
from collections import deque


# Hyper Parameters(超参)   构建全局变量
GAMMA = 0.99   # decay rate of past observations
OBSERVE = 100.   # timesteps to observe before training   表示让游戏自己运行100步
EXPLORE = 200000.   # frames over which to anneal(退火) epsilon   迭代次数
FINAL_EPSILON = 0.0001   # 0.001 # final value of epsilon
INITIAL_EPSILON = 0.01   # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000   # number of previous transitions to remember
BATCH_SIZE = 32   # size of minibatch
UPDATE_TIME = 100   # theta更新的步数

# 可以通过前置try来规范用语
# try:
#     tf.mul
# except:
#     # For new version of tensorflow tf.mul has been removed in new version of tensorflow
# 	# Using tf.multiply to replace tf.mul
#     tf.mul = tf.multiply

class BrainDQN:

	def __init__(self,actions):
		self.replayMemory = deque()   # init replay memory

		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.actions = actions

		# ①、init Q network 调用预测网络
		self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()
		# ②、init Target Q Network  调用目标网络 - 起始状态与Q network相同
		self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()
		# 复制参数，对目标网络的参数进行更新
		self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),
											self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]

		self.createTrainingMethod()

		self.session = tf.InteractiveSession()   # 启动交互式会话
		self.session.run(tf.initialize_all_variables())   # 全局变量初始化

		# saving and loading networks  ★
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_networks")   # 定义存储位置
		if checkpoint and checkpoint.model_checkpoint_path:   # 判定模型是否存在 ★
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)   # 读取模型
				print ("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
				print ("Could not find old network weights")


	"""构建网络"""
	def createQNetwork(self):   # 预测网络和目标网络使用相同的网络结构
		"""网络结构"""
		"""①、初始化网络参数"""
		W_conv1 = self.weight_variable([8,8,4,32])
		b_conv1 = self.bias_variable([32])

		W_conv2 = self.weight_variable([4,4,32,64])
		b_conv2 = self.bias_variable([64])

		W_conv3 = self.weight_variable([3,3,64,64])
		b_conv3 = self.bias_variable([64])

		W_fc1 = self.weight_variable([1600,512])
		b_fc1 = self.bias_variable([512])

		W_fc2 = self.weight_variable([512,self.actions])
		b_fc2 = self.bias_variable([self.actions])

		"""②、构建网络 - 输入图像，输出该图像的动作值函数，因为动作有两个，所以输出形式为[*,*]"""
		# 1、input layer
		stateInput = tf.placeholder("float",[None,80,80,4])   # 输入图像大小，黑白图通道=1，将四张图的通道叠加起来  (?, 80, 80, 4)
		# 2、cnn层
		net = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1)   # 步长为4，(?, 20, 20, 32)
		net = self.max_pool_2x2(net)   # (?, 10, 10, 32)
		net = tf.nn.relu(self.conv2d(net, W_conv2, 2) + b_conv2)   # (?, 5, 5, 64)
		net = tf.nn.relu(self.conv2d(net, W_conv3, 1) + b_conv3)   # (?, 5, 5, 64)

		# 3、全连接层
		net = tf.reshape(net,[-1,1600])   # 25*64 为一个样本的特征(像素)
		net = tf.nn.relu(tf.matmul(net, W_fc1) + b_fc1)
		# 4、out layer (Q Value layer)
		QValue = tf.matmul(net, W_fc2) + b_fc2   # 可以理解为两个动作的分值[分值1，分值2]

		# weight = [W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2]
		return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2   # 输入图像 + 输出Q值 + 权重


	def copyTargetQNetwork(self):   # 将main net 的参数赋予 target net
		self.session.run(self.copyTargetQNetworkOperation)   # 批量运行assign更新命令

	def createTrainingMethod(self):
		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None])

		# ①、tf.multiply(self.QValue, self.actionInput)，actionInput是one-hot结构数据的，所以点乘后得到的结果是，网络输出动作的价值
		# ②、tf.reduce_max(*,1)表示遍历每一行，获得行中最大值，实际上由于点乘的缘故，只有执行动作的动作值函数有值，其他都是0
		Q_Action = tf.reduce_max(tf.multiply(self.QValue, self.actionInput), axis = 1)   # QValue是当前UPDATE_TIME轮次下的 预测网络的输出值，  ★
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))   # MSE误差公式     yInput：预测值
		self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)   # 梯度下降优化器

	"""SGD网络训练"""
	def trainQNetwork(self):
		# Step 1: obtain random minibatch from replay memory  SGD法
		minibatch = random.sample(self.replayMemory,BATCH_SIZE)   # -》random.sample(*，num)：从replayMemory中随机选取BATCH_SIZE 32个样本 ★
		state_batch     = [data[0] for data in minibatch]
		action_batch    = [data[1] for data in minibatch]
		reward_batch    = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]
		terminal        = [data[4] for data in minibatch]

		QValueT_batch = self.session.run(self.QValueT, feed_dict={self.stateInputT:nextState_batch})   # 将下一状态s'传入目标网络中，输出目标Q值
		# QValueT_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})

		# Step 2: calculate y
		y_batch = []   # 批次Targer y 列表。存储的是当前样本条件下的y值
		for i in range(0,BATCH_SIZE):   # 依次遍历minibatch中的每一个样本
			if terminal[i]:   # 终止状态
				y_batch.append(reward_batch[i])   # 目标值计算公式1
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(QValueT_batch[i]))   # 目标值计算公式2

		# 运行梯度训练过程，优化网络
		self.session.run(self.trainStep, feed_dict={
			self.stateInput : state_batch,   # 经验池状态集s
			self.actionInput : action_batch,   # 经验是动作集a
			self.yInput : y_batch   # 目标Q值（由目标网络获得）。list型
			})


	""" observe + 存储 -> explore + 探索/开发 -> train + 开发 ； 保存 + 更新状态 """
	def setPerception(self, action, reward, nextObservation, terminal):   # 这里和源程序的顺序调整了一下，便于理解
		#newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
		newState = np.append(self.currentState[:,:,1:], nextObservation,axis = 2)   # 更新状态。 最左边一列(最左边的图像)被排去，右补一张图像  ★★★
		self.replayMemory.append((self.currentState, action, reward, newState, terminal))   # 存储一组状态数据
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()   # 缓存超出限定50000个，从左开始删。这也是一开始为什么用deque声明replayMemory的原因 ★
		if self.timeStep > OBSERVE:
			self.trainQNetwork()   # 步数超出观测限定100个，开始训练网络。样本数没存满则不会开始训练

		# print info
		state = ""
		if self.timeStep <= OBSERVE:
			state = "observe"
		elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"

		print ("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)

		# save network every 100000 iteration
		if self.timeStep % 10000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.timeStep)   # 存储session会话

		if self.timeStep % UPDATE_TIME == 0:
			self.copyTargetQNetwork()   # 每迭代100次SGD就对Target网络参数进行更新

		self.currentState = newState   # 更新状态(图像) - 在observe、explore、train过程中都是在不停增加样本，replayMemory也是在不停增大或更新
		self.timeStep += 1   # 步数+1

	"""选择动作"""
	def getAction(self):
		# 给 Q 网络传值，得到输出动作值函数。currentState是当前状态(图像)
		# 对value加[]是为了构建[*,*,*,*]的格式
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]   # 执行 Q 网络  <--- ★

		action = np.zeros(self.actions)   # 动作概率
		# 基于ε-贪婪策略 选择动作。使用0 1 动作概率
		if random.random() <= self.epsilon:   # 探索
			action_index = random.randrange(self.actions)
			action[action_index] = 1
		else:   # 开发
			action_index = np.argmax(QValue)   # 这个Qvalue是Q 网络的输出，作为动作选择策略   ★
			action[action_index] = 1

		# change episilon   OBSERVE和EXPLORE用在
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE   # epsilon持续减小，探索性能持续减少

		return action   # 返回动作概率list

	"""初始化状态(输入图像)、权重、偏置、卷积"""
	def setInitState(self,observation):
		self.currentState = np.stack((observation, observation, observation, observation), axis = 2)   # 将四幅图按照深度进行叠加。而在游戏刚开始时，将第一张图叠加四次

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial)   # 初始化权重参数

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

	def conv2d(self,x, W, stride):
		return tf.nn.conv2d(input = x, filter = W, strides = [1, stride, stride, 1], padding = "SAME")

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
		
