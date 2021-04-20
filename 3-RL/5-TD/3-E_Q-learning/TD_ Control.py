'''TD-Agents里包括TD-Prediction'''

from __future__ import print_function
from __future__ import division

import argparse

from TD_Agents import TDAgent
from envs import GridWorld


def main(args):
    # ①、引入环境
    env = GridWorld()
    # ②、引入模型参数
    agent = TDAgent(env, gamma=args.discout, epsilon=args.epsilon, alpha=0.05, lamda=0.7)  # 通过 args.* 调用argparse中的参数 default
    agent.control(method=args.algorithm)  # 执行算法


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='sarsa', help='(可以选择qlearn，或者sarsa)')  # 可选方法有 qlearn 和 sarsa
    parser.add_argument('--discout', default=0.9, help='discout factor', type=float)  # 折扣系数
    parser.add_argument('--epsilon', default=0.5, help='parameter of epsilon greedy policy', type=float)  # ε

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parse()
    main(args)
