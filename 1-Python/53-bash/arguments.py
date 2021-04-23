# -*-coding:utf-8-*-
import argparse

def Arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--aaa', default='a', type=str)       # actor更新 + 目标网络参数更新 下层控制器
    parser.add_argument('--bbb', default='b', type=str)      # 上层控制器

    return parser.parse_args()