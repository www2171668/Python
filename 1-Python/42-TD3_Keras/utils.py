""""""
import numpy as np
import random
import gym
from gym import wrappers
import tensorflow as tf
from keras import backend as K

import os
import csv

def set_env(sess, args):
    seeds = args['random_seed']

    os.environ['PYTHONHASHSEED'] = str(seeds)
    np.random.seed(seeds)
    random.seed(seeds)
    tf.compat.v1.set_random_seed(seeds)
    K.set_session(sess)

    env = gym.make(args['env'])
    env.seed(seeds)
    env.action_space.seed(seeds)

    env_test = gym.make(args['env'])  # \ 测试环境
    env_test.seed(seeds)
    env_test.action_space.seed(seeds)

    return env, env_test

def get_dim(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print('action_space.shape', env.action_space.shape)
    print('observation_space.shape', env.observation_space.shape)
    action_bound = env.action_space.high

    assert (env.action_space.high[0] == -env.action_space.low[0])

    return state_dim, action_dim, action_bound

def record_video(env, args):
    if args['use_gym_monitor']:
        if not args['render_env']:
            env = wrappers.Monitor(env, args['monitor_dir'], video_callable=False, force=True)  # video_callable：是否记录 / force：覆盖旧文件
        else:
            env = wrappers.Monitor(env, args['monitor_dir'], video_callable=lambda episode_id: episode_id % 50 == 0, force=True)  # video_callable：每50个情节记录一次
    return env

def np_save(return_recoder, result_path, result_filename):
    try:
        import pathlib
        pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
    except:
        print("A result directory does not exist and cannot be created. The trial results are not saved")

    np.savetxt(result_filename, np.asarray(return_recoder))

def csv_save(return_dict, csv_name='data/experiments.csv'):
    if not os.path.exists(csv_name):
        with open(csv_name, 'w') as f:  # * 写入 w
            w = csv.DictWriter(f, list(return_dict.keys()))
            w.writeheader()         #* 换行，指针回到头部
            w.writerow(return_dict)
    else:
        with open(csv_name, 'a') as f:
            print(list(return_dict.keys()))
            w = csv.DictWriter(f, list(return_dict.keys()))   #　list(args.keys())是csv文件的标头列表
            w.writerow(return_dict)           #* 写入所有args信息

def model_save(agent, test_iter, model_path):
    try:
        import pathlib
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
    except:
        print("A model directory does not exist and cannot be created. The policy models are not saved")

    agent.save_model(iteration=test_iter, model_path=model_path)
    print('iteration_' + str(test_iter) +' Models saved.')

def build_summaries():
    episode_reward = tf.placeholder(dtype=tf.int64, shape=[])
    tf.compat.v1.summary.scalar("Reward", episode_reward)

    summary_vars = [episode_reward]
    summary_ops = tf.compat.v1.summary.merge_all()

    return summary_ops, summary_vars
