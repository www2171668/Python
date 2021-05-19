import tensorflow as tf

import numpy as np
import gym
from gym import wrappers

from replay_buffer import ReplayBuffer

import argparse
import pprint as pp

from TD3_keras_agent import TD3
from utils import *

import logging

logging.basicConfig(level=logging.CRITICAL)  # 设置日志输出级别，控制debg、info、warning（默认）、error、critical五个级别

# logging.basicConfig(filename='demo.log', filemode='w', level=logging.debg) # 以w重写的形式，将debug等信息输入在log文件中
# logging.debug('this is debug')

def update_policy(sess, env, env_test, args, agent, replay_buffer, action_noise, update_num):
    for update_ite in range(update_num):
        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(args['minibatch_size'])  # \ 一个批次100个数据

        # %% 执行目标actor网络
        noise_clip = 0.5
        noise = np.clip(np.random.normal(0, action_noise, size=(args['minibatch_size'], agent.action_dim)), -noise_clip, noise_clip)
        next_action_batch = agent.predict_actor_target(s2_batch) + noise
        next_action_batch = np.clip(next_action_batch, -agent.action_bound, agent.action_bound)

        # %% 执行目标critic网络
        target_Q1, target_Q2 = agent.predict_critic_target(s2_batch, next_action_batch)
        min_q = np.minimum(target_Q1, target_Q2)

        # %% 计算更新目标值 y_i
        condition = (t_batch == 1)
        min_q[condition] = 0  # 终止状态下的 Q=0
        y_i = np.reshape(r_batch, (args['minibatch_size'], 1)) + agent.gamma * np.reshape(min_q, (args['minibatch_size'], 1))
        # print('y_i', y_i.shape)

        # %% 更新critic
        agent.train_critic(s_batch,
                           a_batch,
                           np.reshape(y_i, (args['minibatch_size'], 1)))

        # %% 更新actor
        if update_ite % args['policy_freq'] == 0:
            a_outs = agent.predict_actor(s_batch)
            grads = agent.action_gradients(s_batch, a_outs)[0]
            agent.train_actor(s_batch, grads)

            # %% 更新目标网络
            agent.update_actor_target_network()
            agent.update_critic_target_network()

# ===========================
#   Agent Training
# ===========================
def train(sess, env, env_test, args, agent):
    # %% summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # %% 初始化
    # Initialize target network weights
    agent.update_actor_target_network()
    agent.update_critic_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(args['buffer_size'], args['random_seed'])

    total_step_cnt = 0
    test_iter = 0
    epi_cnt = 0
    return_test = np.zeros((np.ceil(args['total_step_num'] / args['sample_step_num']).astype('int') + 1))  # \ 平均回报

    result_name = 'TD3_' + args['env'] + '_trial_idx_' + str(args['ite'])
    action_noise = args['action_noise']
    trained_times_steps = 0
    save_cnt = 1

    # %% 训练
    while total_step_cnt in range(args['total_step_num']):  # 训练总次数，1000000
        state = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        T_end = False

        for j in range(args['max_episode_len']):  # 最大步数，1000。测试1000步内的回报
            if args['render_env']:
                env.render()

            # %% 选择动作
            if total_step_cnt < 1000:  # 随机动作
                action = env.action_space.sample()
            else:
                # actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
                action = agent.predict_actor(np.reshape(state, (1, agent.state_dim)))  # + actor_noise()  # 没有噪音的话，输出最优策略
                # print(total_step_cnt, action)
                clipped_noise = np.clip(np.random.normal(0, action_noise, size=env.action_space.shape[0]), -0.5, 0.5)  # 截断噪音
                action = (action + clipped_noise).clip(env.action_space.low, env.action_space.high)

            state2, reward, terminal, _ = env.step(action)  # [0]
            # print(total_step_cnt, terminal)

            # %% 存储经验
            replay_buffer.add(np.reshape(state, (agent.state_dim,)),
                              np.reshape(action, (agent.action_dim,)),
                              reward,
                              terminal,
                              np.reshape(state2, (agent.state_dim,)))

            if j == args['max_episode_len'] - 1:  # 单次情节最大步数
                T_end = True

            state = state2
            ep_reward += reward
            total_step_cnt += 1

            # %% 评估阶段：用确定型策略   每5000步记录一次
            if total_step_cnt >= test_iter * args['sample_step_num'] or total_step_cnt == 1:  # \ 每5000步记录一次
                print('total_step_cnt', total_step_cnt)
                for nn in range(args['test_num']):  # \ 测试10次
                    state_test = env_test.reset()
                    return_epi_test = 0
                    for _ in range(args['max_episode_len']):  # \ 1000步
                        action_test = agent.predict_actor(np.reshape(state_test, (1, agent.state_dim)))
                        state_test2, reward_test, terminal_test, _ = env_test.step(action_test) # [0]
                        state_test = state_test2
                        return_epi_test += reward_test
                        if terminal_test:
                            break

                    # print('test_iter:{:d}, nn:{:d}, return_epi_test: {:d}'.format(int(test_iter), int(nn), int(return_epi_test)))
                    return_test[test_iter] = return_test[test_iter] + return_epi_test / args['test_num']

                print('return_test[{:d}] {:d}'.format(int(test_iter), int(return_test[test_iter])))
                result = sess.run(summary_ops, feed_dict={summary_vars[0]: int(return_test[test_iter])})
                writer.add_summary(summary=result, global_step=int(total_step_cnt))

                test_iter += 1

            # %% 保存模型
            if total_step_cnt > args['save_model_num'] * save_cnt:  # \ 每500000步存一次模型
                model_path = "./Model/td3/" + args['env'] + '/'
                model_save(agent, test_iter, model_path, result_name)
                save_cnt += 1

            # %% 终止当前情节
            if terminal or T_end:
                epi_cnt += 1
                break

        # %% 更新策略
        if total_step_cnt != args['total_step_num'] and total_step_cnt > 1000:  # 每训练完一个情节，按照步数长度更新策略
            update_num = total_step_cnt - trained_times_steps  # \ 控制策略更新次数
            trained_times_steps = total_step_cnt
            update_policy(sess, env, env_test, args, agent, replay_buffer, action_noise, update_num)

    return return_test

def main(args):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                                      gpu_options=gpu_options, log_device_placement=False)

    with tf.compat.v1.Session(config=config) as sess:
        # %% 种子 + 环境 + 维度获取
        rand_seed = set_seed(args['ite'], args)
        env, env_test = set_env(rand_seed, sess, args)
        state_dim, action_dim, action_bound = get_dim(env)

        # %% 模型加载
        agent = TD3(sess, env, state_dim, action_dim, action_bound,
                    args['minibatch_size'], tau=args['tau'],
                    actor_lr=args['actor_lr'], critic_lr=args['critic_lr'],
                    gamma=args['gamma'], hidden_dim=np.asarray(args['hidden_dim']))
        # agent.load_model()

        # %% 记录mp4文件（还不会用）
        env = record_video(env, args)

        # %% 训练
        step_R_i = train(sess, env, env_test, args, agent)  # return_test

        # %% 记录每一次测试的回报值(可能没用)
        result_path = "./results/tf_td3_trials/"
        result_filename = result_path + args['env'] + '_trial_idx_' + str(args['ite']) + '.txt'
        np_save(args['ite'], step_R_i, result_path, result_filename, args)

        # %% 关闭渲染
        if args['use_gym_monitor']:
            env.monitor.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for TD3 agent')

    # %% agent parameters
    parser.add_argument('--actor_lr', type=float, help='actor network learning rate', default=0.001)
    parser.add_argument('--critic_lr', type=float, help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', type=float, help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', type=float, help='soft target update parameter', default=0.005)
    parser.add_argument('--buffer_size', type=int, help='max size of the replay buffer', default=1000000)
    parser.add_argument('--hidden_dim', type=np.asarray, help='max size of the hidden layers', default=(400, 300))

    parser.add_argument('--minibatch_size', type=int, help='size of minibatch for minibatch-SGD', default=100)
    parser.add_argument('--total_step_num', type=int, help='total number of time steps', default=1000000)
    parser.add_argument('--sample_step_num', type=int, help='number of time steps for recording the return', default=5000)
    parser.add_argument('--test_num', type=int, help='number of episode for recording the return', default=10)
    parser.add_argument('--action_noise', type=float, help='parameter of the noise for exploration', default=0.2)

    parser.add_argument('--policy_freq', type=int, help='frequency of updating the policy', default=2)

    # %% run parameters
    parser.add_argument('--ite', type=int, default=0, help='训练编号，在shell里改变')
    # parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}')
    parser.add_argument('--env_id', type=int, default=0, help='choose the gym env- tested on {Pendulum-v0}')
    parser.add_argument('--random_seed', type=int, help='random seed for repeatability', default=1234)
    parser.add_argument('--max_episode_len', type=int, help='max length of 1 episode', default=1000)
    parser.add_argument('--render_env', help='render the gym env', action='store_true')
    parser.add_argument('--use_gym_monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor_dir', help='directory for storing gym results', default='./Video/td3')
    parser.add_argument('--summary_dir', help='directory for storing tensorboard info', default='./results/tf_td3')
    parser.add_argument('--overwrite_result', help='flag for overwriting the trial file', default=True)
    parser.add_argument('--trial_num', type=int, help='number of trials，即训练次数', default=1)
    parser.add_argument('--change_seed', help='change the random seed to obtain different results', default=False)
    parser.add_argument('--save_model_num', type=int, help='number of time steps for saving the network models', default=500000)

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    parser.set_defaults(change_seed=True)
    parser.set_defaults(overwrite_result=True)

    args_tmp = parser.parse_args()

    if args_tmp.env is None:
        env_dict = {0: "Pendulum-v0",
                    1: "InvertedPendulum-v1",
                    2: "InvertedDoublePendulum-v1",
                    3: "Reacher-v1",
                    4: "Swimmer-v1",
                    5: "Ant-v1",
                    6: "Hopper-v2",
                    7: "Walker2d-v1",
                    8: "HalfCheetah-v1",
                    9: "Humanoid-v1",
                    10: "HumanoidStandup-v1",
                    11: "MountainCarContinuous-v0"
                    }
        args_tmp.env = env_dict[args_tmp.env_id]
    args = vars(args_tmp)

    pp.pprint(args)

    main(args)
