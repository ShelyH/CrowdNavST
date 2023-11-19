import os
import os
import random
import sys
import gym
import numpy as np
import torch
import shutil
import logging
import argparse
import configparser

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
dirPath = os.path.dirname(os.path.realpath(__file__))
from crowd_nav.utils.info import *
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument("--env_id", type=str, default="CrowdSimstemporal-v0", help="Environment Id")
    parser.add_argument('--policy', type=str, default='PER_STRL')
    parser.add_argument('--model_dir', type=str, default='data/output/')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--gpu', default=True, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument("--env_name", default="CrowdSimstemporal-v0")  # OpenAI gym environment name
    parser.add_argument('--deterministic', default=True, action='store_true')
    parser.add_argument('--update', default=True, action='store_true')
    parser.add_argument('--visualize', default=True, action='store_true')

    args = parser.parse_args()
    # configure paths

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
    else:
        env_config_file = args.env_config
        policy_config_file = args.policy_config

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)

    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    env = gym.make(args.env_id)
    env.configure(env_config)
    if args.visualize and args.update:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_xlim(-4.8, 4.8)
        ax.set_ylim(-4.8, 4.8)
        ax.set_xlabel('x(m)', fontsize=16, family="Times New Roman")
        ax.set_ylabel('y(m)', fontsize=16, family="Times New Roman")
        # plt.rcParams["font.family"] = "Times New Roman"
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.ion()
        env.ax = ax
        env.fig = fig
        plt.yticks(fontproperties='Times New Roman', size=16)
        plt.xticks(fontproperties='Times New Roman', size=16)

    robot = Robot(env_config, 'robot')
    logging.info('Using robot policy: %s', args.policy)

    action_space = env.action_space.shape[0]
    capacity = policy_config.getint('sac_rnn_tf', 'capacity')
    batchsize = policy_config.getint('sac_rnn_tf', 'batchsize')
    in_mlp_dims = [int(x) for x in policy_config.get('sac_rnn_tf', 'in_mlp_dims').split(', ')]
    action_dims = [int(x) for x in policy_config.get('sac_rnn_tf', 'action_dims').split(', ')]
    value_dims = [int(x) for x in policy_config.get('sac_rnn_tf', 'value_dims').split(', ')]
    policy = policy_factory[args.policy](action_space, capacity, batchsize,
                                         in_mlp_dims, action_dims, value_dims, device)
    policy.save_load_model('load', args.model_dir)
    policy.configure(policy_config)
    robot.set_policy(policy)

    env.set_robot(robot)
    training_step = 0

    success_rate = 0
    collision_rate = 0
    timeout_rate = 0
    total_reward = []
    globaltime = 0
    for episode in range(1, 501):
        time_frames_state = env.reset(phase='train', test_case=random.randint(0, 1000))
        done = False
        tol_reward = 0
        while not done:
            action = policy.get_action(time_frames_state, args.deterministic)
            time_frames_state_, reward, done, info = env.step(action)
            tol_reward += reward
            time_frames_state = time_frames_state_

            if args.visualize and args.update:
                env.render(update=args.update)

        if args.visualize and args.update is False:
            env.render(update=args.update)
            del env.disfromhuman[:]

        if isinstance(info, ReachGoal):
            success_rate += 1
        elif isinstance(info, Collision):
            collision_rate += 1
        elif isinstance(info, Timeout):
            timeout_rate += 1
        else:
            raise ValueError('Invalid end signal from environment')
        globaltime += env.global_time

        total_reward.append(tol_reward)
        print(
            '%s, It takes %.4f seconds to finish. Meannavtime:%.4f, Success rate: %.4f, Collision rate: %.4f, '
            'Timeout rate: %.4f. Final status is %s' % (
                episode, env.global_time, globaltime / episode, success_rate / episode, collision_rate / episode,
                timeout_rate / episode, info))

        training_step += 1
    print(np.mean(total_reward))


if __name__ == '__main__':
    main()
