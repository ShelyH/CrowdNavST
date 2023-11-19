#!/usr/bin/env python2
# coding=utf-8
import argparse
import configparser
import logging
import os
import random
import sys

import gym
import torch
from matplotlib import pyplot as plt

from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.info import *
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.robot import Robot

sys.path.append('../..')


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy', type=str, default='sac_rl')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--num_steps', type=int, default=14)
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--model_dir', type=str, default='../data/output')
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--deterministic', default=True, action='store_true')
    parser.add_argument('--test_case', type=int, default=None)
    args = parser.parse_args()
    # if args.visualize:
    #     fig, ax = plt.subplots(figsize=(9, 8))
    #     ax.set_xlim(-7, 7)
    #     ax.set_ylim(-7, 7)
    #     ax.set_xlabel('x(m)', fontsize=18, family="Times New Roman")
    #     ax.set_ylabel('y(m)', fontsize=18, family="Times New Roman")
    #     # plt.rcParams["font.family"] = "Times New Roman"
    #     labels = ax.get_xticklabels() + ax.get_yticklabels()
    #     [label.set_fontname('Times New Roman') for label in labels]
    #     plt.ion()
    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        # print(env_config_file)
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
    else:
        env_config_file = args.env_config
        policy_config_file = args.policy_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    load_path = args.model_dir
    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    # env.ax = ax
    # env.fig = fig
    env.configure(env_config)
    # configure policy
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy = policy_factory[args.policy](env.action_space.shape[0],
                                         env.state_space.shape[0],
                                         policy_config.getint('sac_rl', 'capacity'),
                                         policy_config.getint('sac_rl', 'batchsize'))

    policy.configure(policy_config)
    # if policy.trainable:
    #     if args.model_dir is None:
    #         parser.error('Trainable policy must be specified with a model weights directory')
    #     policy.get_model().load_state_dict(torch.load(model_weights))
    policy.save_load_model('load', load_path)

    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    policy.set_phase(args.phase)
    # policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()
    success_rate = 0
    collision_rate = 0
    timeout_rate = 0
    for k in range(1, 1000):
        args.test_case = random.randint(0, 500)
        ob = env.reset(args.phase, args.test_case)
        done = False
        # print(robot.policy)
        while not done:
            action = robot.act(ob, args.deterministic)
            # print(action)
            # actiontoObj = ActionXY(action[0][0], action[0][1])
            # actiontoObj = robot.clip_action(action, env_config.getfloat('robot', 'v_pref'))
            # print(ob)
            ob, reward, done, info, _ = env.step(action)
        env.render(update=False)
        if isinstance(info, ReachGoal):
            success_rate += 1
        elif isinstance(info, Collision):
            collision_rate += 1
        elif isinstance(info, Timeout):
            timeout_rate += 1
        # print(type(info))
        else:
            raise ValueError('Invalid end signal from environment')
        logging.info(
            '%s, It takes %.2f seconds to finish. Success rate: %.2f, Collision rate: %.2f, Timeout rate: %.2f. '
            'Final status is %s', k, env.global_time, success_rate / k, collision_rate / k, timeout_rate / k, info)

if __name__ == '__main__':
    main()
