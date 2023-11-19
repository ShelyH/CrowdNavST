#!/usr/bin/env python
# coding=utf-8
import argparse
import configparser
import logging
import os
import shutil
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import gym
import torch
from matplotlib import pyplot as plt

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from utils.kexplorer import kexplorer


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--output_dir', type=str, default='data/output/')
    parser.add_argument('--log_dir', type=str, default='data/output')
    parser.add_argument("--env_id", type=str, default="CrowdSim-v0", help="Environment Id")
    parser.add_argument('--log_path', type=str, default='log/', help="Directory to save logs")
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy', type=str, default='sac_rl')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--num_steps', type=int, default=14)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--multiagent_training', default=True, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument("--seed", type=int, default=1, help="Seed for reproducing")
    parser.add_argument('--deterministic', default=False, action='store_true')
    args = parser.parse_args()
    # configure paths

    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        # key = input('Output directory already exists! Overwrite the folder? (y/n)')
        key = 'y'
        if key == 'y' and not args.resume:
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
            args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
    if make_new_dir:
        os.makedirs(args.output_dir)
        shutil.copy(args.env_config, args.output_dir)
        shutil.copy(args.policy_config, args.output_dir)
        shutil.copy(args.train_config, args.output_dir)
    log_file = os.path.join(args.output_dir, 'output.log')

    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    # print(device)
    logging.info('Using device: %s', device)
    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.visualize:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('x(m)', fontsize=18, family="Times New Roman")
        ax.set_ylabel('y(m)', fontsize=18, family="Times New Roman")
        # plt.rcParams["font.family"] = "Times New Roman"
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.ion()
        env.ax = ax
    robot = Robot(env_config, 'robot')
    logging.info('Using robot policy: %s', args.policy)
    logging.info('Human policy: %s', env_config.get('humans', 'policy'))
    policy = policy_factory[args.policy](env.action_space.shape[0],
                                         env.state_space.shape[0],
                                         policy_config.getint('sac_rl', 'capacity'),
                                         policy_config.getint('sac_rl', 'batchsize'))

    # imitation learning
    if args.il:
        train_config = configparser.RawConfigParser()
        train_config.read(args.train_config)
        il_policy = train_config.get('imitation_learning', 'il_policy')
        if robot.visible:
            safety_space = 0
        else:
            safety_space = train_config.getfloat('imitation_learning', 'safety_space')
        il_policy = policy_factory[il_policy]()
        # print(policy)
        il_policy.multiagent_training = env_config.getboolean('env', 'multiagent_training')
        il_policy.safety_space = safety_space
        robot.set_policy(il_policy)
        env.set_robot(robot)
        kexplorer(num_updates=500, is_rl=False, env=env, robot=robot,
                  policy=il_policy, env_config=env_config, args=args, up=policy)
    logging.info('Experience set size: {}'.format(policy.replay_pool.__len__()))
    logging.info('Finish imitation learning')

    policy.configure(policy_config)

    robot.set_policy(policy)
    logging.info('Current policy:{}'.format(args.policy))
    # print(policy)
    env.set_robot(robot)
    kexplorer(num_updates=int(1e6), is_rl=True, env=env, robot=robot,
              policy=policy, env_config=env_config, args=args, up=policy)


if __name__ == '__main__':
    main()
