import logging
import random
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.append('..')
# import logging
import argparse
import configparser
import os
import shutil
import torch
import gym
# import git

from past.builtins import raw_input
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# logging = Log(__name__).getlog()


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='lstm_rl')
    parser.add_argument('--model_dir', type=str, default='crossTF99.4/output')
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=random.randint(0, 500))
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_xlabel('x(m)', fontsize=18, family="Times New Roman")
    ax.set_ylabel('y(m)', fontsize=18, family="Times New Roman")
    # plt.rcParams["font.family"] = "Times New Roman"
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.ion()
    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
            print('use il_model')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
                print('use resumed_rl_model')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
                # print('use lstm_rl')
    else:
        env_config_file = args.env_config
        policy_config_file = args.policy_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    env.ax = ax
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)

    policy.set_env(env)
    robot.print_info()
    if args.visualize:
        # print('visualize')
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())
        # print(robot.policy)
        while not done:
            action = robot.act(ob, True)
            # print(action)
            ob, _, done, info, _ = env.step(action)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
            # env.render()
            env.render(update=True)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
    else:
        print('no visualize')
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)


if __name__ == '__main__':
    main()
