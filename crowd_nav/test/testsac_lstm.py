import logging
import argparse
import configparser
import os

import numpy as np
import torch

import gym
from matplotlib import pyplot as plt
from crowd_nav.utils.info import *

from crowd_sim.envs.utils.state import JointState
from crowd_nav.utils.utils_sac import rotate
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
import random


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='rnnsac')
    parser.add_argument('--model_dir', type=str, default='../data/output')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument('--env_update', default=True, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--deterministic', default=True, action='store_true')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
    else:
        env_config_file = args.env_config
        policy_config_file = args.policy_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.visualize:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        ax.set_xlabel('x(m)', fontsize=12, family="Times New Roman")
        ax.set_ylabel('y(m)', fontsize=12, family="Times New Roman")
        # plt.rcParams["font.family"] = "Times New Roman"
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.ion()
        env.ax = ax
        env.fig = fig
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy = policy_factory[args.policy](env.action_space.shape[0],
                                         env.state_space.shape[0] - 5 * env_config.getint('sim', 'human_num') + 3,
                                         policy_config.getint('sac_rl', 'capacity'),
                                         policy_config.getint('sac_rl', 'batchsize'))

    policy.configure(policy_config)

    load_path = args.model_dir
    policy.save_load_model('load', load_path)

    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    # policy.set_phase(args.phase)
    # policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need.
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    # policy.set_env(env)
    robot.print_info()
    success_rate = 0
    collision_rate = 0
    globaltime = 0
    total_reward = []
    timeout_rate = 0
    for k in range(1, 501):
        args.test_case = random.randint(0, 500)
        ob = env.reset(args.phase, args.test_case)
        done = False
        state = JointState(robot.get_full_state(), ob)
        state = rotate( state)
        tol_reward = 0
        while not done:
            action = policy.predict(state, args.deterministic)
            ob, reward, done, info, _ = env.step(action)
            tol_reward += reward
            next_state = JointState(robot.get_full_state(), ob)
            next_state = rotate( next_state)
            state = next_state
            # del env.disfromhuman[:]
            if args.visualize and args.env_update:
                env.render(update=args.env_update)
        if args.visualize and not args.env_update:
            env.render()
        if isinstance(info, ReachGoal):
            success_rate += 1
        elif isinstance(info, Collision):
            collision_rate += 1
        elif isinstance(info, Timeout):
            timeout_rate += 1
        # print(type(info))
        else:
            raise ValueError('Invalid end signal from environment')
        globaltime += env.global_time
        total_reward.append(tol_reward)
        print(
            '%s, It takes %.2f seconds to finish. Meannavtime:%.3f, Success rate: %.4f, Collision rate: %.4f, '
            'Timeout rate: %.4f. Total reward :%.2f. Final status is %s' % (k, env.global_time, globaltime / k, success_rate / k,
                                                                            collision_rate / k, timeout_rate / k,
                                                                            tol_reward, info))
    print(np.mean(total_reward))


if __name__ == '__main__':
    main()
