import os
import random
import sys
import gym
import torch
import shutil
import logging
import argparse
import configparser

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
dirPath = os.path.dirname(os.path.realpath(__file__))

from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from crowd_nav.policy.policy_factory import policy_factory
from utils.utils_sac import rotate
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument("--env_id", type=str, default="CrowdSim-v0", help="Environment Id")
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--log_path', type=str, default='log/', help="Directory to save logs")
    parser.add_argument('--policy', type=str, default='tf_rnn_sac')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--gpu', default=True, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument("--env_name", default="CrowdSim-v0")  # OpenAI gym environment name
    parser.add_argument('--batch_size', default=128, type=int)  # mini batch size
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--deterministic', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    # parser.add_argument("--seed", type=int, default=1, help="Seed for reproducing")
    args = parser.parse_args()
    # configure paths
    make_new_dir = True

    if os.path.exists(args.output_dir):
        # key = raw_input('Output directory already exists! Overwrite the folder? (y/n)')
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
    # base_dir = args.log_path + args.env_id + "/SAC_exp{}".format(args.seed)
    # print(base_dir)
    # writer = SummaryWriter(base_dir)
    # configure logging
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:3" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    env = gym.make(args.env_id)
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

    robot = Robot(env_config, 'robot')
    logging.info('Using robot policy: %s', args.policy)
    logging.info('Human policy: %s', env_config.get('humans', 'policy'))
    # configure policy [int(x) for x in config.get('sac_rnn_tf', 'in_mlp_dims').split(', ')]

    action_space = env.action_space.shape[0]
    capacity = policy_config.getint('sac_rnn_tf', 'capacity')
    batchsize = policy_config.getint('sac_rnn_tf', 'batchsize')
    in_mlp_dims = [int(x) for x in policy_config.get('sac_rnn_tf', 'in_mlp_dims').split(', ')]
    action_dims = [int(x) for x in policy_config.get('sac_rnn_tf', 'action_dims').split(', ')]
    value_dims = [int(x) for x in policy_config.get('sac_rnn_tf', 'value_dims').split(', ')]
    policy = policy_factory[args.policy](action_space, capacity, batchsize,
                                         in_mlp_dims, action_dims, value_dims, device)

    policy.configure(policy_config)
    robot.set_policy(policy)
    # policy.set_device(device)
    env.set_robot(robot)
    training_step = 0
    num_updates = int(1e5)
    size0 = env_config.getint('sim', 'human_num')
    size1 = int(in_mlp_dims[-1])

    for episode in range(num_updates):
        ob = env.reset(phase='train')
        done = False
        tol_reward = 0

        state = JointState(robot.get_full_state(), ob)
        state = rotate(state)
        h = torch.zeros([1, size0, size1])

        while not done:
            action, h = policy.get_action(state, h, args.deterministic)

            next_state, reward, done, info, _ = env.step(action)
            tol_reward += reward
            next_state = rotate(JointState(robot.get_full_state(), next_state))

            policy.replay_pool.push(state, action, reward, next_state, done)
            state = next_state

        q1_loss, q2_loss, pi_loss, alpha_loss = policy.optimize(training_step, h)
        # writer.add_scalar("loss/pi_loss", scalar_value=pi_loss, global_step=training_step)
        # writer.add_scalar("loss/alpha_loss", scalar_value=alpha_loss, global_step=training_step)
        # writer.add_scalar("loss/q1_loss", scalar_value=q1_loss, global_step=training_step)
        # writer.add_scalar("loss/q2_loss", scalar_value=q2_loss, global_step=training_step)
        # writer.add_scalar("reward", scalar_value=tol_reward, global_step=training_step)
        logging.info('episode:{}, reward:{}, memory size:{}, time:{},'
                     ' info:{}'.format(episode, ('%.2f' % tol_reward),
                                       policy.replay_pool.__len__(), ('%.2f' % env.global_time), info))
        if episode % args.save_interval == 0 or episode == num_updates - 1:
            policy.save_load_model('save', args.output_dir)
        training_step += 1
    print(training_step, "episodes are training over!")


if __name__ == '__main__':
    main()
