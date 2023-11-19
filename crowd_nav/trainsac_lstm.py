import os
import sys
import gym
import torch
import shutil
import logging
import argparse
import configparser

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from past.builtins import raw_input
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.utils_sac import rotate
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState

dirPath = os.path.dirname(os.path.realpath(__file__))


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='rnnsac')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--deterministic', default=False, action='store_true')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducing")
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument("--env_id", type=str, default="CrowdSim-v0", help="Environment Id")
    parser.add_argument('--log_path', type=str, default='data/log/', help="Directory to save logs")
    args = parser.parse_args()
    # configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        # key = raw_input('Output directory already exists! Overwrite the folder? (y/n)')
        key = 'y'
        if key == 'y' and not args.resume:
            shutil.rmtree(args.output_dir)
            shutil.rmtree(args.log_path)
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
    base_dir = args.log_path + args.env_id + "/saclstm_exp{}".format(args.seed)
    writer = SummaryWriter(base_dir)
    # configure logging
    mode = 'a' if args.resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
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
        ax.set_xlabel('x(m)', fontsize=18, family="Times New Roman")
        ax.set_ylabel('y(m)', fontsize=18, family="Times New Roman")
        # plt.rcParams["font.family"] = "Times New Roman"
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.ion()
        env.ax = ax
        env.fig = fig
    robot = Robot(env_config, 'robot')
    logging.info('Using robot policy: %s', args.policy)
    # configure policy
    policy = policy_factory[args.policy](env.action_space.shape[0],
                                         env.state_space.shape[0] - 5 * env_config.getint('sim', 'human_num') + 3,
                                         policy_config.getint('sac_rl', 'capacity'),
                                         policy_config.getint('sac_rl', 'batchsize'))
    policy.configure(policy_config)
    robot.set_policy(policy)
    env.set_robot(robot)
    training_step = 0
    num_updates = 50000
    for episode in range(num_updates):
        done = False
        state = env.reset('train')

        tol_reward = 0
        state = JointState(robot.get_full_state(), state)
        state = rotate(state)

        while not done:
            action = policy.predict(state, args.deterministic)
            # action = robot.clip_action(action, env_config.getfloat('robot', 'v_pref'))
            ob, reward, done, info, _ = env.step(action)
            tol_reward += reward
            next_state = rotate(JointState(robot.get_full_state(), ob))
            policy.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

        if len(policy.replay_buffer) > 6000:
            q1_loss, q2_loss, pi_loss, alpha_loss = policy.optim_sac(training_step)
            writer.add_scalar("loss/pi_loss", scalar_value=pi_loss, global_step=training_step)
            writer.add_scalar("loss/alpha_loss", scalar_value=alpha_loss, global_step=training_step)
            writer.add_scalar("loss/q1_loss", scalar_value=q1_loss, global_step=training_step)
            writer.add_scalar("loss/q2_loss", scalar_value=q2_loss, global_step=training_step)
        writer.add_scalar("reward", scalar_value=tol_reward, global_step=training_step)
        logging.info('saclstm_episode:{}, reward:{}, memory size:{}, time:{},'
                     ' info:{}'.format(episode, ('%.2f' % tol_reward),
                                       len(policy.replay_buffer), ('%.2f' % env.global_time), info))
        if episode % args.save_interval == 0 or episode == num_updates - 1:
            policy.save_load_model('save', args.output_dir)
        training_step += 1


if __name__ == '__main__':
    main()
