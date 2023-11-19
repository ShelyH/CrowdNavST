#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：RL_study_
@File    ：lineplot.py
@Author  ：HHD
@Date    ：2022/2/23 下午9:04
"""
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams

sns.set()
rcParams['font.family'] = 'Times New Roman'


def running_mean(data, n):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def set_ppodata(args):
    l1_train_pattern1 = r"LSTM in episode (.*) has success rate: (.*), collision rate: (.*), nav time: (.*), " \
                        r"total reward: (.*),V_MeanSquareError:(.*),times:(.*)"

    l1_train_pattern2 = r"episode2:(.*), reward:(.*), memory size:(.*), time:(.*), info:(.*)"

    l2_train_pattern1 = r"GRU in episode (.*) has success rate: (.*), collision rate: (.*), nav time: (.*), " \
                        r"total reward: (.*),V_MeanSquareError:(.*),times:(.*)"

    l2_train_pattern2 = r" Updates2:(.*), num timesteps:(.*), FPS:(.*), Last:(.*), " \
                        r"training episodes mean/median reward:(.*)/(.*), min/max reward:(.*)/(.*)，nav_time:(.*)"

    l3_train_pattern1 = r"l3_episode1:(.*), reward:(.*), memory size:(.*), time:(.*), info:(.*)"
    l3_train_pattern2 = r"l3_episode2:(.*), reward:(.*), memory size:(.*), time:(.*), info:(.*)"

    l1_train_reward1 = []
    l1_train_reward2 = []

    l2_train_reward1 = []
    l2_train_reward2 = []

    l3_train_reward1 = []
    l3_train_reward2 = []

    max_episodes = 50000
    for _, log_file in enumerate(args.log_files):
        with open(log_file, 'r') as file:
            log = file.read()
        for r in re.findall(l1_train_pattern1, log):
            l1_train_reward1.append(float(r[1]))
        l1_train_reward1 = l1_train_reward1[:max_episodes]
        # smooth training plot
        l1_train_reward_smooth1 = running_mean(l1_train_reward1, args.window_size)
        for r in re.findall(l1_train_pattern2, log):
            l1_train_reward2.append(float(r[1]))
        l1_train_reward2 = l1_train_reward2[:max_episodes]
        l1_train_reward_smooth2 = running_mean(l1_train_reward2, args.window_size)

        for r in re.findall(l2_train_pattern1, log):
            l2_train_reward1.append(float(r[4]))
        l2_train_reward1 = l2_train_reward1[:max_episodes]
        l2_train_reward_smooth1 = running_mean(l2_train_reward1, args.window_size)
        for r in re.findall(l2_train_pattern2, log):
            l2_train_reward2.append(float(r[4]))
        l2_train_reward2 = l2_train_reward2[:max_episodes]

        l2_train_reward_smooth2 = running_mean(l2_train_reward2, args.window_size)
        # print(np.var(train_reward_smooth1[2500:]), np.var(train_reward_smooth2[2500:]))
        for r in re.findall(l3_train_pattern1, log):
            l3_train_reward1.append(float(r[1]))
        l3_train_reward1 = l3_train_reward1[:max_episodes]
        l3_train_reward_smooth1 = running_mean(l3_train_reward1, args.window_size)
        for r in re.findall(l3_train_pattern2, log):
            l3_train_reward2.append(float(r[1]))
        l3_train_reward2 = l3_train_reward2[:max_episodes]

        l3_train_reward_smooth2 = running_mean(l3_train_reward2, args.window_size)

    return l1_train_reward_smooth1, l1_train_reward_smooth2, \
        l2_train_reward_smooth1, l2_train_reward_smooth2, \
        l3_train_reward_smooth1, l3_train_reward_smooth2


def get_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+')
    parser.add_argument('--window_size', type=int, default=500)
    args = parser.parse_args()
    l1data1, l1data2, l2data1, l2data2, l3data1, l3data2 = set_ppodata(args)

    return l1data1, l1data2, l2data1, l2data2, l3data1, l3data2


def main():
    l1data1, l1data2, l2data1, l2data2, l3data1, l3data2 = get_data()
    # print(reward1)
    # print(reward2)

    # l1_rewards = np.concatenate((l1data1, l1data2))
    # l1_episode1 = range(len(l1data1))
    # l1_episode2 = range(len(l1data2))
    # l1_episode = np.concatenate((l1_episode1, l1_episode2))
    # sns.lineplot(x=l1_episode, y=l1_rewards, )
    #
    l2_rewards = np.concatenate((l2data1, l2data2))
    l2_episode1 = range(len(l2data1))
    l2_episode2 = range(len(l2data2))
    l2_episodes = np.concatenate((l2_episode1, l2_episode2))
    sns.lineplot(x=l2_episodes, y=l2_rewards)
    #
    # l3_rewards = np.concatenate((l3data1, l3data2))
    # l3_episode1 = range(len(l3data1))
    # l3_episode2 = range(len(l3data2))
    # l3_episodes = np.concatenate((l3_episode1, l3_episode2))
    # sns.lineplot(x=l3_episodes, y=l3_rewards)

    legend_font = {"family": "Times New Roman"}

    plt.xlabel("Episodes", fontproperties='Times New Roman')
    plt.ylabel("Reward", fontproperties='Times New Roman')
    # plt.title("Cumulative Discounted Reward", fontproperties='Times New Roman')
    plt.legend(labels=["LSTM", "GRU", ""], loc=4, prop=legend_font)
    plt.show()


if __name__ == '__main__':
    # get_data()
    main()
