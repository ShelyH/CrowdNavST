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

rcParams['font.family'] = 'Times New Roman'
sns.set()


def running_mean(data, n):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def get_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+')
    parser.add_argument('--window_size', type=int, default=1000)
    args = parser.parse_args()
    train_pattern = r"LSTM in episode (.*) has success rate: (.*), collision rate: (.*), nav time: (.*), " \
                    r"total reward: (.*),V_MeanSquareError:(.*),times:(.*)"

    new_train_pattern = r"GRU in episode (.*) has success rate: (.*), collision rate: (.*), nav time: (.*), " \
                        r"total reward: (.*),V_MeanSquareError:(.*),times:(.*)"

    train_reward = []
    train_reward2 = []
    max_episodes = 1000000
    for _, log_file in enumerate(args.log_files):
        with open(log_file, 'r') as file:
            log = file.read()
        # print(log)
        for r in re.findall(train_pattern, log):
            train_reward.append(float(r[4]))
        # print(train_reward)
        train_reward = train_reward[:max_episodes]
        # smooth training plot

        train_reward_smooth1 = running_mean(train_reward, args.window_size)
        # print(train_reward_smooth1)
        for r in re.findall(new_train_pattern, log):
            train_reward2.append(float(r[4]))
        train_reward2 = train_reward2[:max_episodes]

        train_reward_smooth2 = running_mean(train_reward2, args.window_size)
    # print(np.var(train_reward_smooth1[2500:]), np.var(train_reward_smooth2[2500:]))
    return train_reward_smooth1, train_reward_smooth2


def main():
    reward1, reward2 = get_data()
    # print(reward1)
    # print(reward2)
    rewards = np.concatenate((reward1, reward2))
    episode1 = range(len(reward1))
    episode2 = range(len(reward2))
    episode = np.concatenate((episode1, episode2))
    sns.lineplot(x=episode, y=rewards)
    plt.xlabel("Episodes", fontdict={'family': 'Times New Roman', 'size': 15})
    plt.ylabel("Reward", fontdict={'family': 'Times New Roman', 'size': 15})
    legend_font = {"family": "Times New Roman"}
    # plt.ylabel("Reward")
    plt.title("Cumulative Discounted Reward", fontdict={'family': 'Times New Roman', 'size': 15})

    plt.legend(labels=["LSTM", "GRU", ""], loc=4, prop=legend_font)
    plt.show()


if __name__ == '__main__':
    # get_data()
    main()
