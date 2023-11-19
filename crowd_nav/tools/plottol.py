#!/usr/bin/env python
# encoding: utf-8
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser()
parser.add_argument('--window_size', type=int, default=4000)
args = parser.parse_args()
rcParams['font.family'] = 'Times New Roman'


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def main():
    # define the names of the models you want to plot and the longest episodes you want to show
    max_episodes = 130000
    file1 = '../dataout/crossTF97%/output/output.log'
    file2 = '../dataout/crossTF99.4%/output/output.log'
    file3 = '../dataout/crossTF99.6%/output/output.log'
    with open(file1, 'r') as file1, open(file2, 'r') as file2, open(file3, 'r') as file3:
        log1 = file1.read()
        log2 = file2.read()
        log3 = file3.read()

    train_pattern1 = r"episode:(.*), reward:(.*), memory size:(.*), time:(.*), info:(.*)"
    train_pattern2 = r"episode:(.*), reward:(.*), memory size:(.*), time:(.*), info:(.*)"
    train_pattern3 = r"episode:(.*), reward:(.*), memory size:(.*), time:(.*), info:(.*)"

    train_reward1 = []
    train_reward2 = []
    train_reward3 = []
    infolist2 = []
    navtime1 = []
    navtime2 = []
    navtime3 = []

    strl_success_rate = 0
    strl_timeout_rate = 0
    strl_collision_rate = 0

    for r in re.findall(train_pattern1, log1):
        train_reward1.append(float(r[1]))
        navtime1.append(float(r[3]))

    for r in re.findall(train_pattern2, log2):
        train_reward2.append(float(r[1]))
        infolist2.append((r[4]))
        navtime2.append(float(r[3]))
    for r in re.findall(train_pattern3, log3):
        train_reward3.append(float(r[1]))
        navtime3.append(float(r[3]))

    for info in infolist2[-500:]:
        if info == 'Reaching goal':
            strl_success_rate += 1
        if info == 'Collision':
            strl_collision_rate += 1
        if info == 'Timeout':
            strl_timeout_rate += 1

    train_reward1 = train_reward1[:max_episodes]
    train_reward2 = train_reward2[:max_episodes]
    train_reward3 = train_reward3[:max_episodes]

    navtime1 = navtime1[:max_episodes]
    navtime2 = navtime2[:max_episodes]
    navtime3 = navtime3[:max_episodes]

    train_reward_smooth1 = running_mean(train_reward1, args.window_size)
    train_reward_smooth2 = running_mean(train_reward2, args.window_size)
    train_reward_smooth3 = running_mean(train_reward3, args.window_size)
    train_nav_time1 = running_mean(navtime1, args.window_size)
    train_nav_time2 = running_mean(navtime2, args.window_size)
    train_nav_time3 = running_mean(navtime3, args.window_size)

    print('The success rate:%.4f, collision rate:%.4f,  timeout rate:%.4f' % (strl_success_rate / 500,
                                                                              strl_collision_rate / 500,
                                                                              strl_timeout_rate / 500))

    ax_legends = ['Baseline1', 'Baseline2', 'Ours']

    # 设置刻度格式为K
    def format_ticks(x, pos):
        return f'{x / 1000:.0f}k'

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for x in ax:
        x.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    ax[0].plot(range(len(train_reward_smooth1)), train_reward_smooth1, color="C3", linewidth=1)
    ax[0].plot(range(len(train_reward_smooth2)), train_reward_smooth2, color="C0", linewidth=1)
    ax[0].plot(range(len(train_reward_smooth3)), train_reward_smooth3, color="C1", linewidth=1)
    ax[0].set_title("Reward")
    ax[0].legend(ax_legends, shadow=True, loc='best', prop={'size': 12, 'family': 'Times New Roman'})

    ax[1].plot(range(len(train_nav_time1)), train_nav_time1, color="C3", linewidth=1)
    ax[1].plot(range(len(train_nav_time2)), train_nav_time2, color="C0", linewidth=1)
    ax[1].plot(range(len(train_nav_time3)), train_nav_time3, color="C1", linewidth=1)
    ax[1].set_title("Navtime")
    ax[1].legend(ax_legends, shadow=True, loc='best', prop={'size': 12, 'family': 'Times New Roman'})

    ax = plt.gca()
    ax.patch.set_facecolor('xkcd:white')

    ax.patch.set_alpha(0.5)

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.show()


if __name__ == '__main__':
    main()
