# coding=utf-8
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
# plt.rc('font',family='Times New Roman', size=18)
width = 8
height = 6


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+')
    parser.add_argument('--plot_sr', default=True, action='store_true')
    parser.add_argument('--plot_cr', default=True, action='store_true')
    parser.add_argument('--plot_time', default=True, action='store_true')
    parser.add_argument('--plot_reward', default=True, action='store_true')
    parser.add_argument('--plot_train', default=True, action='store_true')
    parser.add_argument('--plot_VMeanSE', default=True, action='store_true')
    parser.add_argument('--plot_new_train', default=True, action='store_true')
    parser.add_argument('--window_size', type=int, default=300)
    args = parser.parse_args()

    # define the names of the models you want to plot and the longest episodes you want to show
    # models = ['d_distance', 'd_factor']
    max_episodes = 10000

    ax1 = ax2 = ax3 = ax4 = ax5 = None
    ax1_legends = []
    ax2_legends = []
    ax3_legends = []
    ax4_legends = []
    ax5_legends = []

    for _, log_file in enumerate(args.log_files):
        with open(log_file, 'r') as file:
            log = file.read()

        new_train_pattern = r"INFO: TRAIN OurMulti in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), " \
                            r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
                            r"total reward: (?P<reward>[-+]?\d+.\d+)"
        new_train_episode = []
        new_train_sr = []
        new_train_cr = []
        new_train_time = []
        new_train_reward = []

        for r in re.findall(new_train_pattern, log):
            new_train_episode.append(int(r[0]))
            new_train_sr.append(float(r[1]))
            new_train_cr.append(float(r[2]))
            new_train_time.append(float(r[3]))
            new_train_reward.append(float(r[4]))

        new_train_episode = new_train_episode[:max_episodes]
        new_train_sr = new_train_sr[:max_episodes]
        new_train_cr = new_train_cr[:max_episodes]
        new_train_time = new_train_time[:max_episodes]
        new_train_reward = new_train_reward[:max_episodes]
        # new_train_VMeanSE = new_train_VMeanSE[:max_episodes]
        # new_train_times = new_train_times[:max_episodes]

        new_train_sr_smooth = running_mean(new_train_sr, args.window_size)
        new_train_cr_smooth = running_mean(new_train_cr, args.window_size)
        new_train_time_smooth = running_mean(new_train_time, args.window_size)
        new_train_reward_smooth = running_mean(new_train_reward, args.window_size)
        # new_train_VMeanSE_smooth = running_mean(new_train_VMeanSE, args.window_size)
        train_pattern = r"INFO: TRAIN Multi in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), " \
                        r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
                        r"total reward: (?P<reward>[-+]?\d+.\d+)"
        train_episode = []
        train_sr = []
        train_cr = []
        train_time = []
        train_reward = []
        train_VMeanSE = []
        train_times = []
        for r in re.findall(train_pattern, log):
            train_episode.append(int(r[0]))
            train_sr.append(float(r[1]))
            train_cr.append(float(r[2]))
            train_time.append(float(r[3]))
            train_reward.append(float(r[4]))
        train_episode = train_episode[:max_episodes]
        train_sr = train_sr[:max_episodes]
        train_cr = train_cr[:max_episodes]
        train_time = train_time[:max_episodes]
        train_reward = train_reward[:max_episodes]
        # train_VMeanSE = train_VMeanSE[:max_episodes]
        # train_times = train_times[:max_episodes]

        # smooth training plot
        train_sr_smooth = running_mean(train_sr, args.window_size)
        train_cr_smooth = running_mean(train_cr, args.window_size)
        train_time_smooth = running_mean(train_time, args.window_size)
        train_reward_smooth = running_mean(train_reward, args.window_size)
        # train_VMeanSE_smooth = running_mean(train_VMeanSE, args.window_size)

        # plot sr
        if args.plot_sr:

            if ax1 is None:
                fig1, ax1 = plt.subplots(1, 1, figsize=(width, height))
            if args.plot_new_train:
                ax1.plot(range(len(new_train_sr_smooth)), new_train_sr_smooth, color="C3",)
                ax1_legends.append('本文算法')
            if args.plot_train:
                ax1.plot(range(len(train_sr_smooth)), train_sr_smooth, color="C1",)
                ax1_legends.append('对比算法')
            ax1.tick_params(labelsize=16)
            ax1.legend(ax1_legends, shadow=True, loc='best', fontsize=16)
            ax1.spines['top'].set_visible(False)  # 去掉上边框
            ax1.spines['right'].set_visible(False)  # 去掉右边框
            ax1.set_xlabel('训练步数', fontsize=18)
            ax1.set_ylabel('成功率', fontsize=18)
            # ax1.set_title('Success Rate',fontsize=18)

        # plot time
        if args.plot_time:
            if ax2 is None:
                fig2, ax2 = plt.subplots(1, 1, figsize=(width, height))
            if args.plot_new_train:
                ax2.plot(range(len(new_train_time_smooth)), new_train_time_smooth, color="C3",)
                ax2_legends.append('本文算法')
            if args.plot_train:
                ax2.plot(range(len(train_time_smooth)), train_time_smooth,color="C1",)
                ax2_legends.append('对比算法')
            ax2.tick_params(labelsize=16)
            ax2.legend(ax2_legends, shadow=True, loc='best', fontsize=16)
            ax2.spines['top'].set_visible(False)  # 去掉上边框
            ax2.spines['right'].set_visible(False)  # 去掉右边框
            ax2.set_xlabel('训练步数', fontsize=18)
            ax2.set_ylabel('导航时间', fontsize=18)
            # ax2.set_title("Robot's Time to Reach Goal",fontsize=18)

        # plot cr
        if args.plot_cr:
            if ax3 is None:
                fig3, ax3 = plt.subplots(1, 1, figsize=(width, height))
            if args.plot_new_train:
                ax3.plot(range(len(new_train_cr_smooth)), new_train_cr_smooth,color="C3",)
                ax3_legends.append('本文算法')
            if args.plot_train:
                ax3.plot(range(len(train_cr_smooth)), train_cr_smooth, color="C1",)
                ax3_legends.append('对比算法')
            ax3.tick_params(labelsize=16)
            ax3.legend(ax3_legends, shadow=True, loc='best', fontsize=16)
            ax3.spines['top'].set_visible(False)  # 去掉上边框
            ax3.spines['right'].set_visible(False)  # 去掉右边框
            ax3.set_xlabel('训练步数', fontsize=18)
            ax3.set_ylabel('碰撞率', fontsize=18)
            # ax3.set_title('Collision Rate',fontsize=18)

        # plot reward
        if args.plot_reward:
            if ax4 is None:
                fig4, ax4 = plt.subplots(1, 1, figsize=(width, height))
            if args.plot_new_train:
                ax4.plot(range(len(new_train_reward_smooth)), new_train_reward_smooth, color="C3",)
                ax4_legends.append('本文算法')
            if args.plot_train:
                ax4.plot(range(len(train_reward_smooth)), train_reward_smooth, color="C1",)
                # ax4_legends.append(models[i])
                ax4_legends.append('对比算法')

            ax4.tick_params(labelsize=16)
            ax4.legend(ax4_legends, shadow=True, loc='best', fontsize=16)
            # ax4.grid(True)
            ax4 = plt.gca()
            # ax4.patch.set_facecolor('xkcd:mint green')
            ax4.spines['top'].set_visible(False)  # 去掉上边框
            ax4.spines['right'].set_visible(False)  # 去掉右边框
            # ax4.patch.set_facecolor("green")
            ax4.patch.set_alpha(0.5)
            ax4.set_xlabel('训练步数', fontsize=18)
            ax4.set_ylabel('累计奖励', fontsize=18)
            # ax4.set_title('Cumulative Discounted Reward',fontsize=18)

        # plot VMeanSE
        # if args.plot_VMeanSE:
        #     if ax5 is None:
        #         fig5, ax5 = plt.subplots(1,1,figsize=(width,height))
        #     if args.plot_new_train:
        #         ax5.plot(range(len(new_train_VMeanSE_smooth)), new_train_VMeanSE_smooth, 'orange')
        #         ax5_legends.append('Ours')
        #     if args.plot_train:
        #         ax5.plot(range(len(train_VMeanSE_smooth)), train_VMeanSE_smooth,'blue')
        #         ax5_legends.append('Baseline')
        #     ax5.tick_params(labelsize=16)
        #     ax5.legend(ax5_legends, shadow=True, loc='best',fontsize=16)
        #     ax5.spines['top'].set_visible(False)  # 去掉上边框
        #     ax5.spines['right'].set_visible(False)  # 去掉右边框
        #     ax5.set_xlabel('Episodes',fontsize=18)
        #     ax5.set_ylabel('VMeanSE',fontsize=18)
        #     #ax3.set_title('VMeanSE',fontsize=18)

    fig1.savefig('Success_Rate.eps', dpi=600, format='eps')
    fig2.savefig('Time.eps', dpi=600, format='eps')
    fig3.savefig('Collision_Rate.eps', dpi=600, format='eps')
    fig4.savefig('Reward.eps', dpi=600, format='eps')
    # fig5.savefig('VMeanSE.eps', dpi=600, format='eps')
    plt.show()


if __name__ == '__main__':
    main()
