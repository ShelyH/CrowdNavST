import math
import traceback

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

dirPath = os.path.dirname(os.path.realpath(__file__))
"""
plot the performance of core from TensorBoard Log History
"""
DEFAULT_SIZE_GUIDANCE = {
    "scalars": 0,
    "tensors": 0,
}

flatui = ["#EF5D0B", "#2926AB", "#9b59b6", "#2ecc71",
          "#34495e", "#2ecc71", "#e67e22", "#f1c40f"]

material = ["#FFC107", "#E91E63", "#9C27B0", "#607D8B",
            "#2196F3", "#009688", "#795548", "#607D8B"]

sns.set(style="white", font_scale=1.0, rc={"lines.linewidth": 1.2}, palette=sns.color_palette(flatui), color_codes=False)


def plot_data(data, x_axis='num episodes', y_axis="SAC_exp1", hue="algorithm", smooth=1, ax=None, **kwargs):
    if smooth >= 1:
        """
        smooth crossTF99.6% with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)

        for datum in data:
            x = np.asarray(datum[y_axis])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[y_axis] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True, sort=True)

    sns.lineplot(data=data, x=x_axis, y=y_axis, hue=hue, ci='sd', ax=ax, **kwargs)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='lower right', handles=handles[0:], labels=labels[0:], prop={'size': 14, 'family': 'Times New Roman'})
    # ax.legend()
    """Spining up style"""
    ax.grid(True, alpha=0.4)
    # ax.legend(loc='upper right', ncol=9, handlelength=1, frameon=False,
    #           mode="expand", borderaxespad=0.02, prop={'size': 8})

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.tight_layout(pad=1.2)
    plt.xlabel("Episodes", fontproperties='Times New Roman')
    plt.ylabel("Reward", fontproperties='Times New Roman')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.grid = True
    xscale = np.max(np.asarray(data[x_axis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), prop={'size': 14, 'family': 'Times New Roman'})


def load_event_scalars(log_path):
    feature = log_path.split(os.sep)[-1]
    print(f"Processing logfile: {os.path.abspath(log_path)}")
    if feature.find("_") != -1:
        feature = feature.split("_")[-1]
    # print(feature)
    feature = 'reward'
    df = pd.DataFrame()
    try:
        event_acc = EventAccumulator(log_path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        env_list = event_acc.Scalars
        use_tensorflow = False
        if not tags:
            tags = event_acc.Tags()["tensors"]
            env_list = event_acc.tensors.Items
            use_tensorflow = True
        for tag in set(tags) & {"total reward", "reward", "min reward", "max reward", "num steps"}:
            event_list = env_list(tag)
            # if use_tensorflow:
            #     values = list(map(lambda x: float(tf.make_ndarray(x.tensor_proto)), event_list))
            #     step = list(map(lambda x: x.step, event_list))
            #     df[tag] = values
            # else:
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            df = pd.DataFrame({feature: values}, index=step)
    # Dirty catch of DataLossError
    except:
        print("Event file possibly corrupt: {}".format(os.path.abspath(log_path)))
        traceback.print_exc()
    return df


def get_env_alg_log(log_path):
    """
    split Environment log by algorithm
    :param log_path:
    :return:
    """
    alg = log_path.split(os.sep)[-1]
    # print(alg)
    if alg.find("_") != -1:
        alg = alg.rsplit("_", maxsplit=1)[0]

    # print(alg)

    def env_alg_fulldir(x):
        return os.path.join(log_path, x)

    alg_features = [env_alg_fulldir(fea) for fea in os.listdir(log_path) if os.path.isdir(env_alg_fulldir(fea))]
    # print(alg_features)
    if alg_features:
        df = pd.concat([load_event_scalars(feature) for feature in alg_features], axis=1)
    else:
        df = load_event_scalars(log_path)
    # print(df)
    df["num episodes"] = np.arange(1, df.shape[0] + 1)
    df["algorithm"] = [alg] * df.shape[0]
    # print(alg)
    return df


def plot_all_logs(log_dir=None, x_axis=None, y_axis=None, hue=None, smooth=1, env_filter_func=None, alg_filter_func=None):
    if y_axis is None:
        y_axis = ['min reward', 'average reward', 'max reward', 'total reward']

    basedir = os.path.dirname(log_dir)  # ../log/

    def fulldir(x):
        return os.path.join(basedir, x)  # ../log/Ant-v3/

    envs_logdirs = sorted(
        [fulldir(x) for x in os.listdir(basedir) if os.path.isdir(fulldir(x))])  # [../log/Ant-v3/, ../log/Hopper-v3/]
    if env_filter_func:
        envs_logdirs = sorted(filter(env_filter_func, envs_logdirs))
    print("All envs are: ", list(map(os.path.abspath, envs_logdirs)))

    num_envs = len(envs_logdirs)
    sub_plot_height = round(math.sqrt(num_envs))
    sub_plot_width = math.ceil(num_envs / sub_plot_height)

    def envs_fulldir(env_dir, alg_dir):
        return os.path.join(env_dir, alg_dir)

    for y_ax in y_axis:
        k = 0
        fig, axes = plt.subplots(sub_plot_height, sub_plot_width, figsize=(
            8 * sub_plot_width, 6 * sub_plot_height))
        for env_dir in envs_logdirs:
            if sub_plot_height == 1:
                if sub_plot_width == 1:
                    ax = axes
                else:
                    ax = axes[k]
            else:
                ax = axes[k // sub_plot_width][k % sub_plot_width]

            env_id = env_dir.split(os.sep)[-1]
            env_alg_dirs = sorted(
                filter(os.path.isdir, [envs_fulldir(env_dir, alg_dir) for alg_dir in os.listdir(env_dir)]))
            # print(alg_filter_func, env_alg_dirs)
            if alg_filter_func:
                env_alg_dirs = sorted(filter(alg_filter_func, env_alg_dirs))
            print(f"Env id: {env_id}, logs: {list(map(os.path.abspath, env_alg_dirs))}")

            env_log_df = [get_env_alg_log(env_alg_dir) for env_alg_dir in env_alg_dirs]
            # print(env_log_df)
            make_plot(data=env_log_df, x_axis=x_axis, y_axis=y_ax,
                      smooth=smooth, title=env_id, hue=hue, ax=ax)
            k += 1
    plt.show()
    # plt.savefig("../Algorithms/images/bench_trpo_tf2.png")


def make_plot(data, x_axis=None, y_axis=None, title=None, hue=None, smooth=1, estimator='mean', ax=None):
    estimator = getattr(np, estimator)
    if len(data) > 0:
        # print(crossTF99.6%)
        plot_data(data, x_axis=x_axis, y_axis=y_axis, hue=hue,
                  smooth=smooth, ax=ax, estimator=estimator)
    if title:
        ax.set_title(title, fontdict={'family': 'Times New Roman', 'size': 15})


# @click.command()
# @click.option("--log_dir", type=str, default="../log/", help="Directory to load tensorboard logs")
# @click.option("--x_axis", type=str, default="num episodes", help="X axis crossTF99.6%")
# @click.option("--y_axis", type=list, default=["SAC_exp1"], help="Y axis crossTF99.6%(can be multiple)")
# @click.option("--hue", type=str, default="algorithm", help="Hue for legend")
def main(log_dir='/home/hhd/PycharmProjects/CrowdNavRL/crowd_nav/log/', x_axis='num episodes', y_axis=None, hue='algorithm',
         env_filter_func=None, alg_filter_func=None):
    """plot performance of all environments and algorithms

    1. traverse all environments and plots for all algos tensorboard logs in that environment
    2. generate a dataframe for each environment, which contains all algos tensorboard log information

    Args:
        log_dir (str, optional): Directory of tensorboard log files. Defaults to '../Algorithms/pytorch/log/'.
        x_axis (str, optional): X label of plot. Defaults to 'num episodes'.
        y_axis (list, optional): Y label of plot. Defaults to ['SAC_exp1'].
        hue (str, optional): [description]. Defaults to 'algorithm'.
        env_filter_func ([type], optional): Filter function to select enviroments. Defaults to None.
        alg_filter_func ([type], optional): Filter function to select algorithms. Defaults to None.
    """
    if y_axis is None:
        y_axis = ['reward']
    plot_all_logs(log_dir=log_dir, x_axis=x_axis, y_axis=y_axis, hue=hue,
                  smooth=5000, env_filter_func=env_filter_func, alg_filter_func=alg_filter_func)


if __name__ == "__main__":
    def env_filter_func_pg(x): return x.split(os.sep)[-1] in ["CrowdSim-v0"]


    def alg_filter_func(x): return x.split(os.sep)[-1].rsplit("_")[0] in ["DSRNN", "STTFSAC(Ours)", "SAC-LSTM", "SARL"]


    # print(alg_filter_func())
    main(env_filter_func=None, alg_filter_func=alg_filter_func)
    sns.despine()
