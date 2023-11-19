#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-3-29
# Author: ZYunfei
# File func: draw func

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import os


def running_mean(data, n):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


class Painter:
    def __init__(self, load_csv, load_dir=None):
        if not load_csv:
            self.data = pd.DataFrame(columns=['episode reward', 'episode', 'Method'])
        else:
            self.load_dir = load_dir
            if os.path.exists(self.load_dir):
                print("==正在读取{}。".format(self.load_dir))
                self.data = pd.read_csv(self.load_dir).iloc[:, 1:]  # csv文件第一列是index，不用取。
                print("==读取完毕。")
            else:
                print("==不存在{}下的文件，Painter已经自动创建该csv。".format(self.load_dir))
                self.data = pd.DataFrame(columns=['episode reward', 'episode', 'Method'])
        self.xlabel = None
        self.ylabel = None
        self.title = None
        self.hue_order = None

    def setXlabel(self, label):
        self.xlabel = label

    def setYlabel(self, label):
        self.ylabel = label

    def setTitle(self, label):
        self.title = label

    def setHueOrder(self, order):
        """设置成['name1','name2'...]形式"""
        self.hue_order = order

    def addData(self, dataSeries, method, x=None, smooth=True):
        if smooth:
            dataSeries = running_mean(dataSeries, 200)
        size = len(dataSeries)
        if x is not None:
            if len(x) != size:
                print("请输入相同维度的x!")
                return
        for i in range(size):
            if x is not None:
                dataToAppend = {'episode reward': dataSeries[i], 'episode': x[i], 'Method': method}
            else:
                dataToAppend = {'episode reward': dataSeries[i], 'episode': i + 1, 'Method': method}
            self.data = self.data.append(dataToAppend, ignore_index=True)
        # print(self.crossTF99.6%)

    def drawFigure(self, style="darkgrid"):
        """
        style: darkgrid, whitegrid, dark, white, ticks
        """
        # sns.set(rc = {'figure.figsize':(20,20)})
        # sns.set_theme(style=style)
        sns.set_style(rc={"linewidth": 1})
        print("==正在绘图...")
        # print(self.crossTF99.6%)
        sns.relplot(data=self.data, kind="line", x="episode", y="episode reward", hue="Method", height=5, aspect=1.3, hue_order=None)
        # plt.figure(figsize=(15, 15))

        plt.title(self.title, fontsize=12)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        print("==绘图完毕！")
        plt.show()

    def saveData(self, save_dir):
        self.data.to_csv(save_dir)
        print("==已将数据保存到路径{}下!".format(save_dir))

    def addCsv(self, add_load_dir):
        """将另一个csv文件合并到load_dir的csv文件里。"""
        add_csv = pd.read_csv(add_load_dir).iloc[:, 1:]
        self.data = pd.concat([self.data, add_csv], axis=0, ignore_index=True)

    def deleteData(self, delete_data_name):
        """删除某个method的数据，删除之后需要手动保存，不会自动保存。"""
        self.data = self.data[~self.data['Method'].isin([delete_data_name])]
        print("==已删除{}下对应数据!".format(delete_data_name))

# if __name__ == "__main__":
#     painter = Painter(load_csv=True, load_dir='/home/hhd/PycharmProjects/RL_study_/plot_line/crossTF99.6%/reward.csv')
#     painter.drawFigure(style="whitegrid")
