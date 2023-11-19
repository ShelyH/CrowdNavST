import argparse
import re

from draw_line import Painter


def get_data():
    painter = Painter(load_csv=True, load_dir='../data/output/reward.csv')
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+')
    args = parser.parse_args()
    rewardList = []
    for _, log_file in enumerate(args.log_files):
        with open(log_file, 'r') as file:
            log = file.read()
        # train_pattern = r"TRAIN_new in episode (?P<episode>\d+) has success rate: (?P<sr>[0-1].\d+), " \
        #                 r"collision rate: (?P<cr>[0-1].\d+), nav time: (?P<time>\d+.\d+), " \
        #                 r"total reward: (?P<reward>[-+]?\d+.\d+)"
        train_pattern = r" Updates:(.*), num timesteps:(.*), FPS:(.*), Last:(.*), " \
                        r"training episodes mean/median reward:(.*)/(.*), min/max reward:(.*)/(.*)"
        for r in re.findall(train_pattern, log):
            rewardList.append(float(r[4]))

    # painter.deleteData('LSTM')
    painter.addData(rewardList, 'sacTF')
    painter.saveData('crossTF99.6%/output/reward.csv')
    painter.setTitle('sacTF reward')
    painter.setXlabel('episode')
    painter.setYlabel('reward')
    painter.drawFigure()


if __name__ == '__main__':
    get_data()
