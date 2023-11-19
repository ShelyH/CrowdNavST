import math
import random

import matplotlib.pyplot as plt
import numpy as np

# plt.figure(figsize=(10, 10))

fig, ax = plt.subplots(figsize=(8, 7))
ax.tick_params(labelsize=12)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
# ax.set_xlabel('x(m)', fontsize=20)
# ax.set_ylabel('y(m)', fontsize=20)
plt.figure(1, constrained_layout=True)
robot = plt.Circle([0, 0], 0.3, fill=True, linewidth=0.01, color='red')
plt.gca().add_patch(robot)

# 绘制障碍物
obstacles = [((2, 4), 0.5), ((-6, 6), 0.5), ((6, -1), 0.5), ((4, -2), 0.4), ((0, 3), 0.4)]  # 障碍物列表

for obstacle_center, obstacle_radius in obstacles:
    obstacle = plt.Circle(obstacle_center, obstacle_radius, fill=True, linewidth=1, color='green')
    plt.gca().add_patch(obstacle)
# 机器人位置
robot_x, robot_y = 0, 0
# 设置坐标系范围
# plt.xlim([robot_x - 10, robot_x + 10])  # 调整 x 轴范围
# plt.ylim([robot_y - 10, robot_y + 10])  # 调整 y 轴范围

# 激光线的长度
laser_length = 10
# 计算激光线的结束点坐标
# laser_end_x = robot_x + laser_length * math.cos(0.5)
# laser_end_y = robot_y + laser_length * math.sin(0.5)
artists = []
# 绘制360根激光线
def plot_movrobot(robot_x, robot_y):
    # plt.clf()
    for angle in np.linspace(0, 2 * np.pi, num=36):
        laser_end_x = robot_x + laser_length * np.cos(angle)
        laser_end_y = robot_y + laser_length * np.sin(angle)

        # 判断激光线是否与障碍物相交
        intersect = False
        for obstacle_center, obstacle_radius in obstacles:
            dx = obstacle_center[0] - robot_x
            dy = obstacle_center[1] - robot_y
            dist = np.hypot(dx, dy)
            laser_dir_x = dx / dist
            laser_dir_y = dy / dist

            t = laser_length
            # while t >= 0:
            x = robot_x + t * laser_dir_x
            y = robot_y + t * laser_dir_y
            if np.hypot(x - obstacle_center[0], y - obstacle_center[1]) <= obstacle_radius:
                intersect = True
                break
                # t -= 0.1
            print(intersect)
            if intersect:
                break

        # 绘制激光线
        if intersect:
            plt.plot([robot_x, x], [robot_y, y], color='green')
        else:
            plt.plot([robot_x, laser_end_x], [robot_y, laser_end_y], color='red')

def main():
    robot_pos=[[0,0],[2,3]]
    for i in robot_pos:
        plot_movrobot(i[0], i[1])

    # 显示图形
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()