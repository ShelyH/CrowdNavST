# import matplotlib.pyplot as plt
#
# x = [3, 5, 7, 9]
# y1 = [83, 78, 67, 60]
# y2 = [98, 91, 88, 80]
# y3 = [98.4, 93, 91, 85]
# y4 = [100, 95.5, 93, 87]
# y5 = [100, 98, 96, 92]
# y6 = [100, 99.6, 98, 95]
# plt.title('Invisible scenario', fontsize=16, fontname='Arial')  # 折线图标题
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
# plt.xlabel('Number of pedestrians', fontsize=16, fontname='Arial')  # x轴标题
# plt.ylabel('Success rate', fontsize=16, fontname='Arial')  # y轴标题
# plt.plot(x, y1, marker='s', markersize=5, color='b')  # 绘制折线图，添加数据点，设置点的大小
# plt.plot(x, y2, marker='o', markersize=5, color='g')
# plt.plot(x, y3, marker='D', markersize=5, color='c')
# plt.plot(x, y4, marker='v', markersize=5, color='y')
# plt.plot(x, y5, marker='x', markersize=5, color='saddlebrown')
# plt.plot(x, y6, marker='*', markersize=5, color='r')
# plt.xticks(x)
# # for a, b in zip(x, y1): plt.text(a, b, b,  va='bottom', fontsize=10)  # 设置数据标签位置及大小
# # for a, b in zip(x, y2): plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# # for a, b in zip(x, y3): plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# # for a, b in zip(x, y4): plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# # for a, b in zip(x, y5): plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# # for a, b in zip(x, y6): plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# plt.legend(['CADRL', 'LSTM-RL', 'SARL', 'DS-RNN', 'ST$^{2}$', 'STTFRL'], fontsize=13)  # 设置折线名称plt.show()  # 显示折线图
# plt.show()
import matplotlib.pyplot as plt

x = [3, 5, 7, 9]
y1 = [96, 92, 85, 78]
y2 = [99, 93, 90, 82]
y3 = [99.6, 93, 91, 85]
y4 = [100, 99, 94, 88]
y5 = [100, 98, 95, 90]
y6 = [100, 99.6, 99.5, 97]
plt.title('Invisible scenario', fontsize=16, fontname='Arial')  # 折线图标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('Number of pedestrians', fontsize=16, fontname='Arial')  # x轴标题
plt.ylabel('Success rate', fontsize=16, fontname='Arial')  # y轴标题
plt.plot(x, y1, marker='s', markersize=5, color='b')  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, y2, marker='o', markersize=5, color='g')
plt.plot(x, y3, marker='D', markersize=5, color='c')
plt.plot(x, y4, marker='v', markersize=5, color='y')
plt.plot(x, y5, marker='x', markersize=5, color='saddlebrown')
plt.plot(x, y6, marker='*', markersize=5, color='r')
plt.xticks(x)
# for a, b in zip(x, y1): plt.text(a, b, b,  va='bottom', fontsize=10)  # 设置数据标签位置及大小
# for a, b in zip(x, y2): plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, y3): plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, y4): plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, y5): plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, y6): plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
plt.legend(['CADRL', 'LSTM-RL', 'SARL', 'DS-RNN', 'ST$^{2}$', 'STTFRL'], fontsize=13)  # 设置折线名称plt.show()  # 显示折线图
plt.show()
