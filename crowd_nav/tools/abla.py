import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7, 6)) # 创建图实例
x=np.array([5,7,10,13,16])
# x = np.linspace(0,2,100) # 创建x的取值范围
y1 = np.array([0.99,0.94,0.91,0.83,0.79])
ax.plot(x, y1, label='Ours') # 作y1 = x 图，并标记此线名为linear
y2 = np.array([0.95,0.91,0.84,0.79,0.71])
ax.plot(x, y2, label='mlp-based') #作y2 = x^2 图，并标记此线名为quadratic
y3 = np.array([0.97,0.905,0.84,0.71,0.62])
ax.plot(x, y3, label='only ae') # 作y3 = x^3 图，并标记此线名为cubic
ax.set_xlabel('Number of dynamic humans',fontsize=16) #设置x轴名称 x label
ax.set_ylabel('Success rate',fontsize=16) #设置y轴名称 y label

ax.legend(fontsize=16) #自动检测要在图例中显示的元素，并且显示
plt.yticks( size=16)
plt.xticks(size=16)
plt.show() #图形可视化
