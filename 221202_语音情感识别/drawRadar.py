import numpy as np
import matplotlib.pyplot as plt


def draw(data_prob, class_labels: tuple, num_classes: int):
    plt.clf()  # 清除刷新前的图表，防止数据量过大消耗内存
    # 数据
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    data = np.concatenate((data_prob, [data_prob[0]]))  # 闭合
    angles = np.concatenate((angles, [angles[0]]))  # 闭合
    fig = plt.figure(1)
    # polar参数
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, data, 'bo-', linewidth=2)
    ax.fill(angles, data, facecolor='r', alpha=0.25)
    ax.set_thetagrids(
        angles * 180 / np.pi, class_labels, fontproperties="SimHei")
    ax.set_title("Emotion Recognition", va='bottom', fontproperties="SimHei")
    # 在这里设置雷达图的数据最大值
    ax.set_rlim(0, 1)
    ax.grid(True)
    # plt.clf()  # 清除刷新前的图表，防止数据量过大消耗内存
    # plt.show()
    plt.pause(1)  #暂停时间


plt.ion()  # 开启interactive mode

plt.ioff()  # 关闭画图的窗口
plt.show()
