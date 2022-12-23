# import numpy as np
# import matplotlib.pyplot as plt

# # angry, fear, happy, neutral, sad, surprise
# def ratio_pic(data):
#     # 定义各个参数的值

#     # 标签
#     labels = np.array(['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
#     # 数据个数
#     dataLenth = 6
#     # 数据
#     # data = np.array([angry, fear, happy, neutral, sad, surprise])
#     # data = np.array(data)
#     angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)
#     data = np.concatenate((data, [data[0]]))  # 闭合
#     angles = np.concatenate((angles, [angles[0]]))  # 闭合

#     fig = plt.figure()

#     # polar参数
#     ax = fig.add_subplot(111, polar=True)
#     ax.plot(angles, data, 'bo-', linewidth=2)
#     ax.fill(angles, data, facecolor='r', alpha=0.25)
#     ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")
#     ax.set_title("Emotion Recognition", va='bottom', fontproperties="SimHei")

#     # 在这里设置雷达图的数据最大值
#     ax.set_rlim(0, 1)

#     ax.grid(True)
#     plt.ion()
#     plt.show()
#     plt.pause(4)
#     plt.close()

# def Radar(data_prob, class_labels: Tuple, num_classes: int):

#     angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
#     data = np.concatenate((data_prob, [data_prob[0]]))  # 闭合
#     angles = np.concatenate((angles, [angles[0]]))  # 闭合

#     fig = plt.figure()

#     # polar参数
#     ax = fig.add_subplot(111, polar=True)
#     ax.plot(angles, data, 'bo-', linewidth=2)
#     ax.fill(angles, data, facecolor='r', alpha=0.25)
#     ax.set_thetagrids(angles * 180 / np.pi, class_labels)
#     ax.set_title("Emotion Recognition", va='bottom')

#     # 设置雷达图的数据最大值
#     ax.set_rlim(0, 1)

#     ax.grid(True)
#     # plt.ion()
#     plt.show()
#     # plt.pause(4)
#     # plt.close()
