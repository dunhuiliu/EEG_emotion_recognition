# SpeechEmotionRecognition

## 基于SVM的语音情感分析，数据集使用中科大的casia，6种情感，共1200条数据。
1.目前最好的识别率为59%，C=10，MFCC特征个数为48

2.已实现置信概率的计算，等待绘图的接口

3.训练模型请使用`train.py`，或者直接导入已训练好的模型`classfier.m`

4.可以在`realTimeAnalysis.py`中对自己的录音进行实时情感分析

5.读入多个音频文件并进行分析，请直接运行`testEmotion.py`，会播放音频文件并在雷达图中展示其结果
