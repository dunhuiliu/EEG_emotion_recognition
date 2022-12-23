import librosa
import os
from random import shuffle
import numpy as np
from sklearn import svm
import joblib
import sklearn

# C:误差项惩罚参数,对误差的容忍程度。C越大，越不能容忍误差
# gamma：选择RBF函数作为kernel，越大，支持的向量越少；越小，支持的向量越多
# kernel: linear, poly, rbf, sigmoid, precomputed
# decision_function_shape: ovo, ovr(default)
#
# #

path = r'C:\GitHub\svm_test\casia'
EMOTION_LABEL = {
    'angry': '1',
    'fear': '2',
    'happy': '3',
    'neutral': '4',
    'sad': '5',
    'surprise': '6'
}


def getFeature(path, mfcc_feature_num=16):
    y, sr = librosa.load(path)

    # 对于每一个音频文件提取其mfcc特征
    # y:音频时间序列;
    # n_mfcc:要返回的MFCC数量
    mfcc_feature = librosa.feature.mfcc(y, sr, n_mfcc=16)
    zcr_feature = librosa.feature.zero_crossing_rate(y)
    energy_feature = librosa.feature.rmse(y)
    rms_feature = librosa.feature.rmse(y)

    mfcc_feature = mfcc_feature.T.flatten()[:mfcc_feature_num]
    zcr_feature = zcr_feature.flatten()
    energy_feature = energy_feature.flatten()
    rms_feature = rms_feature.flatten()

    zcr_feature = np.array([np.mean(zcr_feature)])
    energy_feature = np.array([np.mean(energy_feature)])
    rms_feature = np.array([np.mean(rms_feature)])

    data_feature = np.concatenate((mfcc_feature, zcr_feature, energy_feature,
                                   rms_feature))
    return data_feature


def getData(mfcc_feature_num=16):
    """找到数据集中的所有语音文件的特征以及语音的情感标签"""
    wav_file_path = []
    person_dirs = os.listdir(path)
    for person in person_dirs:
        if person.endswith('txt'):
            continue
        emotion_dir_path = os.path.join(path, person)
        emotion_dirs = os.listdir(emotion_dir_path)
        for emotion_dir in emotion_dirs:
            if emotion_dir.endswith('.ini'):
                continue
            emotion_file_path = os.path.join(emotion_dir_path, emotion_dir)
            emotion_files = os.listdir(emotion_file_path)
            for file in emotion_files:
                if not file.endswith('wav'):
                    continue
                wav_path = os.path.join(emotion_file_path, file)
                wav_file_path.append(wav_path)

    # 将语音文件随机排列
    shuffle(wav_file_path)
    data_feature = []
    data_labels = []

    for wav_file in wav_file_path:

        data_feature.append(getFeature(wav_file, mfcc_feature_num))
        data_labels.append(int(EMOTION_LABEL[wav_file.split('\\')[-2]]))

    return np.array(data_feature), np.array(data_labels)


def train():
    # 使用svm进行预测
    best_acc = 0
    best_mfcc_feature_num = 0
    best_C = 0

    for C in range(13, 20):
        for i in range(40, 55):
            data_feature, data_labels = getData(i)
            split_num = 200
            train_data = data_feature[:split_num, :]
            train_label = data_labels[:split_num]
            test_data = data_feature[split_num:, :]
            test_label = data_labels[split_num:]
            clf = svm.SVC(
                decision_function_shape='ovo',
                kernel='rbf',
                C=C,
                gamma=0.0001,
                probability=True)
            print("train start")
            clf.fit(train_data, train_label)
            print("train over")
            print(C, i)
            acc_dict = {}
            for test_x, test_y in zip(test_data, test_label):
                pre = clf.predict([test_x])[0]
                if pre in acc_dict.keys():
                    continue
                acc_dict[pre] = test_y
            acc = sklearn.metrics.accuracy_score(
                clf.predict(test_data), test_label)
            if acc > best_acc:
                best_acc = acc
                best_C = C
                best_mfcc_feature_num = i
                print('best_acc', best_acc)
                print('best_C', best_C)
                print('best_mfcc_feature_num', best_mfcc_feature_num)
                print()

            # 保存模型
            joblib.dump(clf,
                        'Models/C_' + str(C) + '_mfccNum_' + str(i) + '.m')

    print('best_acc', best_acc)
    print('best_C', best_C)
    print('best_mfcc_feature_num', best_mfcc_feature_num)


if __name__ == "__main__":
    train()
