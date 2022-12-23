import pickle as pickle
import numpy as np
import math
import pandas as pd

## 对每个窗口进行FFT变化
def bin_power(X, Band, Fs):
    C = np.fft.fft(X)
    C = abs(C)
    Power = np.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[Freq_Index] = sum(
            C[int(np.floor(Freq / Fs * len(X))):int(np.floor(Next_Freq / Fs * len(X)))]
        )
    Power_Ratio = Power / sum(Power)
    return Power, Power_Ratio

## 减基线，重新安排数据结构
def FFT_Processing(sub, channel, band, window_size, step_size, sample_rate):
    meta = []
    with open('E:\JupyterNotebookWorkSpace\dataSet\data_preprocessed_python\s' + sub + '.dat', 'rb') as file:
        subject = pickle.load(file, encoding='latin1')
        for i in range(0, 40):
            # loop over 0-39 trails
            data = subject["data"][i]
            labels = subject["labels"][i]

            base_data = np.zeros((32, 5))
            # 得到特征提取后的基线
            for t in range(3):
                for c in channel:
                    U = data[c][t * 128:(t + 1) * 128]  # 滑动窗口
                    V = bin_power(U, band, sample_rate)  # 对每个窗口进行FFT变化，得到五种频带上的值。
                    base_data[c] = [base_data[c][r] + (V[0] / 3)[r] for r in range(5)]


            start = 384;  # 384（从第三秒开始）
            while start + window_size <= data.shape[1]:
                meta_array = []
                meta_data = []  # meta vector for analysis
                for j in channel:
                    X = data[j][start: start + window_size]  # 滑动窗口
                    Y = bin_power(X, band, sample_rate)  # 对每个窗口进行FFT变化，得到五种频带上的值。
                    Y0 = Y[0] - base_data[j]  #减去特征提取后的基线
                    meta_data = meta_data + list(Y0)

                meta_array.append(np.array(meta_data))
                meta_array.append(labels)

                meta.append(np.array(meta_array))
                start = start + step_size

        meta = np.array(meta)
        np.save('out1\s' + sub, meta, allow_pickle=True, fix_imports=True)


def lower_sample_data(df, label, percent=1):
    '''
    percent:多数类别下采样的数量相对于少数类别样本数量的比例
    '''
    # 将多数类别的样本放在data1,将少数类别的样本放在data0
    data1 = df[label > 0 ]
    data0 = df[label < 0 ]
    label1 = label[label > 0]
    label0 = label[label < 0]

    if (len(data1) < len(data0)):
        data = data0
        lab =label0
        data0 = data1
        label0 = label1
        data1 = data
        label1 = lab
    print("多少比:", len(data1), len(data0))
    index = np.random.randint(len(data1), size=percent * (len(df) - len(data1)))  # 随机给定下采样取出样本的序号
    lower_data1 = data1[list(index)]  # 下采样
    lower_label1 = label1[list(index)]
    return (np.append(lower_data1, data0, axis=0)),(np.append(lower_label1, label0))


def up_sample_data(df, label, percent=1):
    '''
    percent:少数类别样本数量的重采样的比例，可控制，一般不超过0.5，以免过拟合
    '''
    data1 = df[label > 0]
    data0 = df[label < 0]
    label1 = label[label > 0]
    label0 = label[label < 0]

    if (len(data1) < len(data0)):
        data = data0
        lab = label0
        data0 = data1
        label0 = label1
        data1 = data
        label1 = lab
    print( data0.shape, data1.shape)
    index = np.random.randint(len(data0), size= int(percent * (len(df) - len(data0))))  # 随机给定上采样取出样本的序号
    up_data0 = data0[list(index)]  # 上采样
    up_label0 = label0[list(index)]
    print( up_data0.shape)
    return(np.append(up_data0, data1, axis=0)),(np.append(up_label0, label1))

def show_accuracy(M,L):
    output = M
    label = L
    k = 0
    l = 0

    for i in range(len(label)):
        k = k + (output[i] - label[i])*(output[i] - label[i]) #square difference
        if (output[i] == label[i]):
           l = l + 1
    return len(label),l
