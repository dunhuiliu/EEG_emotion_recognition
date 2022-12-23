"""1. Global Variables setup
FileName data\SXX.dat, XX \ in [0, 31]
*data: 40x40x8064: trialxchannelxdata
*label: 40x4: video/trialxlabel(valence, arousal, dominance, liking)
"""
channel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,30, 31]
band = [4, 8, 12, 16, 25, 45]  # 5 bands
window_size = 128  # Averaging band power of 1sec
step_size = 128  # Each 1 sec update once
sample_rate = 128  # Sampling rate of 128 Hz
subjectList = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
               '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']

import numpy as np
import pandas as pd
import math

from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
import pickle as pickle

import os
import time
from sklearn.metrics import accuracy_score

import fftprocess
import MMD
import trAdaboost as Tra
import torch
import sys

""" 2. FFT with pyeeg-bin_power """
for subjects in subjectList:
    fftprocess.FFT_Processing(subjects, channel, band, window_size, step_size, sample_rate)
""" 3.数据处理  """
data = []
L = []
for subjects in subjectList:
    with open('out1\s' + subjects + '.npy', 'rb') as file:
        sub = np.load(file, allow_pickle=True)  ## 2400*2
        for i in range(0, sub.shape[0]):
            data.append(sub[i][0])
            L.append(sub[i][1])
data = np.array(data)
L = np.array(L)
# 归一化处理，对每个样本计算其p-范数（L1和l2)，然后每个元素除以该范数
data = normalize(data)
print("dataset:", data.shape, L.shape)
np.save('out2\data', data, allow_pickle=True, fix_imports=True) #76800*160=32X40X60*160
np.save('out2\L', L, allow_pickle=True, fix_imports=True) #76800*4

##二分类标签
val = np.ones(L.shape[0])
for i in range(L.shape[0]):
    if L[i][0] <= 5:
        val[i] = -1
    else:
        val[i] = 1
aro = np.ones(L.shape[0])
for i in range(L.shape[0]):
    if L[i][1] <= 5:
        aro[i] = -1
    else:
        aro[i] = 1
print("dataset:", L.shape, aro.shape, val.shape)
np.save('out2\\aro', aro, allow_pickle=True, fix_imports=True) #76800*1
np.save('out2\\val', val, allow_pickle=True, fix_imports=True) #76800*1


""" 4.打乱所有人数据  """
with open('out2\data.npy', 'rb') as file:
    data = np.load(file, allow_pickle=True)  ## 76800*160
with open('out2\L.npy', 'rb') as file:
    L = np.load(file, allow_pickle=True)  ## 76800*4
with open('out2\\val.npy', 'rb') as file:
        val = np.load(file, allow_pickle=True)  ## 76800*4
data_raw = data
val_raw = val
dataindex = np.arange(1280)
# print(dataindex)
np.random.shuffle(dataindex)
# print(dataindex)

for number in range(1280):
    index = dataindex[number]
    # print(index,number)
    data_raw[number*60:(number+1)*60] = data[index*60:(index+1)*60]
    val_raw[number*60:(number+1)*60] = val [index*60:(index+1)*60]
print(data_raw.shape)


print(data_raw is data)
print(dataindex)
for number in range(1280):
    print(data_raw[number*60][0]

""" 5.划分训练集和测试集"""
data_training_raw = data_raw[:960*60]
label_training_raw = val_raw[:960*60]

data_testing = data_raw[960*60:]
label_testing = val_raw[960*60:]
print(label_training_raw[0],label_training_raw[60],label_training_raw[120],label_training_raw[180])
print(label_testing[0],label_testing[60],label_testing[120],label_testing[180])
print(data_testing.shape,data_training_raw.shape)
Val_S = svm.SVC(kernel='rbf', probability=True)
Val_S.fit(data_training_raw , label_training_raw)
acc_valid = Val_S.score(data_testing, label_testing)

print(acc_valid)


