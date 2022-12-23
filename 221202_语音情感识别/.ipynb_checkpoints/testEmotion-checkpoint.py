import os
from random import shuffle
from train import getFeature
from drawRadar import draw
from sklearn.externals import joblib
import numpy as np
import pyaudio
import wave

path = r'C:\Github\svm_test\casia'

wav_paths = []

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
            wav_paths.append(wav_path)

# 将语音文件随机排列
shuffle(wav_paths)

model = joblib.load("classfier.m")

p = pyaudio.PyAudio()
for wav_path in wav_paths:
    f = wave.open(wav_path, 'rb')
    stream = p.open(
        format=p.get_format_from_width(f.getsampwidth()),
        channels=f.getnchannels(),
        rate=f.getframerate(),
        output=True)
    data = f.readframes(f.getparams()[3])
    stream.write(data)
    stream.stop_stream()
    stream.close()
    f.close()
    data_feature = getFeature(wav_path, 48)
    print(model.predict([data_feature]))
    print(model.predict_proba([data_feature]))
    labels = np.array(['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'])

    draw(model.predict_proba([data_feature])[0], labels, 6)

p.terminate()
