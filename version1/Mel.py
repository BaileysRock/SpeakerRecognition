"""
数据集使用 LibriSpeech 的 train-clean-100，在本项目结构中的目录结构为：

SpeakerRecognition                  - 项目根目录
├── dataset                         - 数据集所在目录
│   ├── 19                          - 讲话人 id
│   │   ├── 198
│   │   │   ├── 19-198-0000.flac    - 音频文件
│   │   │   ├── 19-198-0000.csv     - 音频文件对应的 mfcc 特征矩阵(由 load_mel_feature 函数生成)
│   │   │   └── ...
│   │   └── 227
│   ├── 26
│   │   ├── 495
│   │   └── 496
│   └── ...
│
└── version1                        - 代码
    └── ...py

在本文件中，默认 pwd 为 SpeakerRecognition/
"""

import librosa
import librosa.display
import soundfile as sf
import numpy as np
import os

from tqdm import tqdm


def gen_file_paths(human_id: str) -> list[str]:
    """
    指定说话人 id，生成这个人所有音频文件的路径列表

    :param human_id: 说话人 id
    """

    paths: list[str] = []

    for _, dirs, _ in os.walk("./dataset/" + human_id + "/"):
        if len(dirs) == 0:
            continue
        for dir in dirs:
            for path, _, filename in os.walk("./dataset/" + human_id + "/" + dir + "/"):
                for zz in filename:
                    if zz.endswith(".flac"):
                        paths.append(path + zz)

    return paths


def rebuild(outfile: str, rate: int, array: np.array):
    """
    根据 np.array 类型的向量重建音频文件

    :param outfile: 重建的音频文件将被写入到 outfile.wav 中
    :param rate: 采样率
    :param array: 音频向量
    """

    # Write out audio as 24bit PCM WAV
    sf.write(f"{outfile}.wav", array, rate, subtype="PCM_24")

    # Write out audio as 24bit Flac
    # sf.write(f"{outfile}.flac", array, rate, format="flac", subtype="PCM_24")

    # Write out audio as 16bit OGG
    # sf.write(f"{outfile}.ogg", array, rate, format="ogg", subtype="vorbis")


def get_mel_feature(infile, rate):
    """
    从 infile 文件中读取音频，并提取其中的 mfcc 特征

    :param infile: 输入音频文件
    :param rate: 采样率
    :return: 返回的 mfcc 特征矩阵，其 shape == (128, 180)
    """

    y, _ = librosa.load(infile, sr=rate)

    yy = librosa.resample(
        y,
        rate,
        rate * 2.0833333 / librosa.get_duration(y=y, sr=rate),
    )
    mm = librosa.feature.mfcc(yy, sr=rate, n_mfcc=128)

    return mm  # Expected: mm.shape == (128, 180)


def dump_mel_feature(sr: int):
    """
    提取数据集中所有音频文件的 mfcc 特征矩阵，
    并保存到与音频文件名同名的 csv 文件中

    :param sr: 采样率
    """

    human_id_list = os.listdir("./dataset/")
    for i in tqdm(range(len(human_id_list))):
        audio_paths = gen_file_paths(human_id_list[i])
        for path in audio_paths:
            features = get_mel_feature(path, sr)
            np.savetxt(path.replace(".flac", ".csv"), features.reshape(1, -1))


def load_mel_feature(human_count: int) -> list[tuple[np.array, int]]:
    """
    从 csv 文件中加载特征

    :param human_count: 加载多少人的音频特征
    :return: 返回一个 list，其中每个元素是一个 tuple，
             tuple[0] 是音频的 mfcc 特征矩阵，tuple[1] 是音频的标签
    """

    feature_label = []

    human_id_list = os.listdir("./dataset/")
    for i in tqdm(range(min(len(human_id_list), human_count))):
        audio_paths = gen_file_paths(human_id_list[i])
        for path in audio_paths:
            features = np.loadtxt(path.replace(".flac", ".csv")).reshape(128, 180)
            label = int(path.split("/")[-3])
            feature_label.append((features, label))

    return feature_label
