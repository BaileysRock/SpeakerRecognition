import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import os

from datasets import gen_file_paths
from tqdm import tqdm


def rebuild(outfile, rate, array):
    # Write out audio as 24bit PCM WAV
    sf.write(f"{outfile}.wav", array, rate, subtype="PCM_24")

    # Write out audio as 24bit Flac
    # sf.write(f"{outfile}.flac", array, rate, format="flac", subtype="PCM_24")

    # Write out audio as 16bit OGG
    # sf.write(f"{outfile}.ogg", array, rate, format="ogg", subtype="vorbis")


def get_mel_feature(infile, rate):
    y, _ = librosa.load(infile, sr=rate)

    yy = librosa.resample(
        y,
        rate,
        rate * 2.0833333 / librosa.get_duration(y=y, sr=rate),
    )
    mm = librosa.feature.mfcc(yy, sr=rate, n_mfcc=128)

    return mm  # Expected: mm.shape == (128, 180)


def dump_mel_feature(sr: int):
    human_id_list = os.listdir("./dataset/")
    for i in tqdm(range(len(human_id_list))):
        audio_paths = gen_file_paths(human_id_list[i])
        for path in audio_paths:
            features = get_mel_feature(path, sr)
            np.savetxt(path.replace(".flac", ".csv"), features.reshape(1, -1))


def load_mel_feature(human_count: int) -> list[tuple[np.array, int]]:
    feature_label = []

    human_id_list = os.listdir("./dataset/")
    for i in tqdm(range(min(len(human_id_list), human_count))):
        audio_paths = gen_file_paths(human_id_list[i])
        for path in audio_paths:
            features = np.loadtxt(path.replace(".flac", ".csv")).reshape(128, 180)
            label = int(path.split("/")[-3])
            feature_label.append((features, label))

    return feature_label
