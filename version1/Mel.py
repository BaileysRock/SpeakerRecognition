import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np


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
    mm = librosa.feature.mfcc(yy, sr=rate, n_mfcc=1024)

    return mm  # Expected: mm.shape == (1024, 180)


if __name__ == "__main__":
    SAMPLE_RATE = 44100
    print(get_mel_feature("./2-3.wav", SAMPLE_RATE).shape)

    # plt.figure("原始信号")
    # plt.title("Signal")
    # plt.xlabel("Time (samples)")
    # plt.ylabel("Amplitude")
    # plt.plot(y)

    # plt.figure("快速傅立叶变换")
    # n_fft = 2048
    # ft = np.abs(librosa.stft(y[:n_fft], hop_length=n_fft + 1))
    # plt.plot(ft)
    # plt.title("Spectrum")
    # plt.xlabel("Frequency Bin")
    # plt.ylabel("Amplitude")

    # plt.figure("频谱图")
    # spec = np.abs(librosa.stft(y, hop_length=512))
    # spec = librosa.amplitude_to_db(spec, ref=np.max)
    # librosa.display.specshow(spec, sr=sample_rate, x_axis="time", y_axis="log")
    # plt.colorbar(format="%+2.0f dB")
    # plt.title("Spectrogram")

    # plt.figure("Mel谱图")
    # spec = np.abs(librosa.stft(y, hop_length=512))
    # mel_spect = librosa.feature.melspectrogram(
    #     y=y, sr=sample_rate, n_fft=2048, hop_length=1024
    # )
    # mel_spect = librosa.power_to_db(spec, ref=np.max)
    # librosa.display.specshow(mel_spect, y_axis="mel", fmax=8000, x_axis="time")
    # plt.title("Mel Spectrogram")
    # plt.colorbar(format="%+2.0f dB")

    # mm = librosa.feature.mfcc(yy1, sr=sample_rate, n_mfcc=128)
    # print(mm)
    # print("shape = " + str(mm.shape))
    # mm = librosa.feature.mfcc(y2, sr=sample_rate, n_mfcc=128)
    # print(mm)
    # print("shape = " + str(mm.shape))

    # plt.show()
