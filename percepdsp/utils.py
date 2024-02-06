import math
import librosa
import librosa.feature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def get_erb_filterbank(num_bands, nfft, nfftborder):
    dense_filterbank = np.zeros((num_bands, nfft // 2 + 1), dtype=np.float32)
    for i in range(len(nfftborder) - 1):
        band_size = nfftborder[i + 1] - nfftborder[i]
        for j in range(band_size):
            frac = 1.0 * j / band_size
            dense_filterbank[i, nfftborder[i] + j] += 1 - frac
            dense_filterbank[i + 1, nfftborder[i] + j] += frac
    return dense_filterbank.T


def get_comb_hann_window(comb_half_order):
    comb_hann_window = np.zeros(comb_half_order * 2 + 1)
    temp_sum = 0
    for i in range(1, comb_half_order * 2 + 2, 1):
        comb_hann_window[i - 1] = 0.5 - 0.5 * math.cos(2.0 * math.pi * i / (comb_half_order * 2 + 2))
        temp_sum += comb_hann_window[i - 1]
    for i in range(1, comb_half_order * 2 + 2, 1):
        comb_hann_window[i - 1] /= temp_sum
    return comb_hann_window


def get_vorbis_window(nfft):
    vorbis_window = np.zeros(nfft, dtype=np.float32)
    for i in range(nfft // 2):
        vorbis_window[i] = np.sin(.5 * np.pi * np.sin(.5 * np.pi * (i + .5) / (nfft // 2)) ** 2)
        vorbis_window[nfft - i - 1] = vorbis_window[i]
    return vorbis_window


def show_melspec(y=None, S=None, fp=None, size=(30, 5), title="", pause=0, n_fft=960 // 3, hop_length=480 // 3,
                 win_length=960 // 3, window='hann'):
    if S is not None:
        y = librosa.istft(S,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          window=window,
                          center=False)
    elif fp is not None:
        y, fs = librosa.load(fp, sr=None)

    else:
        assert y is not None, "either y or S must be non empty"

    fm = matplotlib.font_manager.FontProperties()
    fm.set_size(30)

    fig, ax = plt.subplots(figsize=size)
    tmp_melspec = librosa.feature.melspectrogram(y=y, n_mels=340 // 3, power=2)  # n_fft=2048, hop_length=512
    tmp_mel_dB = librosa.power_to_db(tmp_melspec, ref=np.max)
    Nimg = librosa.display.specshow(tmp_mel_dB, x_axis='time', y_axis='mel')  # , sr=fs, fmax=fs//2, ax=ax)
    fig.colorbar(Nimg, ax=ax, format='%+2.0f dB')
    ax.set_title(title, fontproperties=fm)
    plt.draw()
    plt.savefig(title + ".jpg")
    return title + ".jpg"

