import tensorflow_io as tfio
from keras import backend as K
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import io
import PIL.Image
import numpy as np


# @tf.function
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    # wav = tf.reshape(wav[:len_of_samples*num_examples], (num_examples, len_of_samples))
    return wav


# @tf.function
def load_csv(filename):
    file_contents = tf.io.read_file(filename)
    split_data = tf.strings.split(file_contents, '\n')[7:-1]
    csv_data = tf.io.decode_csv(split_data,
                                field_delim=' ',
                                record_defaults=[tf.constant(0.0),
                                                 tf.constant(0.0),
                                                 tf.constant(0.0)])
    csv_data = csv_data[2]
    # csv_data = csv_data[:frames_per_example*num_examples]
    # csv_data = tf.reshape(csv_data, (num_examples, frames_per_example))
    return csv_data


def filter_by_f0_len_wrapper(num_examples, frames_per_example):
    # @tf.function
    def filter_by_f0_len(noisy_path, clean_path, noisy_f0_path, noisy_corr_path, clean_f0_path):
        file_contents = tf.io.read_file(noisy_f0_path)
        split_data = tf.strings.split(file_contents, '\n')[7:-1]
        csv_data = tf.io.decode_csv(split_data, field_delim=' ',
                                    record_defaults=[tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)])
        cond = tf.greater(len(csv_data[2]), frames_per_example * num_examples)
        return cond
    return filter_by_f0_len


def load_blob_for_map_wrapper(len_of_samples, frames_per_example):
    # @tf.function
    def load_blob_for_map(noisy_path, clean_path, noisy_f0_path, noisy_corr_path, clean_f0_path):
        y = load_wav_16k_mono(noisy_path)
        num_examples = len(y) // len_of_samples
        y = tf.reshape(y[:len_of_samples * num_examples], (num_examples, len_of_samples))

        x = load_wav_16k_mono(clean_path)
        x = tf.reshape(x[:len_of_samples * num_examples], (num_examples, len_of_samples))

        noisy_f0 = load_csv(noisy_f0_path)
        noisy_f0 = noisy_f0[:frames_per_example * num_examples]
        noisy_f0 = tf.reshape(noisy_f0, (num_examples, frames_per_example))

        noisy_corr = load_csv(noisy_corr_path)
        noisy_corr = noisy_corr[:frames_per_example * num_examples]
        noisy_corr = tf.reshape(noisy_corr, (num_examples, frames_per_example))

        clean_f0 = load_csv(clean_f0_path)
        clean_f0 = clean_f0[:frames_per_example * num_examples]
        clean_f0 = tf.reshape(clean_f0, (num_examples, frames_per_example))

        # file_id = os.path.basename(noisy_path)
        return {"batch_y": y, "batch_x": x, "noisy_f0": noisy_f0, "noisy_corr": noisy_corr, "clean_f0": clean_f0}
    return load_blob_for_map


def get_filepath(noisy_dir, clean_dir, noisy_f0_dir, noisy_corr_dir, clean_f0_dir, noraw=False):
    noisy_files = []
    clean_files = []
    noisy_f0_files = []
    noisy_corr_files = []
    clean_f0_files = []
    for fn in os.listdir(noisy_dir):
        if noraw and (fn.find('wtnoise') < 0) and (fn.find('click') < 0):
            continue
        clean_path = os.path.join(clean_dir, fn)
        if not os.path.exists(clean_path):
            print("Warning::need", clean_path)
            continue
        clean_f0_path = os.path.join(clean_f0_dir, fn[:-3] + "f0")
        if not os.path.exists(clean_f0_path):
            print("Warning::need", clean_f0_path)
            continue
        noisy_f0_path = os.path.join(noisy_f0_dir, fn[:-3] + "f0")
        if not os.path.exists(noisy_f0_path):
            print("Warning::need", noisy_f0_path)
            continue
        noisy_corr_path = os.path.join(noisy_corr_dir, fn[:-3] + "corr")
        if not os.path.exists(noisy_corr_path):
            print("Warning::need", noisy_corr_path)
            continue
        noisy_files.append(os.path.join(noisy_dir, fn))
        clean_files.append(clean_path)
        noisy_f0_files.append(noisy_f0_path)
        noisy_corr_files.append(noisy_corr_path)
        clean_f0_files.append(clean_f0_path)
    print("Total examples::", len(noisy_files))
    return noisy_files, clean_files, noisy_f0_files, noisy_corr_files, clean_f0_files


def gen_plot(y, y_hat):
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(y_hat.T, interpolation='none', cmap=plt.cm.jet, origin='lower', aspect='auto')
    plt.subplot(1, 2, 2)
    plt.imshow(y.T, interpolation='none', cmap=plt.cm.jet, origin='lower', aspect='auto')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def plot_to_board(step, y, y_hat, title, file_writer):
    plot_buf = gen_plot(y, y_hat)
    image = PIL.Image.open(plot_buf)
    image = np.expand_dims(image, axis=0)
    with file_writer.as_default():
        tf.summary.image(title, image, step=step)
