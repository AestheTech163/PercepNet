import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class CombFilter(keras.layers.Layer):
    def __init__(self, min_f0, fs, nfft, batchsize, num_padding_front,
                 len_of_samples, hop_length, comb_order, frames_per_example, comb_window,
                 **kwargs):
        super().__init__(**kwargs)
        self.min_f0 = min_f0
        self.fs = fs
        self.nfft = nfft
        self.batchsize = batchsize
        self.len_of_samples = len_of_samples
        self.num_padding_front = num_padding_front
        self.hop_length = hop_length
        self.comb_window = comb_window
        self.comb_order = comb_order
        self.frames_per_example = frames_per_example
        max_f0_in_samples = fs // min_f0
        num_padding_front = max_f0_in_samples * comb_order
        num_padding_end = max_f0_in_samples * comb_order + nfft
        zero_padding_front_np = np.zeros((batchsize, num_padding_front))
        zero_padding_end_np = np.zeros((batchsize, num_padding_end))
        self.zero_padding_front = tf.constant(zero_padding_front_np, dtype=tf.float32)
        self.zero_padding_end = tf.constant(zero_padding_end_np, dtype=tf.float32)

    def get_comb_indices(self, f0tensor):
        tmp_mask = tf.cast(f0tensor < self.min_f0, tf.float32)
        f0_for_comb = tmp_mask * (10 * self.fs) + (1 - tmp_mask) * f0tensor

        f0_in_samples = tf.cast(tf.round(self.fs / f0_for_comb), tf.int32)
        frames_heads = tf.range(self.num_padding_front,
                                self.len_of_samples + self.num_padding_front - self.nfft + self.hop_length,
                                self.hop_length)
        frames_heads = tf.expand_dims(frames_heads, axis=0)
        frames_heads = tf.tile(frames_heads, [self.batchsize, 1])
        comb_range = tf.range(-self.comb_order, self.comb_order + 1, 1)
        comb_range = tf.broadcast_to(comb_range, [self.batchsize, self.frames_per_example, 2 * self.comb_order + 1])
        f0_in_samples = tf.expand_dims(f0_in_samples, axis=-1)
        comb_begin_indicies = f0_in_samples * comb_range
        frames_heads = tf.expand_dims(frames_heads, axis=-1)
        comb_clips_begin = frames_heads + comb_begin_indicies
        comb_clips_indices = tf.expand_dims(comb_clips_begin, axis=-1) + tf.range(self.nfft)
        return comb_clips_indices

    def do_comb(self, audio_pad, comb_clips_indices):
        comb_slices = tf.gather(audio_pad, comb_clips_indices, axis=1, batch_dims=1)
        comb_window_broadcast = tf.broadcast_to(self.comb_window, comb_slices.shape[:2] + [2 * self.comb_order + 1])
        comb_slices_T = tf.transpose(comb_slices, perm=(0, 1, 3, 2))
        comb_slices_T = tf.stop_gradient(comb_slices_T)
        ycomb = tf.linalg.matvec(comb_slices_T, comb_window_broadcast)
        return ycomb

    def call(self, inputs):
        x_audio, y_audio, f0tensor = inputs

        comb_clips_indices = self.get_comb_indices(f0tensor)
        x_audio_pad = tf.concat([self.zero_padding_front, x_audio, self.zero_padding_end], axis=-1)
        xcomb_frames = self.do_comb(x_audio_pad, comb_clips_indices)

        y_audio_pad = tf.concat([self.zero_padding_front, y_audio, self.zero_padding_end], axis=-1)
        ycomb_frames = self.do_comb(y_audio_pad, comb_clips_indices)

        return xcomb_frames, ycomb_frames


class CombFilter2(keras.layers.Layer):
    def __init__(self, min_f0, fs, nfft, batchsize, num_padding_front,
                 len_of_samples, hop_length, comb_order, frames_per_example, comb_window,
                 **kwargs):
        super().__init__(**kwargs)
        self.min_f0 = min_f0
        self.fs = fs
        self.nfft = nfft
        self.batchsize = batchsize
        self.len_of_samples = len_of_samples
        self.num_padding_front = num_padding_front
        self.hop_length = hop_length
        self.comb_window = comb_window
        self.comb_order = comb_order
        self.frames_per_example = frames_per_example
        max_f0_in_samples = fs // min_f0
        num_padding_front = max_f0_in_samples * comb_order
        num_padding_end = max_f0_in_samples * comb_order + nfft
        zero_padding_front_np = np.zeros((batchsize, num_padding_front))
        zero_padding_end_np = np.zeros((batchsize, num_padding_end))
        self.zero_padding_front = tf.constant(zero_padding_front_np, dtype=tf.float32)
        self.zero_padding_end = tf.constant(zero_padding_end_np, dtype=tf.float32)

    def get_comb_indices(self, f0tensor):
        tmp_mask = tf.cast(f0tensor < self.min_f0, tf.float32)
        f0_for_comb = tmp_mask * (10 * self.fs) + (1 - tmp_mask) * f0tensor

        f0_in_samples = tf.cast(tf.round(self.fs / f0_for_comb), tf.int32)
        frames_heads = tf.range(self.num_padding_front,
                                self.len_of_samples + self.num_padding_front - self.nfft + self.hop_length,
                                self.hop_length)
        frames_heads = tf.expand_dims(frames_heads, axis=0)
        frames_heads = tf.tile(frames_heads, [self.batchsize, 1])
        comb_range = tf.range(-self.comb_order, self.comb_order + 1, 1)
        comb_range = tf.broadcast_to(comb_range, [self.batchsize, self.frames_per_example, 2 * self.comb_order + 1])
        f0_in_samples = tf.expand_dims(f0_in_samples, axis=-1)
        comb_begin_indicies = f0_in_samples * comb_range
        frames_heads = tf.expand_dims(frames_heads, axis=-1)
        comb_clips_begin = frames_heads + comb_begin_indicies
        comb_clips_indices = tf.expand_dims(comb_clips_begin, axis=-1) + tf.range(self.nfft)
        return comb_clips_indices

    def do_comb(self, audio_pad, comb_clips_indices):
        comb_slices = tf.gather(audio_pad, comb_clips_indices, axis=1, batch_dims=1)
        comb_window_broadcast = tf.broadcast_to(self.comb_window, comb_slices.shape[:2] + [2 * self.comb_order + 1])
        comb_slices_T = tf.transpose(comb_slices, perm=(0, 1, 3, 2))
        comb_slices_T = tf.stop_gradient(comb_slices_T)
        ycomb = tf.linalg.matvec(comb_slices_T, comb_window_broadcast)
        return ycomb

    def call(self, inputs):
        y_audio, f0tensor = inputs

        comb_clips_indices = self.get_comb_indices(f0tensor)

        y_audio_pad = tf.concat([self.zero_padding_front, y_audio, self.zero_padding_end], axis=-1)
        ycomb_frames = self.do_comb(y_audio_pad, comb_clips_indices)

        return ycomb_frames
