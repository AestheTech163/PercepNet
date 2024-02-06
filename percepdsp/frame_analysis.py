import tensorflow.keras as keras
import tensorflow as tf


class FrameAnalysis(keras.layers.Layer):
    def __init__(self, erbbank, do_framing=True, n_fft=960 // 3, hop_length=480 // 3,
                 window_fn=tf.signal.vorbis_window, pad_end=False, name=None):
        super().__init__(name=name)
        self.erbbank = erbbank
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.pad_end = pad_end

        if do_framing:
            self.stft_layer = keras.layers.Lambda(self.keras_stft)
        else:
            self.window = window_fn(n_fft)
            self.stft_layer = keras.layers.Lambda(self.my_stft)

    def keras_stft(self, x):
        X = tf.signal.stft(x,
                           fft_length=self.n_fft,
                           frame_length=self.n_fft,
                           frame_step=self.hop_length,
                           window_fn=self.window_fn,
                           pad_end=self.pad_end,
                           name=self.name)
        return X

    def my_stft(self, frames):
        frames_windowed = self.window * frames
        X = tf.signal.rfft(frames_windowed)
        return X

    def call(self, inputs):
        cpx_spec = self.stft_layer(inputs)
        pow_spec = tf.cast(cpx_spec * tf.math.conj(cpx_spec), tf.float32)
        erbbands = tf.matmul(pow_spec, self.erbbank)
        return cpx_spec, erbbands

