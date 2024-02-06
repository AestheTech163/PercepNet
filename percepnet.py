import numpy as np
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, LSTM, GRU, Dropout, Embedding
from keras.constraints import Constraint
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Multiply, Layer, Conv1D, Conv2D, Reshape, Concatenate
from percepdsp.comb_filter import CombFilter
from percepdsp.frame_analysis import FrameAnalysis


def tf_compute_band_corr(X, P, X_erb, P_erb, erb_bank):
    corr = tf.math.real(tf.math.conj(X) * P)
    bands_corr = tf.matmul(corr, erb_bank)
    bands_corr = bands_corr/(tf.sqrt(X_erb*P_erb)+1e-7)
    return bands_corr


def tf_compute_band_y_phat(Y, P_hat, Y_erb, erb_bank):
    corr = tf.math.real(tf.math.conj(Y) * P_hat)
    bands_corr = tf.matmul(corr, erb_bank)
    bands_corr = bands_corr/(Y_erb+1e-7)
    return bands_corr


def tf_compute_r_with_author_formulae(X, Y, P, X_erb, Y_erb, P_erb, sigma_square, erb_bank):
    # q_x
    q_x = tf_compute_band_corr(X, P, X_erb, P_erb, erb_bank)
    q_x_square = q_x * q_x
    # q_y
    q_y = tf_compute_band_corr(Y, P, Y_erb, P_erb, erb_bank)
    q_y_square = q_y * q_y
    # q_phat
    under_sqrt = tf.maximum(0, (1-sigma_square)*q_y_square + sigma_square)
    q_phat = q_y / (tf.sqrt(under_sqrt)+1e-7)
    # a, b, c
    a = tf.maximum(0, q_phat*q_phat - q_x_square)
    b = q_phat*q_y*(1-q_x_square)
    c = q_x_square - q_x_square
    # r
    under_sqrt = tf.maximum(0, b*b + a*c)
    alpha = (np.sqrt(under_sqrt)-b)/(a+1e-15)
    r = alpha / (1 + alpha)
    r = tf.minimum(1, tf.maximum(0, r))
    return r, q_phat, q_x


def tf_compute_r_with_my_formulae(X,     Y,     P,     P_hat,
                                  X_erb, Y_erb, P_erb, P_hat_erb,
                                  sigma_square, erb_bank):

    q_x = tf_compute_band_corr(X, P, X_erb, P_erb, erb_bank)
    q_x_square = q_x * q_x

    q_y = tf_compute_band_corr(Y, P, Y_erb, P_erb, erb_bank)
    q_y_square = q_y * q_y

    corr_y_phat_prime = tf_compute_band_y_phat(Y, P_hat, Y_erb, erb_bank)

    under_sqrt = tf.maximum(0., (1.-sigma_square) * q_y_square + sigma_square)
    q_phat = q_y / (tf.sqrt(under_sqrt)+1e-7)

    d = tf.sqrt(P_hat_erb/(Y_erb+1e-7))
    a = q_phat*q_phat - q_x_square
    b = q_phat*q_y*d - q_x_square * corr_y_phat_prime
    c = q_x_square - q_y_square

    under_sqrt = tf.maximum(0., b*b + d*d * a*c)
    a = tf.maximum(a, 0.)
    alpha = (tf.sqrt(under_sqrt)-b)/(d*d * a + 1e-7)

    r = alpha / (1. + alpha)
    r = tf.minimum(1., tf.maximum(0., r))

    return r, q_phat, q_x


def tf_adjust_r_g(r, g, q_phat, q_x):
    n0 = 0.03
    C = 1. + n0
    mask = tf.cast(q_phat < q_x, tf.float32)
    numerator = tf.maximum(0.0, C - q_x*q_x)
    denominator = tf.maximum(0.0, C - q_phat*q_phat) + 1e-7
    g_att = tf.sqrt(numerator/denominator)
    r_adjust = r * (1.0-mask) + mask * 0.999
    g_adjust = mask * g * g_att + g * (1.-mask)
    g_adjust = tf.minimum(1.0, g_adjust)
    return r_adjust, g_adjust


def build_feature_and_gb_rb_model(batchsize, len_of_samples, frames_per_example,
                                  erb_bank, comb_filter, comb_filter2,
                                  comb_hann_window_np, min_f0):
    y = Input(batch_shape=(batchsize, len_of_samples), name='batch_y')
    x = Input(batch_shape=(batchsize, len_of_samples), name='batch_x')

    noisy_f0 = Input(batch_shape=(batchsize, frames_per_example), name='noisy_f0')
    noisy_corr = Input(batch_shape=(batchsize, frames_per_example), name='noisy_corr')
    clean_f0 = Input(batch_shape=(batchsize, frames_per_example), name='clean_f0')

    # computing ideal subband gains and ideal subband pitch strength
    # note: using clean f0 to calculate ideal pitch strength (r)
    xcomb_frames, ycomb_frames = comb_filter((x, y, clean_f0))

    Y, Y_erb = FrameAnalysis(erb_bank, do_framing=True)(y)
    X, X_erb = FrameAnalysis(erb_bank, do_framing=True)(x)

    P_hat, P_hat_erb = FrameAnalysis(erb_bank, do_framing=False)(ycomb_frames)
    P, P_erb = FrameAnalysis(erb_bank, do_framing=False)(xcomb_frames)

    sigma_square = np.sum(comb_hann_window_np ** 2)
    r, q_phat, q_x = tf_compute_r_with_my_formulae(X, Y, P, P_hat, X_erb, Y_erb, P_erb, P_hat_erb, sigma_square)

    ideal_subband_gains = tf.math.sqrt(X_erb / (1e-7 + Y_erb))
    ideal_subband_gains = tf.maximum(tf.minimum(ideal_subband_gains, 1.), 0.)

    adjusted_r, adjusted_g = tf_adjust_r_g(r, ideal_subband_gains, q_phat, q_x)

    # using noisy pitch as features and using noisy pitch to calculate comb signal
    ycomb_2_frames = comb_filter2((y, noisy_f0))
    P_hat_2, P_hat_erb_2 = FrameAnalysis(erb_bank, do_framing=False)(ycomb_2_frames)
    corr_Y_Phat_2 = tf_compute_band_corr(Y, P_hat_2, Y_erb, P_hat_erb_2, erb_bank)

    valid_noisy_f0_mask = tf.cast(noisy_f0 > min_f0, tf.float32)
    noisy_f0_fixed = valid_noisy_f0_mask * noisy_f0
    noisy_f0_normalize = tf.expand_dims(noisy_f0_fixed / 700, axis=-1)

    noisy_corr_expand = tf.expand_dims(noisy_corr, axis=-1)
    band_norms = tf.math.log(tf.sqrt(Y_erb) + 1e-7)

    # concat all features
    main_input = tf.concat([band_norms, corr_Y_Phat_2, noisy_f0_normalize, noisy_corr_expand], axis=-1)
    print("main_input", main_input.shape)

    feature_model = Model(inputs=[y, x, noisy_f0, noisy_corr, clean_f0],
                          outputs=[main_input, adjusted_g, adjusted_r, Y, P_hat])

    return feature_model


def gbcost(y_true, y_pred):
    return K.mean(10 * (K.sqrt(y_pred) - K.sqrt(y_true)) ** 4 +
                  (K.sqrt(y_pred) - K.sqrt(y_true)) ** 2)


def rbcost(y_true, y_pred):
    return K.mean(K.square(K.sqrt(1 - y_pred) - K.sqrt(1 - y_true)))


class WeightClip(Constraint):
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


def conti_f0_repr(f0):
    # continuous f0 representation
    freqs = tf.linspace(0., 700., 128)
    freqs = tf.cast(freqs, tf.float32)
    f0ex = tf.expand_dims(f0, axis=-1)
    grid = tf.multiply(freqs, 1 / f0ex)
    # while training, perterbing
    # perterb = tf.random.normal(grid.shape, mean=0, stddev=0.03, dtype=tf.float32)
    # f0repr = (0.5 - 0.5*tf.cos(grid*np.pi))**4 + perterb
    # testing no perterb
    # f0repr = (0.5 - 0.5*tf.cos(grid*np.pi))**8
    f0repr = (tf.cos(grid * np.pi)) ** 8

    f0repr_stoped = tf.stop_gradient(f0repr)
    return f0repr_stoped


def build_percepNet():
    # reg = 0.000001
    # constraint = WeightClip(0.499)
    numUnits = 256

    print('Build Small PercepNet model...')
    # initializer = initializers.RandomUniform(minval=-0.1, maxval=0.1)

    main_input = Input(shape=(None, 46), name='main_input')
    # tmp = Dense(128, activation='tanh', name='input_dense')(main_input)
    feat = tf.gather(main_input, axis=-1, indices=np.arange(44))

    f0 = tf.gather(main_input, axis=-1, indices=[44])
    f0 = tf.squeeze(f0)
    corr = tf.gather(main_input, axis=-1, indices=[45])
    # corr = tf.squeeze(corr)
    scale = tf.maximum(corr, 0.0)

    # using pitch embedding
    # f0_sqz = tf.cast((tf.log(x)-np.log(80.))/(np.log(700.1)-np.log(80.))*256, tf.int32)
    # f0_embed_layer = Embedding(256, 64, name='f0_embed')
    # f0_embed = f0_embed_layer(f0_sqz) * corr
    f0_repr = conti_f0_repr(f0)
    f0_vector = Dense(64, use_bias=False, activation='linear', name='f0_basis')(f0_repr)
    f0_vector = f0_vector * scale

    aug_feat = Concatenate()([feat, f0_vector])

    cnn1 = Conv1D(512, kernel_size=5,
                  strides=1,
                  use_bias=False,
                  activation='tanh',
                  padding='causal',
                  name='conv_layer1')(aug_feat)

    cnn2 = Conv1D(512, kernel_size=3,
                  strides=1,
                  use_bias=False,
                  activation='tanh',
                  padding='causal',
                  name='conv_layer2')(cnn1)

    # gru 512
    gru1 = GRU(numUnits, return_sequences=True, name='noise_gru1',
               )(cnn2)
    # gru 512
    gru2 = GRU(numUnits, return_sequences=True, name='noise_gru2',
               )(gru1)
    # gru 512
    gru3 = GRU(numUnits, return_sequences=True, name='noise_gru3',
               )(gru2)
    # gru 512
    gru4 = GRU(numUnits, return_sequences=True, name='noise_gru4',
               )(gru3)

    # gb
    gb_input = keras.layers.concatenate([cnn2, gru1, gru2, gru3, gru4])
    gb_output = Dense(22, activation='sigmoid', name='denoise_output')(gb_input)

    # rb
    rb_input = keras.layers.concatenate([cnn2, gru3])
    rb_input2 = GRU(numUnits, return_sequences=True, name='denoise_gru',
                    )(rb_input)
    rb_output = Dense(22, activation='sigmoid', name='denoise_rb_output')(rb_input2)

    gr_model = Model(inputs=main_input, outputs=[gb_output, rb_output])

    return gr_model
