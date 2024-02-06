import os
import PIL.Image
import librosa.display
import soundfile as sf
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint

from training_utils import filter_by_f0_len_wrapper, load_blob_for_map_wrapper, get_filepath, plot_to_board
from config import *
from percepdsp.utils import show_melspec, get_erb_filterbank, get_vorbis_window, get_comb_hann_window
from percepdsp.comb_filter import CombFilter, CombFilter2
from percepnet import build_feature_and_gb_rb_model, build_percepNet, gbcost, rbcost

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, enable=True)

assert num_bands == len(nfftborder)

# path_to_train_y = "../data/noisetrain/train/noisy"
# path_to_train_x = "../data/noisetrain/train/clean"
# path_to_train_f0 = "../data/noisetrain/train/clean_f0"

# path_to_val_y = "../data/noisetrain/val/noisy"
# path_to_val_x = "../data/noisetrain/val/clean"
# path_to_val_f0 = "../data/noisetrain/val/clean_f0"

path_to_train_y = "../data/smallset/train/noisy"
path_to_train_x = "../data/smallset/train/clean"
path_to_train_clean_f0 = "../data/smallset/train/clean_f0"
path_to_train_noisy_f0 = "../data/smallset/train/noisy_f0"

path_to_val_y = "../data/smallset/val/noisy"
path_to_val_x = "../data/smallset/val/clean"
path_to_val_clean_f0 = "../data/smallset/val/clean_f0"
path_to_val_noisy_f0 = "../data/smallset/val/noisy_f0"


# noisy_dir, clean_dir, noisy_f0_dir, noisy_corr_dir, clean_f0_dir
train_filepath = get_filepath(path_to_train_y,
                              path_to_train_x,
                              path_to_train_noisy_f0,
                              path_to_train_noisy_f0,
                              path_to_train_clean_f0,
                              noraw=True)

val_filepath = get_filepath(path_to_val_y,
                            path_to_val_x,
                            path_to_val_noisy_f0,
                            path_to_val_noisy_f0,
                            path_to_val_clean_f0,
                            noraw=True)

train_set = tf.data.Dataset.from_tensor_slices(train_filepath)

train_set = train_set.filter(filter_by_f0_len_wrapper(num_examples, frames_per_example))
train_set.shuffle(100)
train_set = train_set.repeat()
train_set = train_set.map(load_blob_for_map_wrapper(len_of_samples, frames_per_example), num_parallel_calls=4)
train_set = train_set.unbatch()
train_set = train_set.batch(batchsize)

val_set = tf.data.Dataset.from_tensor_slices(val_filepath)
val_set = val_set.filter(filter_by_f0_len_wrapper(num_examples, frames_per_example))
val_set = val_set.map(load_blob_for_map_wrapper(len_of_samples, frames_per_example), num_parallel_calls=4)
val_set = val_set.unbatch()
val_set = val_set.batch(batchsize)

num_padding_front = max_f0_in_samples * comb_order
comb_hann_window_np = get_comb_hann_window(comb_order)
comb_window = tf.constant(comb_hann_window_np, dtype=tf.float32)

erbbank = get_erb_filterbank(num_bands, nfft, nfftborder)
comb_filter = CombFilter(min_f0, fs, nfft, batchsize, num_padding_front,
                         len_of_samples, hop_length, comb_order,
                         frames_per_example, comb_window)
comb_filter2 = CombFilter2(min_f0, fs, nfft, batchsize, num_padding_front,
                         len_of_samples, hop_length, comb_order,
                         frames_per_example, comb_window)
feature_model = build_feature_and_gb_rb_model(batchsize,
                                              len_of_samples,
                                              frames_per_example,
                                              erbbank,
                                              comb_hann_window_np)
gb_rb_model = build_percepNet()
optimizer = optimizers.Adam(learning_rate=1e-5)


runName = "0316v1-percepnet-16k-exp2"
savePath = f"./models_{runName}/"
os.makedirs(savePath, exist_ok=True)

epochs = 500

steps_per_epoch = len(train_filepath[0]) // batchsize
val_steps = len(val_filepath[0]) // batchsize
# for testing
# steps_per_epoch = 40
# val_steps = 20
print("epochs", epochs)
print("steps_per_epoch", steps_per_epoch)
print("val_steps", val_steps)

train_loss_results = []
val_loss_results = []

epoch_loss_avg = tf.keras.metrics.Mean()
epoch_gb_loss_avg = tf.keras.metrics.Mean()
epoch_rb_loss_avg = tf.keras.metrics.Mean()
val_loss_avg = tf.keras.metrics.Mean()
val_gb_loss_avg = tf.keras.metrics.Mean()
val_rb_loss_avg = tf.keras.metrics.Mean()

logdir = os.path.join(savePath, "log")
file_writer = tf.summary.create_file_writer(logdir)


@tf.function
def train_step(inputs):
    with tf.device("/gpu:1"):
        feature, ideal_gb, ideal_rb, Y, P_hat = feature_model(inputs, training=False)
    with tf.GradientTape() as tape:
        gb_hat, rb_hat = gb_rb_model(feature, training=True)
        gb_loss = gbcost(ideal_gb, gb_hat)
        rb_loss = rbcost(ideal_rb, rb_hat)
        loss_value = gb_w * gb_loss + rb_w * rb_loss
    grads = tape.gradient(loss_value, gb_rb_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, gb_rb_model.trainable_weights))

    # Update training metric.
    epoch_loss_avg.update_state(loss_value)
    epoch_gb_loss_avg.update_state(gb_loss)
    epoch_rb_loss_avg.update_state(rb_loss)

    return loss_value


def val_step(val_set):
    # Run a validation loop at the end of each epoch.
    for i, inputs in enumerate(val_set):
        feature, ideal_gb, ideal_rb, Y, P_hat = feature_model(inputs, training=False)
        gb_hat, rb_hat = gb_rb_model(feature, training=False)
        gb_loss = gbcost(ideal_gb, gb_hat)
        rb_loss = rbcost(ideal_rb, rb_hat)
        loss_value = gb_w * gb_loss + rb_w * rb_loss

        val_loss_avg.update_state(loss_value)  # Add current batch loss
        val_gb_loss_avg.update_state(gb_loss)
        val_rb_loss_avg.update_state(rb_loss)
        if i >= val_steps:
            break

    print("[Eval] Epoch:%d, AvgValLoss:%.5f, gb_loss:%.5f, rb_loss:%.5f\r"
          % (epoch,
             val_loss_avg.result(),
             val_gb_loss_avg.result(),
             val_rb_loss_avg.result(),
             ),
          flush=True
          )
    return ideal_gb, ideal_rb, gb_hat, rb_hat, Y, P_hat


# training loop
vidx = 3
for step, inputs in enumerate(train_set):
    epoch = step // steps_per_epoch + 1
    if epoch > epochs:
        print("[Train] Done!")
        break

    batch_loss = train_step(inputs)

    if step % 20 == 19:
        print("[Train] Epoch:%d, GlobalSteps:%d, BatchLoss:%.5f, TotalAvgLoss:%.5f, gb_loss:%.5f, rb_loss:%.5f\r"
              % (epoch, step + 1, float(batch_loss),
                 epoch_loss_avg.result(),
                 epoch_gb_loss_avg.result(),
                 epoch_rb_loss_avg.result(),
                 ),
              flush=True
              )

    if (step % steps_per_epoch) == (steps_per_epoch - 1):
        # End epoch
        train_loss_results.append(epoch_loss_avg.result())

        batch_gb, batch_rb, batch_gb_hat, batch_rb_hat, Y, P_hat = val_step(val_set)

        print("Saving model...", flush=True)
        gb_rb_model.save(os.path.join(savePath, f"epoch{epoch}_base_percep_emb.keras"))

        batch_gb = batch_gb.numpy()[vidx, :, :]
        batch_rb = batch_rb.numpy()[vidx, :, :]
        batch_gb_hat = batch_gb_hat.numpy()[vidx, :, :]
        batch_rb_hat = batch_rb_hat.numpy()[vidx, :, :]

        plot_to_board(step + 1, batch_gb, batch_gb_hat, "gb, gb_hat")
        plot_to_board(step + 1, batch_rb, batch_rb_hat, "rb, rb_hat")

        with file_writer.as_default():
            tf.summary.scalar('Train gb loss', epoch_gb_loss_avg.result(), step=step)
            tf.summary.scalar('Train rb loss', epoch_rb_loss_avg.result(), step=step)
            tf.summary.scalar('Eval gb loss', val_gb_loss_avg.result(), step=step)
            tf.summary.scalar('Eval rb loss', val_rb_loss_avg.result(), step=step)

        # Reset training metrics at the end of each epoch
        epoch_loss_avg.reset_states()
        epoch_gb_loss_avg.reset_state()
        epoch_rb_loss_avg.reset_state()

        val_loss_avg.reset_state()
        val_gb_loss_avg.reset_state()
        val_rb_loss_avg.reset_state()

        # test_filename = "/data/voice/wentao/data/DNS-Challenge-2020/noisetrain/val/noisy/fileid_9979.wav"
        # test_y, fs_tmp = librosa.load(test_filename, sr=fs)
        # assert fs_tmp==fs

        # y_slice = last_inputs['batch_y'][idx,:,:]
        # x_slice = last_inputs['batch_x'][idx,:,:]
        # f0_slice = last_inputs['f0'][idx,:]
        # corr_slice = last_inputs['corr'][idx,:]

        rb_hat_test = batch_rb_hat
        gb_hat_test = batch_gb_hat

        Y = Y[vidx, :, :]
        P_hat = P_hat[vidx, :, :]

        rb_hat_interp = tf.matmul(rb_hat_test, erbbank.T)
        inv_rb_hat_interp = (1.0 - rb_hat_interp)
        Y_filtered = Y * tf.cast(inv_rb_hat_interp, tf.complex64)
        P_hat_filtered = P_hat * tf.cast(rb_hat_interp, tf.complex64)
        rb_hat_filtered_Y = Y_filtered + P_hat_filtered

        gb_hat_interp = tf.matmul(gb_hat_test, erbbank.T)
        gb_hat_interp = tf.cast(gb_hat_interp, tf.complex64)
        rb_hat_gb_hat_denoised_Y = gb_hat_interp * rb_hat_filtered_Y

        rb_hat_gb_hat_denoised_y = librosa.istft(rb_hat_gb_hat_denoised_Y.numpy().T,
                                                 n_fft=nfft,
                                                 hop_length=hop_length,
                                                 win_length=nfft,
                                                 window=get_vorbis_window(nfft),
                                                 center=False)

        rb_hat_gb_hat_denoised_y_file_path = "./tmp_denoise-16k-rb_hat-gb_hat-f0y.wav"
        sf.write(rb_hat_gb_hat_denoised_y_file_path, rb_hat_gb_hat_denoised_y, fs)

        jpgfile = show_melspec(fp=rb_hat_gb_hat_denoised_y_file_path, size=(12, 4),
                               title="Denoised with rb-hat then gb-hat", pause=0, window=get_vorbis_window(nfft))

        image = PIL.Image.open(jpgfile)
        image = np.expand_dims(image, axis=0)
        with file_writer.as_default():
            tf.summary.image(jpgfile, image, step=step)
