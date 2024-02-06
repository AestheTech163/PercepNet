import numpy as np

num_examples = 1
fs = 16000
comb_order = 5
min_f0 = 80
max_f0 = 700
max_f0_in_samples = fs // min_f0

nfft = 320
hop_length = 160
batchsize = 32
len_of_seconds = 15  # 每条训练数据的时长
len_of_samples = fs * len_of_seconds
frames_per_example = (len_of_samples - nfft) // hop_length + 1
print("frames_per_example", frames_per_example)

num_bands = 22  # 子带数量
nfftborder = np.array([
    0, 2, 4, 6, 8, 10, 12,
    14, 16, 18, 21, 25, 31,
    37, 45, 55, 66, 79, 94,
    112, 134, 160], dtype=np.int32)  # 子带的频点边界

gb_w = 1.0
rb_w = 1.0


