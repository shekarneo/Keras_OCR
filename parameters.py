CHAR_VECTOR = "-/.,?!'\"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
letters = [letter for letter in CHAR_VECTOR]
num_classes = len(letters) + 1
img_w, img_h = 128, 64
val_batch_size = 16
downsample_factor = 4
# Network parameters
conv_filters = [32, 64, 128, 256, 512]
kernel_size = [(5, 5), (3, 3)]
pool_size = [(2, 2), (1, 2), (2, 1)]
rnn_size = 512
max_text_len = 32
input_shape = (img_w, img_h, 1)