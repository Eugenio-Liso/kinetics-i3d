import tensorflow as tf
import numpy as np
import i3d
import glob
import sonnet as snt

# CODE TAKEN FROM: https://github.com/deepmind/kinetics-i3d/issues/1

_SAMPLE_VIDEO_FRAMES = 64
_IMAGE_SIZE = 224
_NUM_CLASSES = 101
_EPOCHS = 10
_BATCH_SIZE = 4

_FILE_LOC_TRAIN = glob.glob("data/train/*.npy")
print("[Total Files: {}]".format(len(_FILE_LOC_TRAIN)))

_MEAN_DATA = np.load("data/mean_data__ucf.npy")[np.newaxis, :, :, :, :]
TRAINING = True
print("Mean_Data: {}".format(_MEAN_DATA.shape))

rgb_input = tf.placeholder(
    tf.float32,
    shape=(None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

y_true = tf.placeholder(
    tf.float32,
    shape=(None, _NUM_CLASSES))

with tf.variable_scope('RGB'):
    rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
    rgb_net, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
    end_point = 'Logits'
    with tf.variable_scope(end_point):
        rgb_net = tf.nn.avg_pool3d(rgb_net, ksize=[1, 2, 7, 7, 1],
                                   strides=[1, 1, 1, 1, 1], padding=snt.VALID)
        if TRAINING:
            rgb_net = tf.nn.dropout(rgb_net, 0.7)
        logits = i3d.Unit3D(output_channels=_NUM_CLASSES,
                            kernel_shape=[1, 1, 1],
                            activation_fn=None,
                            use_batch_norm=False,
                            use_bias=True,
                            name='Conv3d_0c_1x1')(rgb_net, is_training=True)

        logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
        averaged_logits = tf.reduce_mean(logits, axis=1)

    # predictions = tf.nn.softmax(averaged_logits)

rgb_variable_map = {}

for variable in tf.global_variables():
    if variable.name.split("/")[-4] == "Logits": continue
    if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable

# print(rgb_variable_map)
rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

model_logits = averaged_logits
model_predictions = tf.nn.softmax(model_logits)
