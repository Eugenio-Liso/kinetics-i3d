#!python
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import numpy as np
import tensorflow as tf
import i3d
from logging_utils import logger_factory as lf

logger = lf.getBasicLogger(os.path.basename(__file__))
_IMAGE_SIZE = 224
# _SAMPLE_PATHS = {
#     'rgb': 'data/CricketShot.npy',
#     'flow': 'data/CricketShot.npy',
# }
_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}
_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('eval_type', '', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
tf.flags.DEFINE_string('input_folder_rgb', '', 'Input folder of videos, as .npy, with RGB format')
tf.flags.DEFINE_string('input_folder_flow', '', 'Input folder of videos, as .npy, with Flow format')


# np.set_printoptions(threshold=np.inf)
def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = FLAGS.eval_type
    imagenet_pretrained = FLAGS.imagenet_pretrained
    input_rgb_video_folder = FLAGS.input_folder_rgb
    input_flow_video_folder = FLAGS.input_folder_flow
    executions_times_with_video_name = []

    if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

    logger.info(f'Chosen evaluation type: {eval_type}')

    numOfBatchFrames = 16  # Can be changed

    if eval_type == 'rgb600':
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
    else:
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    flow_input, flow_saver, rgb_input, rgb_saver, model_logits, model_predictions = nn_construction(
        eval_type, numOfBatchFrames)

    if input_rgb_video_folder:
        for filename_rgb in os.listdir(input_rgb_video_folder):
            video_name = filename_rgb.split(".")[0]
            complete_filename_rgb = os.path.join(input_rgb_video_folder, filename_rgb)
            if input_flow_video_folder:
                foundFile = False
                complete_filename_flow = ""
                for filename_flow in os.listdir(input_flow_video_folder):
                    if filename_flow.startswith(video_name):
                        foundFile = True
                        complete_filename_flow = os.path.join(input_flow_video_folder, filename_flow)
                        break
                    else:
                        continue
                if not foundFile:
                    raise ValueError("A file with the same name must exists in the two folders")
                else:
                    execution_time_with_video_name = prediction_phase(eval_type,
                                                                      imagenet_pretrained,
                                                                      complete_filename_rgb,
                                                                      complete_filename_flow,
                                                                      video_name,
                                                                      flow_input,
                                                                      flow_saver,
                                                                      rgb_input,
                                                                      rgb_saver,
                                                                      model_logits,
                                                                      model_predictions,
                                                                      numOfBatchFrames,
                                                                      kinetics_classes)
            else:
                execution_time_with_video_name = prediction_phase(eval_type,
                                                                  imagenet_pretrained,
                                                                  complete_filename_rgb,
                                                                  "",
                                                                  video_name,
                                                                  flow_input,
                                                                  flow_saver,
                                                                  rgb_input,
                                                                  rgb_saver,
                                                                  model_logits,
                                                                  model_predictions,
                                                                  numOfBatchFrames,
                                                                  kinetics_classes)
            executions_times_with_video_name.append(execution_time_with_video_name)
    elif input_flow_video_folder:
        for filename in os.listdir(input_flow_video_folder):
            video_name = filename.split(".")[0]
            complete_filename_flow = os.path.join(input_flow_video_folder, filename)
            execution_time_with_video_name = prediction_phase(eval_type,
                                                              imagenet_pretrained,
                                                              "",
                                                              complete_filename_flow,
                                                              video_name,
                                                              flow_input,
                                                              flow_saver,
                                                              rgb_input,
                                                              rgb_saver,
                                                              model_logits,
                                                              model_predictions,
                                                              numOfBatchFrames,
                                                              kinetics_classes)

            executions_times_with_video_name.append(execution_time_with_video_name)
    else:
        raise ValueError("Must specify one folder between RGB and flow at least")
    print(executions_times_with_video_name)
    # input_video_rgb = FLAGS.input_video_rgb
    # input_video_flow = FLAGS.input_video_flow


def prediction_phase(eval_type,
                     imagenet_pretrained,
                     input_video_rgb,
                     input_video_flow,
                     video_name,
                     flow_input,
                     flow_saver,
                     rgb_input,
                     rgb_saver,
                     model_logits,
                     model_predictions,
                     numOfBatchFrames,
                     kinetics_classes):
    logger.info(f"Model inputs: RGB -> {input_video_rgb} | Flow -> {input_video_flow}")

    rgb_sample = None
    flow_sample = None
    input_video_frames_rgb = None
    input_video_frames_flow = None

    # .shape produces a tuple like (1, 150, 224, 224, 3)
    # First is batch size
    # Num of frames of the input video
    # Width input video
    # Height input video
    # Channels input video

    if input_video_rgb and input_video_rgb.strip():  # Check string not empty
        rgb_sample = np.load(input_video_rgb)
        # logger.info(rgb_sample)
        input_video_frames_rgb = rgb_sample.shape[1]

    if input_video_flow and input_video_flow.strip():  # Check string not empty
        flow_sample = np.load(input_video_flow)
        # logger.info(flow_sample)
        input_video_frames_flow = flow_sample.shape[1]

    if eval_type == 'joint':  # Consistency check
        input_rgb_frames = rgb_sample.shape
        input_flow_frames = flow_sample.shape
        if input_rgb_frames[1] != input_flow_frames[1]:
            raise ValueError(
                f"The frames of the input videos are not equal: RGB -> {input_rgb_frames} BUT FLOW -> {input_flow_frames}. Are they from the same video?")
        else:
            input_video_frames_rgb = input_rgb_frames[1]
            input_video_frames_flow = input_flow_frames[1]

    logger.info(f"Input frames: RGB -> {input_video_frames_rgb} | Flow -> {input_video_frames_flow}")

    if input_video_frames_rgb is not None:
        totalNumOfFrames = input_video_frames_rgb - 1
    elif input_video_frames_flow is not None:
        totalNumOfFrames = input_video_frames_flow - 1
    else:
        raise ValueError("The number of frames is undefined")

    execution_times_with_video_name = []
    maximumFrames = False
    exec_times_with_segments = []
    sliceIndex = 0

    while sliceIndex < totalNumOfFrames:
        nextSliceIndex = sliceIndex + numOfBatchFrames
        print(f"Current sliceIndex: {sliceIndex}-{nextSliceIndex}")

        if (nextSliceIndex - 1) > totalNumOfFrames:
            print("Reached max num of frames")
            exceedFrames = (nextSliceIndex - 1) - totalNumOfFrames
            print(f"Exceeding frames: {exceedFrames}")
            maximumFrames = True

        with tf.Session() as sess:
            # print(rgb_sample[:, 0, :, :, :])
            # for idx in range(rgb_sample.shape[1]):
            #     print(rgb_sample[:, idx, :, :, :].shape)
            #     break
            feed_dict = {}
            if eval_type in ['rgb', 'rgb600', 'joint']:
                slicedRGBInput = rgb_sample[:, sliceIndex:nextSliceIndex, :, :, :]
                if maximumFrames:
                    # Takes the last N frames to compensate
                    startSliceIdx = ((numOfBatchFrames - 1) - exceedFrames)
                    endSliceIdx = numOfBatchFrames
                    slicedRGBInput = np.concatenate((slicedRGBInput,
                                                     slicedRGBInput[:, startSliceIdx:endSliceIdx, :, :, :]),
                                                    axis=1)

                # print(f"SLICED RGB INPUT: {slicedRGBInput.shape[1]}")
                if imagenet_pretrained:
                    rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
                else:
                    rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
                tf.logging.info('RGB checkpoint restored')
                tf.logging.info('RGB data loaded, shape=%s', str(slicedRGBInput.shape))
                feed_dict[rgb_input] = slicedRGBInput

            if eval_type in ['flow', 'joint']:
                slicedFlowInput = flow_sample[:, sliceIndex:nextSliceIndex, :, :, :]
                if maximumFrames:
                    # Takes the last N frames to compensate
                    startSliceIdx = ((numOfBatchFrames - 1) - exceedFrames)
                    endSliceIdx = numOfBatchFrames
                    slicedFlowInput = np.concatenate((slicedFlowInput,
                                                      slicedFlowInput[:, startSliceIdx:endSliceIdx, :, :, :]),
                                                     axis=1)
                if imagenet_pretrained:
                    flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
                else:
                    flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
                tf.logging.info('Flow checkpoint restored')
                tf.logging.info('Flow data loaded, shape=%s', str(slicedFlowInput.shape))
                feed_dict[flow_input] = slicedFlowInput
            start_time = time.time()
            out_logits, out_predictions = sess.run(
                [model_logits, model_predictions],
                feed_dict=feed_dict)
            end_time = time.time()
            exec_time = end_time - start_time

            logger.info("--- Execution time: %s seconds ---" % exec_time)
            segment = [(sliceIndex + 1), nextSliceIndex]
            exec_time = exec_time
            exec_times_with_segments.append((segment, exec_time))

            out_logits = out_logits[0]
            out_predictions = out_predictions[0]
            sorted_indices = np.argsort(out_predictions)[::-1]

            logger.info('Norm of logits: %f' % np.linalg.norm(out_logits))
            logger.info('Top classes and probabilities')
            for index in sorted_indices[:20]:
                logger.info("Probability: {}, logit: {}, kinetics_class predicted: {}".format(out_predictions[index],
                                                                                              out_logits[index],
                                                                                              kinetics_classes[index]))

        sliceIndex = nextSliceIndex
    execution_time_with_video_name = {
        video_name: exec_times_with_segments
    }
    return execution_time_with_video_name


def nn_construction(eval_type, numOfBatchFrames):
    NUM_CLASSES = 400
    if eval_type == 'rgb600':
        NUM_CLASSES = 600

    flow_input = None
    flow_saver = None
    rgb_input = None
    rgb_saver = None

    if eval_type in ['rgb', 'rgb600', 'joint']:
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(
            tf.float32,
            shape=(1, numOfBatchFrames, _IMAGE_SIZE, _IMAGE_SIZE, 3))

        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(
                rgb_input, is_training=False, dropout_keep_prob=1.0)

        rgb_variable_map = {}
        for variable in tf.global_variables():

            if variable.name.split('/')[0] == 'RGB':
                if eval_type == 'rgb600':
                    rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
                else:
                    rgb_variable_map[variable.name.replace(':0', '')] = variable

        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(
            tf.float32,
            shape=(1, numOfBatchFrames, _IMAGE_SIZE, _IMAGE_SIZE, 2))
        with tf.variable_scope('Flow'):
            flow_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, _ = flow_model(
                flow_input, is_training=False, dropout_keep_prob=1.0)
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
    if eval_type == 'rgb' or eval_type == 'rgb600':
        model_logits = rgb_logits
    elif eval_type == 'flow':
        model_logits = flow_logits
    else:
        model_logits = rgb_logits + flow_logits
    model_predictions = tf.nn.softmax(model_logits)

    return flow_input, flow_saver, rgb_input, rgb_saver, model_logits, model_predictions


if __name__ == '__main__':
    tf.app.run(main)
