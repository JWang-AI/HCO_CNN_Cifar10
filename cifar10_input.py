# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def distorted_inputs(file_name, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

    Args:
      file_name: Path to the CIFAR-10 train_data.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    '''filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
            '''

    # Create a queue that produces the filenames to read.
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    def parser(record):
        features = tf.parse_single_example(
            record,
            features={
                "label": tf.FixedLenFeature([], tf.int64),
                "image": tf.FixedLenFeature([], tf.string)
            }
        )
        image = tf.decode_raw(features['image'], tf.uint8)
        reshaped_image = tf.reshape(image, [32, 32, 3])
        reshaped_image = tf.cast(reshaped_image, tf.float32)
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)
        float_image = tf.image.per_image_standardization(distorted_image)
        label = tf.cast(features['label'], tf.int32)
        float_image.set_shape([height, width, 3])
        return float_image, label

    with tf.name_scope('data_augmentation'):
        dataset = tf.data.TFRecordDataset(file_name)
        dataset = dataset.map(parser, num_parallel_calls=8).repeat().batch(batch_size).shuffle(
            buffer_size=5 * batch_size)
        iterator = dataset.make_one_shot_iterator()
        img_input, labels = iterator.get_next()
        img_input.set_shape([batch_size, height, width, 3])
    # Generate a batch of images and labels by building up a queue of examples.
    return img_input, tf.reshape(labels, [batch_size])


def inputs(eval_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    filename = ""
    if eval_data:
        filename = os.path.join(data_dir, 'eval.tfrecords')
    else:
        pass

    def parser(record):
        features = tf.parse_single_example(
            record,
            features={
                "label": tf.FixedLenFeature([], tf.int64),
                "image": tf.FixedLenFeature([], tf.string)
            }
        )
        image = tf.decode_raw(features['image'], tf.uint8)
        reshaped_image = tf.reshape(image, [32, 32, 3])
        reshaped_image = tf.cast(reshaped_image, tf.float32)
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                               height, width)
        float_image = tf.image.per_image_standardization(resized_image)
        label = tf.cast(features['label'], tf.int32)
        return float_image, label

    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(parser, num_parallel_calls=8).repeat().batch(batch_size).shuffle(buffer_size=5*batch_size)
        iterator = dataset.make_one_shot_iterator()
        img_input, labels = iterator.get_next()
        # Generate a batch of images and labels by building up a queue of examples.
        img_input.set_shape([batch_size, height, width, 3])
    return img_input, tf.reshape(labels, [batch_size])
