from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import scipy.io

# Process images of this size. Note that this differs from the original SVHN
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the SVHN data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 73257
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 26032


def read_svhn(data_dir, train=True):
  """Load SVHN images and labels.

  Args:
    data_dir:  Directory where data resides.
    train:     Boolean flag for data file to read: train/test.

  Returns:
    images: Images. nd-array in NHWC format.
    labels: Labels. nd-array containing labels of [batch_size x 1] size.
  """
  fname = 'train_32x32.mat' if train else 'test_32x32.mat'
  filename = os.path.join(data_dir, fname)
  if not tf.gfile.Exists(filename):
    raise ValueError('Failed to find file: ' + filename)

  data = scipy.io.loadmat(filename)
  images = data['X']
  # convert from HWCN -> NHWC
  images = np.transpose(images, (3,0,1,2))

  labels = data['y']
  labels[labels == 10] = 0

  return images, labels

def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for SVHN training using the Dataset API.

  Args:
    data_dir:   Directory where data resides.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  images, labels = read_svhn(data_dir, train=True)

  dataset = tf.data.Dataset.from_tensor_slices((images, labels))

  def distort_image(image, label):
    reshaped_image = tf.cast(image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    label.set_shape([1])

    return float_image, label
  dataset = dataset.map(distort_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  dataset = dataset.shuffle(min_queue_examples)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat()
  iterator = dataset.make_one_shot_iterator()

  image_batch, label_batch = iterator.get_next()
  return image_batch, label_batch


def inputs(data_dir, batch_size):
  """Construct distorted input for SVHN training using the Dataset API.

  Args:
    data_dir:   Directory where data resides.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  images, labels = read_svhn(data_dir, train=True)

  dataset = tf.data.Dataset.from_tensor_slices((images, labels))

  def process_image(image, label):
    reshaped_image = tf.cast(image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    label.set_shape([1])

    return float_image, label
  dataset = dataset.map(process_image)

  # Ensure that the random shuffling has good mixing properties.
  # min_fraction_of_examples_in_queue = 0.4
  # min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
  #                          min_fraction_of_examples_in_queue)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()

  image_batch, label_batch = iterator.get_next()
  return image_batch, label_batch
