# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import glob
import pickle

from datasets import dataset_factory
from datasets.sections import TRAIN_SPLITS_TO_SIZES, TEST_SPLITS_TO_SIZES
from nets import nets_factory
from preprocessing import preprocessing_factory
from dann import revgrad

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'test_dataset_file', None, 'test dataset tfrecord')

tf.app.flags.DEFINE_integer(
    'num_domain_layers', 2, 'Number of hidden layers for domain classification.')

tf.app.flags.DEFINE_integer(
    'num_classes', 6, 'Number of classes in the dataset.')

tf.app.flags.DEFINE_string(
    'domain', 'target', 'source/target')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.9,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_integer(
    'eval_interval_secs', 300, 'Eval time interval (in s)')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

FLAGS = tf.app.flags.FLAGS


def _get_parser(domain):
    if domain not in ['source', 'target']:
        raise ValueError('Domain must be source/target')
    dlbl = 0 if domain == 'source' else 1

    def _parse_example(serialized_record):
        features = tf.parse_single_example(serialized_record,
            features={
                'feature_map': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            }, name='features')

        feature_map = tf.decode_raw(features['feature_map'], tf.float32)
        class_label = features['label']
        domain_label = dlbl

        return feature_map, class_label, domain_label
    return _parse_example


def main(_):
  if not FLAGS.test_dataset_file:
    raise ValueError('You must supply the dataset directory with --test_dataset_file')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Declare datasets #
    ######################
    filenames = eval(FLAGS.test_dataset_file)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_get_parser(FLAGS.domain))
    dataset = dataset.batch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, class_labels, domain_labels = iterator.get_next()

    ####################
    # Define the model #
    ####################
    features.set_shape([None, 2048])
    class_logits, domain_logits = revgrad(features, FLAGS.num_domain_layers, FLAGS.num_classes, 1., 1., FLAGS.batch_size, train=False)

    if FLAGS.quantize:
      tf.contrib.quantize.create_eval_graph()

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    class_predictions = tf.argmax(class_logits, 1)
    domain_predictions = tf.argmax(domain_logits, 1)
    # class_labels = tf.squeeze(class_labels)
    # domain_labels = tf.squeeze(domain_labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'AdaptationAccuracy': slim.metrics.streaming_accuracy(domain_predictions, domain_labels),
        'ClassificationAccuracy': slim.metrics.streaming_accuracy(class_predictions, class_labels),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.

      ## META
      # Dataset    - num_train/num_test
      # Shoreview  - 6090 / 1523
      # Minnetonka - 6818 / 1705
      # StLouis    - 6782 / 1696
      # Shorevw + Mtonka  - 12908 / 3228
      num_batches = math.ceil(6782 / float(FLAGS.batch_size))

    def _eval(checkpoint_path):
        tf.logging.info('Evaluating %s' % checkpoint_path)
        slim.evaluation.evaluate_once(
            master=FLAGS.master,
            checkpoint_path=checkpoint_path,
            logdir=FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            variables_to_restore=variables_to_restore)

    def _eval_loop(checkpoint_dir):
        slim.evaluation.evaluation_loop(
            master=FLAGS.master,
            checkpoint_dir=checkpoint_dir,
            logdir=FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            variables_to_restore=variables_to_restore,
            eval_interval_secs=FLAGS.eval_interval_secs)

    _eval(tf.train.latest_checkpoint(FLAGS.checkpoint_path))

    # if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    #   ## LOOP
    #   _eval_loop(FLAGS.checkpoint_path)

    #   ## LATEST CKPT
    #   # checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    #   # _eval(checkpoint_path)

    #   ## EVAL ALL CKPTS
    #   # checkpoint_paths = sorted(glob.glob('{}/model.ckpt-*data*'.format(FLAGS.checkpoint_path)), key=lambda s: int(s.split('-')[1].split('.')[0]))
    #   # for checkpoint_path in checkpoint_paths[-1:]:
    #   #   _eval('.'.join(checkpoint_path.split('.')[:3]))
    # else:
    #   _eval(FLAGS.checkpoint_path)


if __name__ == '__main__':
  tf.app.run()
