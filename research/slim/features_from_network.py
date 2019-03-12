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
import os, glob
import pickle
from skimage.io import imread

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

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
    'dataset_dir', None, 'The directory where the dataset files are stored.')

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
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_integer(
    'eval_interval_secs', 10, 'Eval time interval (in s)')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    # dataset = dataset_factory.get_dataset(
    #     FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=None,
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    # print('-------BATCH SIZE: {}--------'.format(FLAGS.batch_size))
    # provider = slim.dataset_data_provider.DatasetDataProvider(
    #     dataset,
    #     shuffle=False,
    #     common_queue_capacity=2 * FLAGS.batch_size,
    #     common_queue_min=FLAGS.batch_size)
    # [image, label] = provider.get(['image', 'label'])
    # label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    # Load data
    data = pickle.load(open("{}/data_list.pkl".format(FLAGS.dataset_dir), 'rb'))
    means = data['means']
    raw_image = tf.placeholder(tf.uint8, shape=[500,500,3])
    image = image_preprocessing_fn(raw_image, eval_image_size, eval_image_size, means=means)
    images = [image]

    ####################
    # Define the model #
    ####################
    logits, end_points = network_fn(images)
    features = end_points['global_pool']
    logits = tf.squeeze(logits)
    features = tf.squeeze(features)  # shape should be (2048,) now

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

    # predictions = tf.argmax(logits, 1)
    # labels = tf.squeeze(labels)

    # Define the metrics:
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    #     'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
    #     'Recall_2': slim.metrics.streaming_recall_at_k(logits, labels, 2),
    #     'Recall_3': slim.metrics.streaming_recall_at_k(logits, labels, 3),
    # })

    # Print the summaries to screen.
    # for name, value in names_to_values.items():
    #   summary_name = 'eval/%s' % name
    #   op = tf.summary.scalar(summary_name, value, collections=[])
    #   op = tf.Print(op, [value], summary_name)
    #   tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    # if FLAGS.max_num_batches:
    #   num_batches = FLAGS.max_num_batches
    # else:
    #   # This ensures that we make a single pass over all of the data.
    #   num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

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

    # if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    #   ## LOOP
    #   # _eval_loop(FLAGS.checkpoint_path)

    #   ## LATEST CKPT
    # checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    #   _eval(checkpoint_path)
    saver = tf.train.Saver()

    #   ## EVAL ALL CKPTS
    #   # checkpoint_paths = sorted(glob.glob('{}/model.ckpt-*data*'.format(FLAGS.checkpoint_path)), key=lambda s: int(s.split('-')[1].split('.')[0]))
    #   # for checkpoint_path in checkpoint_paths:
    #   #   _eval('.'.join(checkpoint_path.split('.')[:2]))
    # else:
    #   _eval(FLAGS.checkpoint_path)

    # cats = data['cats']
    # ff_idx = cats.index("frozenfood")
    # cats.pop(ff_idx)
    cats = ["frozenfood"]
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.checkpoint_path)
        for cat in cats:
            print("------------------- {} -------------------".format(cat.upper()))
            imgfiles = glob.glob("{}/{}/tmpl*jpg".format(FLAGS.dataset_dir, cat))
            num_imgfiles = len(imgfiles)
            for i, imgfile in enumerate(imgfiles):
                I = imread(imgfile)
                lgts, ftrs = sess.run([logits, features], feed_dict={ raw_image: I })
                feature_dump = { 'logits': lgts, 'features': ftrs }
                pickle.dump(feature_dump, 
                        open("{}/{}/stpaul_resnet_features_{}.pkl".format(FLAGS.dataset_dir, cat, imgfile.split('/')[-1][6:-4]), 
                            "wb"))
                if i % 100 == 0:
                    print("{}/{}".format(i,num_imgfiles))


if __name__ == '__main__':
  tf.app.run()
