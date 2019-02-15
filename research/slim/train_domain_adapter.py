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
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import pickle
from tensorflow.python import debug as tf_debug

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from input_pipeline import input_pipeline
from dann import revgrad

slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.learning import train_step
from tensorflow.python.training.evaluation import _get_or_create_eval_step

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy. Note For '
                            'historical reasons loss from all clones averaged '
                            'out and learning rate decay happen per clone '
                            'epochs')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'test_every_n_steps', 500,
    'The frequency with which accuracy metrics should be run.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_every_n_steps', 60,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_every_n_steps', 300,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'adaptation_loss_weight', 1.0, 'Relative weight of adaptation loss vs classification loss')

tf.app.flags.DEFINE_float(
    'reconstruction_loss_weight', 1.0, 'Relative weight of reconstruction loss vs adaptation loss')

tf.app.flags.DEFINE_float(
    'classification_loss_weight', 0.0, 'Relative weight of adaptation loss vs classification loss')

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'sgd',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

tf.app.flags.DEFINE_integer(
    'quantize_delay', -1,
    'Number of steps to start quantized training. Set to -1 would disable '
    'quantized training.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.9,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_integer(
    'num_classes', 6, 'Number of classes in the dataset.')

tf.app.flags.DEFINE_integer(
    'num_domain_layers', 2, 'Number of hidden layers for domain classification.')

tf.app.flags.DEFINE_string(
    'source_dataset_files', None, 'The directory where the source dataset files are stored.')

tf.app.flags.DEFINE_string(
    'target_dataset_file', None, 'The directory where the target dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 256, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', 10 ** 5,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  # Note: when num_clones is > 1, this will actually have each clone to go
  # over each epoch FLAGS.num_epochs_per_decay times. This is different
  # behavior from sync replicas and is expected to produce different results.
  decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay /
                    FLAGS.batch_size)

  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized' % FLAGS.optimizer)
  return optimizer


def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  variables_to_restore = {}
  for var in slim.get_model_variables():
    if var.op.name.startswith('resnet_v2_152/domain_adapter') or var.op.name.startswith('resnet_v2_152/domain_reconstructor'):
      continue
    variables_to_restore[var.op.name] = var

  # import ipdb; ipdb.set_trace()
  # checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % (checkpoint_path))

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


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
        class_label = slim.one_hot_encoding(features['label'], FLAGS.num_classes)
        domain_label = slim.one_hot_encoding(dlbl, 2)

        return feature_map, class_label, domain_label
    return _parse_example


def main(_):
  if not FLAGS.source_dataset_files:
    raise ValueError('You must supply the dataset directory with --source_dataset_files')
  if not FLAGS.target_dataset_file:
    raise ValueError('You must supply the dataset directory with --target_dataset_file')
  if not FLAGS.max_number_of_steps:
    raise ValueError('You must supply maximum number of steps')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default() as g:
    # Create global_step
    global_step = slim.create_global_step()
    # Create eval step
    _get_or_create_eval_step()

    ######################
    # Declare datasets #
    ######################
    source_filenames = eval(FLAGS.source_dataset_files)
    source_dataset = tf.data.TFRecordDataset(source_filenames)
    source_dataset = source_dataset.shuffle(buffer_size=5000)
    source_dataset = source_dataset.repeat()
    source_dataset = source_dataset.map(_get_parser('source'))
    source_dataset = source_dataset.batch(FLAGS.batch_size)
    source_iterator = source_dataset.make_one_shot_iterator()
    source_tuple = source_iterator.get_next()

    target_filenames = FLAGS.target_dataset_file
    target_dataset = tf.data.TFRecordDataset(target_filenames)
    target_dataset = target_dataset.shuffle(buffer_size=5000)
    target_dataset = target_dataset.repeat()
    target_dataset = target_dataset.map(_get_parser('target'))
    target_dataset = target_dataset.batch(FLAGS.batch_size)
    target_iterator = target_dataset.make_one_shot_iterator()
    target_tuple = target_iterator.get_next()

    # var for measuring training progress
    p = tf.divide(global_step, tf.cast(FLAGS.max_number_of_steps, tf.int64), name='training_progress')
    # goes from 0 to 1 as training progress to supress noisy domain signal during initial training phase
    l = 2. / (1. + tf.exp(-10. * p)) - 1
    domain_adaptation_weight = tf.cast(3. * l, tf.float32, name='domain_adaptation_weight')

    ####################
    # Define the model #
    ####################
    source_features, source_class_labels, source_domain_labels = source_tuple
    target_features, _, target_domain_labels = target_tuple
    features = tf.concat((source_features, target_features), 0)
    features = tf.ensure_shape(features, [2 * FLAGS.batch_size, 2048])

    source_class_logits, domain_logits = revgrad(features, FLAGS.num_domain_layers, FLAGS.num_classes, domain_adaptation_weight, FLAGS.weight_decay, FLAGS.batch_size)
    source_domain_logits = domain_logits[:FLAGS.batch_size]
    target_domain_logits = domain_logits[FLAGS.batch_size:]

    #############################
    # Specify the loss function #
    #############################
    slim.losses.softmax_cross_entropy(
        source_class_logits, source_class_labels, scope='source_classification_loss')
    slim.losses.softmax_cross_entropy(
        source_domain_logits, source_domain_labels, scope='source_domain_adaptation_loss')
    slim.losses.softmax_cross_entropy(
        target_domain_logits, target_domain_labels, scope='target_domain_adaptation_loss')

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summaries.add(tf.summary.scalar('training_progress', p))
    summaries.add(tf.summary.scalar('domain_adaptation/weight', domain_adaptation_weight))

    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
    update_ops = []

    # Add summaries for end_points.
    # end_points = clones[0].outputs
    # for end_point in end_points:
    #   x = end_points[end_point]
    #   summaries.add(tf.summary.histogram('activations/' + end_point, x))
    #   summaries.add(tf.summary.scalar('sparsity/' + end_point,
    #                                   tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    summaries.add(tf.summary.scalar('Losses/source_classification_loss', 
        tf.get_collection(tf.GraphKeys.LOSSES, 'source_classification_loss')[0]))
    summaries.add(tf.summary.scalar('domain_adaptation/source_loss', 
        tf.get_collection(tf.GraphKeys.LOSSES, 'source_domain_adaptation_loss')[0]))
    summaries.add(tf.summary.scalar('domain_adaptation/target_loss', 
        tf.get_collection(tf.GraphKeys.LOSSES, 'target_domain_adaptation_loss')[0]))
    summaries.add(tf.summary.scalar('Losses/domain_adaptation_loss', 
        tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES, '.*domain_adaptation'))))
    summaries.add(tf.summary.scalar('domain_adaptation/total_loss', 
        tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES, '.*domain_adaptation'))))

    # # Add summaries for variables.
    # for variable in slim.get_model_variables():
    #   summaries.add(tf.summary.histogram(variable.op.name, variable))

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    # learning_rate = _configure_learning_rate(test_dataset.num_samples, global_step)
    learning_rate = tf.divide(tf.cast(1e-4, tf.float64), (1. + 10 * p) ** 0.75, name='learning_rate')
    optimizer = _configure_optimizer(learning_rate)
    summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    if FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    model_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = model_loss + regularization_loss
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('Losses/regularization_loss', regularization_loss))
    summaries.add(tf.summary.scalar('Losses/total_loss', total_loss))

    # Compute gradients
    gradients = optimizer.compute_gradients(total_loss, var_list=variables_to_train)

    # regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # print_ops = []
    # for reg in regs:
    #     summary_name = reg.name
    #     op = tf.summary.scalar(summary_name, reg, collections=[])
    #     op = tf.Print(op, [reg], summary_name)
    #     print_ops.append(op)

    ### PRINT COMPARE
    # resnet_v2_152/block3/unit_6/bottleneck_v2/conv2/kernel/Regularizer/l2_regularizer:0[0.000394384027]
    # resnet_v2_152/block3/unit_6/bottleneck_v2/conv1/kernel/Regularizer/l2_regularizer:0[0.00018388957]
    # resnet_v2_152/block3/unit_5/bottleneck_v2/conv3/kernel/Regularizer/l2_regularizer:0[0.000451341883]
    # resnet_v2_152/logits/kernel/Regularizer/l2_regularizer:0[0.0136216842]

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(gradients, global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    # print_op = tf.group(*print_ops)
    # with tf.control_dependencies([update_op, print_op]):
    with tf.control_dependencies([update_op]):
      train_op = tf.identity(total_loss, name='train_op')

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=g)

    # Define saver - only diff from default is in what ckpts to keep
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)

    sess = tf.Session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    if tf.train.latest_checkpoint(FLAGS.train_dir):
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
    else:
        sess.run(tf.global_variables_initializer())

    # global_step
    gs = sess.run(global_step)

    while gs < FLAGS.max_number_of_steps:
        gs, loss = sess.run([global_step, train_op])

        if gs % FLAGS.log_every_n_steps == 0:
            print('global step: %d: loss = %f' % (gs, loss))

        if gs % FLAGS.save_every_n_steps == 0:
            print('global step: %d: checkpoint' % (gs))
            saver.save(sess, FLAGS.train_dir + '/model.ckpt', global_step=gs)

        if gs % FLAGS.save_summaries_every_n_steps == 0:
            print('global step: %d: summary' % (gs))
            gs, loss, summaries = sess.run([global_step, train_op, summary_op])
            # import ipdb; ipdb.set_trace()
            summary_writer.add_summary(summaries, global_step=gs)
            summary_writer.flush()

    summary_writer.close()


    ###########################
    # Kicks off the training. #
    ###########################
    # slim.learning.train(
    #     train_tensor,
    #     session_wrapper=tf_debug.LocalCLIDebugWrapperSession,
    #     logdir=FLAGS.train_dir,
    #     master=FLAGS.master,
    #     is_chief=(FLAGS.task == 0),
    #     # init_fn=_get_init_fn(),
    #     summary_op=summary_op,
    #     number_of_steps=FLAGS.max_number_of_steps,
    #     log_every_n_steps=FLAGS.log_every_n_steps,
    #     save_summaries_secs=FLAGS.save_summaries_secs,
    #     saver=saver,
    #     save_interval_secs=FLAGS.save_interval_secs,
    #     sync_optimizer=optimizer if FLAGS.sync_replicas else None)


if __name__ == '__main__':
  tf.app.run()
