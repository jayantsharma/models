"""Evaluation for SVHN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import svhn
import svhn_input

parser = svhn.parser

parser.add_argument('--eval_dir', type=str, default='/tmp/svhn_eval',
                    help='Directory where to write event logs.')

parser.add_argument('--eval_data', type=str, default='test',
                    help='Either `test` or `train_eval`.')

parser.add_argument('--checkpoint_dir', type=str, default='/tmp/svhn_train',
                    help='Directory where to read model checkpoints.')

parser.add_argument('--eval_interval_secs', type=int, default=60*5,
                    help='How often to run the eval.')

parser.add_argument('--num_examples', type=int, default=26032,
                    help='Number of examples to run.')

parser.add_argument('--run_once', type=bool, default=True,
                    help='Whether to run eval only once.')


def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/svhn_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = num_iter * FLAGS.batch_size
    step = 0
    while step < num_iter:
      predictions = sess.run([top_k_op])
      true_count += np.sum(predictions)
      step += 1

    # Compute precision @ 1.
    precision = true_count / total_sample_count
    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

    summary = tf.Summary()
    summary.ParseFromString(sess.run(summary_op))
    summary.value.add(tag='Precision @ 1', simple_value=precision)
    summary_writer.add_summary(summary, global_step)

def evaluate():
  """Eval SVHN for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for SVHN.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = svhn_input.inputs(FLAGS.data_dir, FLAGS.batch_size)
    labels = tf.reshape(labels, [-1])
    labels = tf.cast(labels, tf.int64)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = svhn.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        svhn.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  tf.app.run(main=main)
