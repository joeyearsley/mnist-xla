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
"""Simple MNIST classifier example with JIT XLA and timelines.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import tensorflow as tf
import tensorflow.contrib.nccl as nccl

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline

from model import get_model
from dataset import get_iterators

FLAGS = None
GPUS = 1

def get_post_init_ops():
  # Copy initialized values for variables on GPU 0 to other GPUs.
  global_vars = tf.global_variables()
  var_by_name = dict([(v.name, v) for v in global_vars])
  post_init_ops = []
  for v in global_vars:
    split_name = v.name.split('/')
    # TODO(b/62630508): use more specific prefix than v or v0.
    if split_name[0] == 'v0' or not v.name.startswith('v'):
      continue
    split_name[0] = 'v0'
    copy_from = var_by_name['/'.join(split_name)]
    post_init_ops.append(v.assign(copy_from.read_value()))
  return post_init_ops


def main(_):
  training = tf.Variable(True)

  accuracies = []
  training_steps = []
  optimisers = []
  device_grads = []
  losses = []

  for device_num in range(GPUS):
    with tf.variable_scope('v{}'.format(device_num)):
      with tf.device('/cpu:0'):
        train_path = os.path.join(FLAGS.data_dir, 'train')
        test_path  = os.path.join(FLAGS.data_dir, 'test')
        x, y_ = get_iterators(train_path, test_path)

      with tf.device('/gpu:{}'.format(device_num)):
        y = get_model(x, training=training)

        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
        losses.append(cross_entropy)

        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), dtype=tf.int32), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracies.append(accuracy)

        params = [v for v in tf.get_collection('trainable_variables')
                  if v.name.startswith('v%s/' % device_num)]

        opt = tf.train.GradientDescentOptimizer(0.01)
        optimisers.append(opt)

        grads = opt.compute_gradients(cross_entropy, params)

        device_grads.append(grads)


  new_device_grads = []
  for grad_and_vars in zip(*device_grads):
    scaled_grads = [g for g, _ in grad_and_vars]
    summed_grads = nccl.all_sum(scaled_grads)

    aggregated_device_grads = []
    for (_, v), g in zip(grad_and_vars, summed_grads):
      aggregated_device_grads.append([g, v])

    new_device_grads.append(aggregated_device_grads)

  aggregated_device_grads = [list(x) for x in zip(*new_device_grads)]

  training_ops = []
  for d, device in enumerate(['/gpu:{}'.format(x) for x in range(GPUS)]):
    with tf.device(device):
      opt = optimisers[d]
      avg_grads = aggregated_device_grads[d]
      training_ops.append(optimisers[d].apply_gradients(avg_grads))


  config = tf.ConfigProto()
  jit_level = 0
  if FLAGS.xla:
    # Turns on XLA JIT compilation.
    jit_level = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level
  run_metadata = tf.RunMetadata()
  sess = tf.Session(config=config)
  
  sess.run(tf.global_variables_initializer())


  local_var_init_op = tf.local_variables_initializer()
  variable_mgr_init_ops = [local_var_init_op]
  with tf.control_dependencies([local_var_init_op]):
    variable_mgr_init_ops.extend(get_post_init_ops())
  local_var_init_op_group = tf.group(*variable_mgr_init_ops)
  sess.run(local_var_init_op_group)


  # Get handles to enable iterator feeding.
  sess.run([tf.get_collection('trn_iterator_inits'), tf.get_collection('val_iterator_inits')])
  training_handles = sess.run(tf.get_collection('trn_iterator_handles'))
  test_handles = sess.run(tf.get_collection('test_iterator_handles'))
  feedable_handles = tf.get_collection('feedable_iterator_handles')
  training_feed_dict = dict(zip(feedable_handles, training_handles))
  test_feed_dict = dict(zip(feedable_handles, test_handles))

  # Train
  train_step = tf.group(training_ops)
  loss = tf.reduce_mean(losses)

  loss_window = 200
  loss_agg = np.zeros(loss_window)
  for i in range(FLAGS.train_loops):
    # Create a timeline for the last loop and export to json to view with
    # chrome://tracing/.
    if i == train_loops - 1:
      sess.run([loss, train_step],
               feed_dict=training_feed_dict,
               options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
               run_metadata=run_metadata)
      trace = timeline.Timeline(step_stats=run_metadata.step_stats)
      with open('timeline.ctf.json', 'w') as trace_file:
        trace_file.write(trace.generate_chrome_trace_format())
    else:
      l, _ = sess.run([loss, train_step], feed_dict=training_feed_dict)
      loss_agg[i % loss_window] = l

      print('Loss: {} /r'.format(np.mean(loss_agg)))
  # Test trained model
  # Change dataset to test version

  # Assign training = false
  sess.run([tf.get_collection('test_iterator_inits'), training.assign(False)])
  print(sess.run(accuracy, feed_dict=test_feed_dict))
  sess.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  parser.add_argument(
      '--xla', type=bool, default=False, help='Turn xla via JIT on')
  parser.add_argument(
      '--train_loops', type=int, default=1000, help='How many training steps to do')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
