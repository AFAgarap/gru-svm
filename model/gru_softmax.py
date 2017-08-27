# Copyright 2017 Abien Fred Agarap. All Rights Reserved.
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

"""Implementation of GRU+Softmax model for Intrusion Detection"""

__version__ = '0.2.2'
__author__ = 'Abien Fred Agarap'

import argparse
import data
import numpy as np
import os
import tensorflow as tf
import time

# hyper-parameters
BATCH_SIZE = 256
CELL_SIZE = 256
DROPOUT_P_KEEP = 0.8
HM_EPOCHS = 2
LEARNING_RATE = 1e-6
N_CLASSES = 2
SEQUENCE_LENGTH = 21


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def train_model(train_data, test_data, checkpoint_path, log_path, model_name):
    with tf.name_scope('input'):
        # [BATCH_SIZE, SEQUENCE_LENGTH, 10]
        x_input = tf.placeholder(dtype=tf.float32, shape=[None, SEQUENCE_LENGTH, 10], name='x_input')

        # [BATCH_SIZE, N_CLASSES]
        y_input = tf.placeholder(dtype=tf.float32, shape=[None, N_CLASSES], name='y_input')

    # [BATCH_SIZE, CELL_SIZE]
    state = tf.placeholder(dtype=tf.float32, shape=[None, CELL_SIZE], name='initial_state')

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    p_keep = tf.placeholder(tf.float32, name='p_keep')

    cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
    drop_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=p_keep)

    # outputs: [BATCH_SIZE, SEQUENCE_LENGTH, CELL_SIZE]
    # states: [BATCH_SIZE, CELL_SIZE]
    outputs, states = tf.nn.dynamic_rnn(drop_cell, x_input, initial_state=state, dtype=tf.float32)

    states = tf.identity(states, name='H')

    with tf.name_scope('final_training_ops'):
        with tf.name_scope('weights'):
            weight = tf.get_variable('weights', initializer=tf.random_normal([CELL_SIZE, N_CLASSES], stddev=0.01))
            variable_summaries(weight)
        with tf.name_scope('biases'):
            bias = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[N_CLASSES]))
            variable_summaries(bias)
        hf = tf.transpose(outputs, [1, 0, 2])
        last = tf.gather(hf, int(hf.get_shape()[0]) - 1)
        with tf.name_scope('Wx_plus_b'):
            output = tf.matmul(last, weight) + bias
            tf.summary.histogram('pre-activations', output)

    # Softmax
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_input))
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.name_scope('accuracy'):
        predicted_class = tf.nn.softmax(output)
        with tf.name_scope('correct_prediction'):
            correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y_input, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar('accuracy', accuracy)

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    saver = tf.train.Saver(max_to_keep=1000)

    current_state = np.zeros([BATCH_SIZE, CELL_SIZE])  # initialize H (current_state) with values of zeros

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())  # variable initializer

    merged = tf.summary.merge_all()  # merge all the summaries collected from TF graph

    timestamp = str(time.asctime())  # get the time in seconds since the Epoch

    # create an event file to contain the TF graph summaries for training
    train_writer = tf.summary.FileWriter(log_path + timestamp + '-training', graph=tf.get_default_graph())

    # create an event file to contain the TF graph summaries for validation
    validation_writer = tf.summary.FileWriter(log_path + timestamp + '-validation', graph=tf.get_default_graph())

    with tf.Session() as sess:

        sess.run(init_op)

        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

        # check if a trained model exists
        if checkpoint and checkpoint.model_checkpoint_path:
            # load the graph of the trained model
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            # restore variables to resume training
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                train_example_batch, train_label_batch = sess.run([train_data[0], train_data[1]])

                # changed the range of labels for Softmax to {0, 1}
                train_label_batch[train_label_batch == -1] = 0

                # dictionary for key-value pair input for training
                feed_dict = {x_input: train_example_batch, y_input: train_label_batch, state: current_state,
                             learning_rate: LEARNING_RATE, p_keep: DROPOUT_P_KEEP}

                summary, _, epoch_loss, next_state = sess.run([merged, optimizer, loss, states],
                                                              feed_dict=feed_dict)

                # Display training accuracy every 100 steps and at step 0
                if step % 100 == 0:
                    accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
                    print('step [{}] train -- loss : {}, accuracy : {}'.format(step, epoch_loss, accuracy_))
                    train_writer.add_summary(summary, step)
                    saver.save(sess, checkpoint_path + model_name, global_step=step)

                # Validate training every 100 steps
                if step % 100 == 0 and step > 0:
                    test_example_batch, test_label_batch = sess.run([test_data[0], test_data[1]])

                    # change the range of labels for Softmax to {0, 1}
                    test_label_batch[test_label_batch == -1] = 0

                    # dictionary for key-value pair input for validation
                    feed_dict = {x_input: test_example_batch, y_input: test_label_batch,
                                 state: np.zeros([BATCH_SIZE, CELL_SIZE]), p_keep: 1.0}

                    summary, test_loss, test_accuracy = sess.run([merged, loss, accuracy], feed_dict=feed_dict)

                    print('step [{}] validation -- loss : {}, accuracy : {}'.format(step, test_loss, test_accuracy))

                    validation_writer.add_summary(summary, step)

                current_state = next_state
                step += 1
        except tf.errors.OutOfRangeError:
            print('EOF -- training done at step {}'.format(step))
        except KeyboardInterrupt:
            print('Training interrupted at {}'.format(step))
        finally:
            train_writer.close()
            validation_writer.close()
            coord.request_stop()

        coord.join(threads)

        saver = tf.train.Saver()
        saver.save(sess, checkpoint_path + model_name, global_step=step)


def parse_args():
    parser = argparse.ArgumentParser(description='GRU+Softmax for Intrusion Detection')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-t', '--train_dataset', required=True, type=str,
                       help='path of the training dataset to be used')
    group.add_argument('-v', '--validation_dataset', required=True, type=str,
                       help='path of the validation dataset to be used')
    group.add_argument('-c', '--checkpoint_path', required=True, type=str,
                       help='path where to save the trained model')
    group.add_argument('-l', '--log_path', required=True, type=str,
                       help='path where to save the TensorBoard logs')
    group.add_argument('-m', '--model_name', required=True, type=str,
                       help='filename for the trained model')
    arguments = parser.parse_args()
    return arguments


def main(arguments):
    train_data = data.input_pipeline(path=arguments.train_dataset, batch_size=BATCH_SIZE,
                                     num_classes=N_CLASSES, num_epochs=HM_EPOCHS)

    test_data = data.input_pipeline(path=arguments.validation_dataset, batch_size=BATCH_SIZE,
                                    num_classes=N_CLASSES, num_epochs=1)

    train_model(train_data=train_data, test_data=test_data, checkpoint_path=arguments.checkpoint_path,
                log_path=arguments.log_path, model_name=arguments.model_name)


if __name__ == '__main__':
    args = parse_args()

    main(args)
