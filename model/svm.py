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

"""Implementation of SVM for Intrusion Detection"""

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import argparse
import data
import os
import tensorflow as tf
import time

# Hyper-parameters
BATCH_SIZE = 256
HM_EPOCHS = 1
LEARNING_RATE = 0.01
N_CLASSES = 2
SEQUENCE_LENGTH = 21
SVM_C = 1


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
    """Implementation of the SVM model"""

    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

    with tf.name_scope('input'):
        # [BATCH_SIZE, SEQUENCE_LENGTH]
        x_input = tf.placeholder(dtype=tf.float32, shape=[None, SEQUENCE_LENGTH], name='x_input')

        # [BATCH_SIZE, N_CLASSES]
        y_input = tf.placeholder(dtype=tf.float32, shape=[None, N_CLASSES], name='y_input')

    with tf.name_scope('training_ops'):
        with tf.name_scope('weights'):
            xav_init = tf.contrib.layers.xavier_initializer
            weight = tf.get_variable(name='weights', shape=[SEQUENCE_LENGTH, N_CLASSES], initializer=xav_init())
            variable_summaries(weight)
        with tf.name_scope('biases'):
            bias = tf.get_variable(name='biases', initializer=tf.constant(0.1, shape=[N_CLASSES]))
            variable_summaries(bias)
        with tf.name_scope('Wx_plus_b'):
            y_hat = tf.matmul(x_input, weight) + bias
            tf.summary.histogram('pre-activations', y_hat)

    # L2-SVM
    with tf.name_scope('svm'):
        regularization = 0.5 * tf.reduce_sum(tf.square(weight))
        hinge_loss = tf.reduce_sum(tf.square(tf.maximum(tf.zeros([BATCH_SIZE, N_CLASSES]), 1 - y_input * y_hat)))
        with tf.name_scope('loss'):
            loss = regularization + SVM_C * hinge_loss
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.name_scope('accuracy'):
        predicted_class = tf.sign(y_hat)
        predicted_class = tf.identity(predicted_class, name='prediction')
        with tf.name_scope('correct_prediction'):
            correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y_input, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    tf.summary.scalar('accuracy', accuracy)

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    saver = tf.train.Saver(max_to_keep=1000)

    # variable initializer
    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

    # merge all the summaries in the inference graph
    merged = tf.summary.merge_all()

    # get the time tuple, and parse to str
    timestamp = str(time.asctime())

    # event file to contain TF graph summaries during training
    train_writer = tf.summary.FileWriter(log_path + timestamp + '-training', graph=tf.get_default_graph())

    # event file to contain TF graph summaries during validation
    validation_writer = tf.summary.FileWriter(log_path + timestamp + '-validation', graph=tf.get_default_graph())

    with tf.Session() as sess:

        sess.run(init_op)

        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

        # check if a trained model exists
        if checkpoint and checkpoint.model_checkpoint_path:
            # load the graph from the trained model
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            # restore the variables
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                # get the training features and labels
                train_example_batch, train_label_batch = sess.run([train_data[0], train_data[1]])

                # decode the one-hot encoded data to single-digit integer
                train_example_batch_t = tf.transpose(train_example_batch, [2, 0, 1])
                train_example_batch_decoded = tf.argmax(train_example_batch_t, 0)
                train_example_batch_decoded_ = sess.run(train_example_batch_decoded)

                # dictionary for key-value pair input for training
                feed_dict = {x_input: train_example_batch_decoded_, y_input: train_label_batch,
                             learning_rate: LEARNING_RATE}

                summary, _, epoch_loss = sess.run([merged, optimizer, loss], feed_dict=feed_dict)

                # display training accuracy and loss every 100 steps and at step 0
                if step % 100 == 0:
                    accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
                    print('step [{}] train -- loss : {}, accuracy : {}'.format(step, epoch_loss, accuracy_))
                    train_writer.add_summary(summary, step)
                    saver.save(sess, checkpoint_path + model_name, global_step=step)

                # display validation accuracy and loss every 100 steps
                if step % 100 == 0 and step > 0:
                    # get the validation features and labels
                    test_example_batch, test_label_batch = sess.run([test_data[0], test_data[1]])

                    # decode the one-hot encoded data to single-digit integer
                    test_example_batch_t = tf.transpose(test_example_batch, [2, 0, 1])
                    test_example_batch_decoded = tf.argmax(test_example_batch_t, 0)
                    test_example_batch_decoded_ = sess.run(test_example_batch_decoded)

                    # dictionary for key-value pair input for validation
                    feed_dict = {x_input: test_example_batch_decoded_, y_input: test_label_batch}

                    summary, test_loss, test_accuracy = sess.run([merged, loss, accuracy], feed_dict=feed_dict)

                    print('step [{}] validation -- loss : {}, accuracy : {}'.format(step, test_loss, test_accuracy))
                    validation_writer.add_summary(summary, step)

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

        saver.save(sess, checkpoint_path + model_name, global_step=step)


def parse_args():
    parser = argparse.ArgumentParser(description='SVM for Intrusion Detection')
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
