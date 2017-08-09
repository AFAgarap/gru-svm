import argparse
import data
import math
import numpy as np
import tensorflow as tf
import time

BATCH_SIZE = 500
CELLSIZE = 512
NLAYERS = 3
SVMC = 1
learning_rate = 1e-6
pkeep = 0.8

CKPT_PATH = 'ckpt/gru_svm/'
MODEL_NAME = 'gru_svm'
TRAIN_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/train/foobar'
TEST_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/test'
LOGS_PATH = 'logs/'


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


def main():
    examples, labels, keys = data.input_pipeline(path=TRAIN_PATH, batch_size=BATCH_SIZE, num_epochs=1)

    seqlen = examples.shape[1]
    with tf.name_scope('input'):
        x = tf.placeholder(shape=[None, seqlen, 1], dtype=tf.float32, name='x_input')
        y_input = tf.placeholder(shape=[None], dtype=tf.int32, name='y_input')
        y = tf.one_hot(y_input, 2, 1.0, -1.0, dtype=tf.float32, name='y_onehot')
    Hin = tf.placeholder(shape=[None, CELLSIZE * NLAYERS], dtype=tf.float32, name='Hin')

    network = [tf.contrib.rnn.GRUCell(CELLSIZE) for _ in range(NLAYERS)]
    dropcells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in network]
    mcell = tf.contrib.rnn.MultiRNNCell(dropcells, state_is_tuple=False)
    mcell = tf.contrib.rnn.DropoutWrapper(mcell, output_keep_prob=pkeep)

    Hr, H = tf.nn.dynamic_rnn(mcell, x, initial_state=Hin, dtype=tf.float32)
    # Hr, H = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    Hf = tf.transpose(Hr, [1, 0, 2])
    last = tf.gather(Hf, int(Hf.get_shape()[0]) - 1)

    with tf.name_scope('weights'):
        weight = tf.Variable(tf.truncated_normal([CELLSIZE, 1], stddev=0.01), tf.float32, name='weight')
        variable_summaries(weight)
    with tf.name_scope('biases'):
        bias = tf.Variable(tf.constant(0.1, shape=[1]), name='bias')
        variable_summaries(bias)
    with tf.name_scope('Wx_plus_b'):
        logits = tf.matmul(last, weight) + bias
        tf.summary.histogram('pre-activations', logits)
    # activations = act(logits, name='activation')
    # tf.summary.histogram('activations', activations)

    # prediction = tf.nn.softmax(logits)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input)
    # train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)

    with tf.name_scope('svm_loss'):
        regularization_loss = 0.5 * tf.reduce_sum(tf.square(weight))
        hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE, 1]), 1 - tf.cast(y_input, tf.float32) * logits))
        with tf.name_scope('loss'):
            loss = regularization_loss + SVMC * hinge_loss
    tf.summary.scalar('loss', loss)

    with tf.name_scope('accuracy'):
        predicted_class = tf.sign(logits)
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(y_input, tf.cast(predicted_class, tf.int32))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    merged = tf.summary.merge_all()
    timestamp = str(math.trunc(time.time()))
    writer = tf.summary.FileWriter(LOGS_PATH + timestamp, graph=tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=1000)

    with tf.Session() as sess:
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(CKPT_PATH)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        train_loss = 0

        try:
            inH = np.zeros([BATCH_SIZE, CELLSIZE * NLAYERS])
            for index in range(100):
                # while not coord.should_stop():
                example_batch, label_batch, key_batch = sess.run([examples, labels, keys])

                label_batch[label_batch == 0] = -1

                feed_dict = {x: example_batch[..., np.newaxis], y_input: label_batch, Hin: inH}

                summary, _, train_loss_, outH, accuracy_, y_input_ = sess.run(
                    [merged, train_step, loss, H, accuracy, y_input],
                    feed_dict=feed_dict)

                if index % 10 == 0:
                    writer.add_summary(summary, index)
                    saved_file = saver.save(sess, CKPT_PATH + MODEL_NAME + timestamp, global_step=index)

                train_loss += train_loss_

                print('[{}] loss : {}, accuracy : {}'.format(index, (train_loss / 1000), accuracy_))

                train_loss = 0

                inH = outH
        except tf.errors.OutOfRangeError:
            print('EOF reached.')
        except KeyboardInterrupt:
            print('Interrupted by user at {}'.format(index))
        finally:
            coord.request_stop()
            writer.close()
        coord.join(threads)
        saver = tf.train.Saver()
        saver.save(sess, CKPT_PATH + MODEL_NAME, global_step=index)

        print('Accuracy : {}'.format(sess.run(accuracy,
                                              feed_dict={x: example_batch[..., np.newaxis], y_input: label_batch,
                                                         Hin: inH})))


def parse_args():
    parser = argparse.ArgumentParser(description='GRU-SVM Model for Intrusion Detection')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--test', action='store_true',
                       help='test trained model')
    group.add_argument('-t', '--train', action='store_true',
                       help='train model')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = parse_args()

    if args['train']:
        # fetch the data
        # examples, labels, keys = data.input_pipeline(path=TRAIN_PATH, batch_size=BATCH_SIZE, num_epochs=1)
        # main(examples, labels, keys)
        main()

    elif args['test']:
        examples, labels, keys = data.input_pipeline(path=TEST_PATH, num_epochs=1)
