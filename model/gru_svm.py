import argparse
import data
import numpy as np
import os
import tensorflow as tf


BATCH_SIZE = 200
CELLSIZE = 512
NLAYERS = 3
SVMC = 1
learning_rate = 0.01

CKPT_PATH = 'ckpt/gru_svm'
MODEL_NAME = 'gru_svm'
TRAIN_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/train/6'
TEST_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/test'

SESSION_CONFIG = tf.ConfigProto(
		device_count = {'CPU' : 1, 'GPU' : 0},
		allow_soft_placement=True,
		log_device_placement=True,
	)


def main():
	examples, labels, keys = data.input_pipeline(path=TRAIN_PATH, batch_size=BATCH_SIZE, num_epochs=1)

	seqlen = examples.shape[1]

	x = tf.placeholder(shape=[None, seqlen, 1], dtype=tf.float32, name='x')
	y_input = tf.placeholder(shape=[None], dtype=tf.int32, name='y_input')
	y = tf.one_hot(y_input, 2, dtype=tf.float32, name='y')
	Hin = tf.placeholder(shape=[None, CELLSIZE*NLAYERS], dtype=tf.float32, name='Hin')

	# cell = tf.contrib.rnn.GRUCell(CELLSIZE)
	network = []
	for index in range(NLAYERS):
		network.append(tf.contrib.rnn.GRUCell(CELLSIZE))

	mcell = tf.contrib.rnn.MultiRNNCell(network, state_is_tuple=False)
	Hr, H = tf.nn.dynamic_rnn(mcell, x, initial_state=Hin, dtype=tf.float32)

	Hf = tf.transpose(Hr, [1, 0, 2])
	last = tf.gather(Hf, int(Hf.get_shape()[0]) - 1)

	weight = tf.Variable(tf.truncated_normal([CELLSIZE, 2], stddev=0.01), tf.float32, name='weights')
	bias = tf.Variable(tf.constant(0.1, shape=[2]), name='bias')
	logits = tf.matmul(last, weight) + bias

	regularization_loss = 0.5 * tf.reduce_sum(tf.square(weight))
	hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE, 1]), 1 - y * logits))
	loss = regularization_loss + SVMC * hinge_loss

	train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

	with tf.Session() as sess:
		sess.run(init_op)

		train_loss = 0
		ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
		saver = tf.train.Saver()
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		try:
			for index in range(10):
				# for j in range(1000):
				example_batch, label_batch, key_batch = sess.run([examples, labels, keys])
				_, train_loss_ = sess.run([train_step, loss],
					feed_dict = { x : example_batch[..., np.newaxis],
									y_input : label_batch,
									Hin : np.zeros([BATCH_SIZE, CELLSIZE * NLAYERS])
								})
				train_loss += train_loss_
				print('[{}] loss : {}'.format(index, (train_loss / 1000)))
				train_loss = 0
		except tf.errors.OutOfRangeError:
			print('EOF reached.')
		except KeyboardInterrupt:
			print('Interrupted by user at {}'.format(index))
		finally:
			coord.request_stop()
		coord.join(threads)
		saver = tf.train.Saver()
		saver.save(sess, CKPT_PATH + MODEL_NAME, global_step=index)

main()