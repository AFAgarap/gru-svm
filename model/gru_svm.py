# algorithms to check co-linearity between variables
# identify which variables can be removed from features

import argparse
import data
import numpy as np
import os
import tensorflow as tf


BATCH_SIZE = 500
CELLSIZE = 512
NLAYERS = 3
SVMC = 1
learning_rate = 0.01

CKPT_PATH = 'ckpt/gru_svm/'
MODEL_NAME = 'gru_svm'
TRAIN_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/train/foobar'
TEST_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/test'

logs_path = 'logs/'

def main():
	examples, labels, keys = data.input_pipeline(path=TRAIN_PATH, batch_size=BATCH_SIZE, num_epochs=1)
	
	seqlen = examples.shape[1]

	x = tf.placeholder(shape=[None, seqlen, 1], dtype=tf.float32, name='x')
	y_input = tf.placeholder(shape=[None], dtype=tf.int32, name='y_input')
	# y = tf.one_hot(y_input, 2, dtype=tf.float32, name='y')
	Hin = tf.placeholder(shape=[None, CELLSIZE*NLAYERS], dtype=tf.float32, name='Hin')

	# cell = tf.contrib.rnn.GRUCell(CELLSIZE)
	network = []
	for index in range(NLAYERS):
		network.append(tf.contrib.rnn.GRUCell(CELLSIZE))

	mcell = tf.contrib.rnn.MultiRNNCell(network, state_is_tuple=False)
	Hr, H = tf.nn.dynamic_rnn(mcell, x, initial_state=Hin, dtype=tf.float32)
	# Hr, H = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

	Hf = tf.transpose(Hr, [1, 0, 2])
	last = tf.gather(Hf, int(Hf.get_shape()[0]) - 1)

	weight = tf.Variable(tf.truncated_normal([CELLSIZE, 1], stddev=0.01), tf.float32, name='weights')
	bias = tf.Variable(tf.constant(0.1, shape=[1]), name='bias')
	logits = tf.matmul(last, weight) + bias

	# prediction = tf.nn.softmax(logits)
	# loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input)
	# train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)

	regularization_loss = 0.5 * tf.reduce_sum(tf.square(weight))
	hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE, 1]), 1 - tf.cast(y_input, tf.float32) * logits))
	loss = regularization_loss + SVMC * hinge_loss	

	predicted_class = tf.sign(logits)
	correct_prediction = tf.equal(y_input, tf.cast(predicted_class, tf.int32))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	train_step = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(loss)

	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

	writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

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
			inH = np.zeros([BATCH_SIZE, CELLSIZE * NLAYERS])	
			# for index in range(100):
			while not coord.should_stop():
				example_batch, label_batch, key_batch = sess.run([examples, labels, keys])
				label_batch[label_batch == 0] = -1
				_, train_loss_, outH, accuracy_, y_input_ = sess.run([train_step, loss, H, accuracy, y_input],
					feed_dict = { x : example_batch[..., np.newaxis],
									y_input : label_batch,
									Hin : inH
								})
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
		coord.join(threads)
		saver = tf.train.Saver()
		saver.save(sess, CKPT_PATH + MODEL_NAME, global_step=index)

		print('Accuracy : {}'.format(sess.run(accuracy,
			feed_dict={ x : example_batch[..., np.newaxis], y_input : label_batch, Hin : np.zeros([BATCH_SIZE, CELLSIZE * NLAYERS])})))

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