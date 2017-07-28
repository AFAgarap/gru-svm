import argparse
import data
import numpy as np
import os
import tensorflow as tf

TRAIN_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/train/6/attack'
TEST_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/test'

SESSION_CONFIG = tf.ConfigProto(
		device_count = {'CPU' : 1, 'GPU' : 0},
		allow_soft_placement=True,
		log_device_placement=True,
	)

def main(examples, labels, keys):

	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

	with tf.Session() as sess:
		sess.run(init_op)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		try:
			while not coord.should_stop():
				example_batch, label_batch, key_batch = sess.run([examples, labels, keys])
				print('[{}] : {}, {}'.format(key_batch, example_batch, label_batch))
				print(type(example_batch), ': ', type(label_batch))
				print('{} : {} : {}'.format(key_batch.__len__(), example_batch.__len__(), label_batch.__len__()))
		except tf.errors.OutOfRangeError:
			print('EOF reached.')
		finally:
			coord.request_stop()
		coord.join(threads)


def parse_args():
	parser = argparse.ArgumentParser(description='GRU-SVM Model for Intrusion Detection, built with tf.scan')
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
		examples, labels, keys = data.input_pipeline(path=TRAIN_PATH, num_epochs=1)
		main(examples, labels, keys)

	elif args['test']:
		examples, labels, keys = data.input_pipeline(path=TEST_PATH, num_epochs=1)