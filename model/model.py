import argparse
import data
import numpy as np
import os
import tensorflow as tf

TRAIN_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/train'
TEST_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/test'

def parse_args():
	parser = argparse.ArgumentParser(description='GRU-SVM Model for Intrusion Detection, built with tf.scan')
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('-g', '--test', action='store_true',
		help='test train model')
	group.add_argument('-t', '--train', action='store_true',
		help='train model')
	args = vars(parser.parse_args())
	return args

if __name__ == '__main__':
	args = parse_args()

	if args['train']:
		data = data.load_data(TRAIN_PATH)
		
	elif args['test']:
		print('test')