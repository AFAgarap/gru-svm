from os.path import join
from os import walk
import pandas as pd
import tensorflow as tf

COL_NAMES = ['duration', 'service', 'src_bytes', 'dest_bytes', 'count', 'same_srv_rate',
			'serror_rate', 'srv_serror_rate', 'dst_host_count', 'dst_host_srv_count',
			'dst_host_same_src_port_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
			'flag', 'ids_detection', 'malware_detection', 'ashula_detection', 'label', 'src_ip_add',
			'src_port_num', 'dst_ip_add', 'dst_port_num', 'start_time', 'protocol']

def file_len(filename):
	with open(filename) as file:
		for index, line in enumerate(file):
			pass
	return index + 1

def list_files(path):
	'''Returns the list of files present in the path'''
	file_list = []
	for (dirpath, dirnames, filenames) in walk(path):
		file_list.extend(join(dirpath, filename) for filename in filenames)
	return file_list

def read_from_csv(filename_queue):

	# TF reader
	reader = tf.TextLineReader()

	# default values, in case of empty columns
	record_defaults = [[0.0] for x in range(24)]

	key, value = reader.read(filename_queue)

	duration, service, src_bytes, dest_bytes, count, same_srv_rate, \
	serror_rate, srv_serror_rate, dst_host_count, dst_host_srv_count, \
	dst_host_same_src_port_rate, dst_host_serror_rate, dst_host_srv_serror_rate, \
	flag, ids_detection, malware_detection, ashula_detection, label, src_ip_add, \
	src_port_num, dst_ip_add, dst_port_num, start_time, protocol = \
	tf.decode_csv(value, record_defaults=record_defaults)

	features = tf.stack([duration, service, src_bytes, dest_bytes, count, same_srv_rate,
						serror_rate, srv_serror_rate, dst_host_count, dst_host_srv_count,
						dst_host_same_src_port_rate, dst_host_serror_rate, dst_host_srv_serror_rate,
						flag, ids_detection, malware_detection, ashula_detection, src_ip_add,
						src_port_num, dst_ip_add, dst_port_num, start_time, protocol])

	return features, label, key

def input_pipeline(path, batch_size=10, num_epochs=None):

	# create a list to store the filenames
	files = list_files(path=path)

	filename_queue = tf.train.string_input_producer(files, num_epochs=1, shuffle=True)

	example, label, key = read_from_csv(filename_queue)

	min_after_dequeue = 10 * batch_size
	capacity = min_after_dequeue + 3 * batch_size

	example_batch, label_batch, key_batch = tf.train.shuffle_batch(
		[example, label, key], batch_size=batch_size, capacity=capacity,
		min_after_dequeue=min_after_dequeue)

	return example_batch, label_batch, key_batch
	
	# with tf.Session(config=SESSION_CONFIG) as sess:
	# 	coord = tf.train.Coordinator()
	# 	threads = tf.train.start_queue_runners(coord=coord)

	# 	try:
	# 		while not coord.should_stop():

	# 	except tf.errors.OutOfRangeError:
	# 		print('EOF Reached.')
	# 	finally:
	# 		coord.request_stop()
	# 	coord.join(threads)