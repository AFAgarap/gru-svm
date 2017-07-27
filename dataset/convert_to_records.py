import argparse
import os
import sys
import standardize_data
import tensorflow as tf

FLAGS = None
PATH = '/home/darth/GitHub Projects/gru_svm/dataset/train'

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(dataset, name):
	"""Converts a dataset to tfrecords"""

	filename_queue = tf.train.string_input_producer(dataset, num_epochs=1)

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

	filename = os.path.join(FLAGS.directory, name + '.tfrecords')
	print('Writing {}'.format(filename))
	writer = tf.python_io.TFRecordWriter(filename)
	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		try:
			while not coord.should_stop():
				example, l = sess.run([features, label])
				print('Writing {dataset} : {example}, {label}'.format(dataset=sess.run(key),
						example=example, label=l))
				example_to_write = tf.train.Example(features=tf.train.Features(feature={
					'duration' : _float_feature(example[0]),
					'service' : _int64_feature(int(example[1])),
					'src_bytes' : _float_feature(example[2]),
					'dest_bytes' : _float_feature(example[3]),
					'count' : _float_feature(example[4]),
					'same_srv_rate' : _float_feature(example[5]),
					'serror_rate' : _float_feature(example[6]),
					'srv_serror_rate' : _float_feature(example[7]),
					'dst_host_count' : _float_feature(example[8]),
					'dst_host_srv_count' : _float_feature(example[9]),
					'dst_host_same_src_port_rate' : _float_feature(example[10]),
					'dst_host_serror_rate' : _float_feature(example[11]),
					'dst_host_srv_serror_rate' : _float_feature(example[12]),
					'flag' : _int64_feature(int(example[13])),
					'ids_detection' : _int64_feature(int(example[14])),
					'malware_detection' : _int64_feature(int(example[15])),
					'ashula_detection' : _int64_feature(int(example[16])),
					'label' : _int64_feature(int(l)),
					'src_ip_add' : _float_feature(example[17]),
					'src_port_num' : _float_feature(example[18]),
					'dst_ip_add' : _float_feature(example[19]),
					'dst_port_num' : _float_feature(example[20]),
					'start_time' : _float_feature(example[21]),
					'protocol' : _int64_feature(int(example[22])),
					}))
				writer.write(example_to_write.SerializeToString())
			writer.close()
		except tf.errors.OutOfRangeError:
			print('Done converting -- EOF reached.')
		finally:
			coord.request_stop()

		coord.join(threads)

def main(unused_argv):
	files = standardize_data.list_files(path=PATH)

	convert_to(dataset=files, name='train')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--directory',
		type=str,
		default='/home/darth/GitHub Projects/gru_svm/dataset/train/tfrecords',
		help='Directory to write the converted result'
		)

	parser.add_argument(
		'--validation_size',
		type=int,
		default=5000,
		help="""\
		Number of examples to separate from the training data for the validation set.\
		"""
		)

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)