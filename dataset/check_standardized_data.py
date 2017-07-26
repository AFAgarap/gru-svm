import os
import tensorflow as tf

PATH = '/home/darth/Desktop/preprocessed/train/3'

files = []

for dirpath, dirnames, filenames in os.walk(PATH):
	files.extend(os.path.join(dirpath, filename) for filename in filenames)

filename_queue = tf.train.string_input_producer(files)

reader = tf.TextLineReader()

record_defaults = [[1.0] for x in range(24)]

key, value = reader.read(filename_queue)

col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, \
col11, col12, col13, col14, col15, col16, col17, col18, col19, \
col20, col21, col22, col23, col24 = tf.decode_csv(value, record_defaults=record_defaults)

features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10,
					col11, col12, col13, col14, col15, col16, col17, col19,
					col20, col21, col22, col23, col24])

with tf.Session() as sess:
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	for index in range(50):
		example, label = sess.run([features, col18])
		print('{} : example {}, label {}'.format(index, example, label))

	coord.request_stop()
	coord.join(threads)