import os
import pandas as pd
import tensorflow as tf
import standardize_data

# the path of the dataset to check
PATH = '/home/darth/GitHub Projects/gru_svm/dataset/train/5/normal'

# list to contain the filenames in PATH
files = []

# append the files under PATH to the list files[]
for dirpath, dirnames, filenames in os.walk(PATH):
	files.extend(os.path.join(dirpath, filename) for filename in filenames)

# TF queue to contain the list of files
filename_queue = tf.train.string_input_producer(files)

# TF reader
reader = tf.TextLineReader()

# default values, in case of empty columns
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

print('----')

# dataframe to contain the desired dataset
df = pd.DataFrame()

# dataframe to contain the wrong dataset
# this will only be used for checking
# the result must be __len__() == 0
df_n = pd.DataFrame()

for file in files:
	df = df.append(pd.read_csv(filepath_or_buffer=file, names=standardize_data.col_names, engine='python'))

print('Done appending...')

# check if there are any label
# that does not belong to the df
df_n = df[df['label'] == 1]

# if there are no outcast data
# the splitting of dataset was successful
print('Success' if df_n.__len__() == 0 else 'An unexpected error occurred.')