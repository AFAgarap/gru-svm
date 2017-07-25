import numpy as np
import pandas as pd
import os
from os import walk
from sklearn import preprocessing

# path of the dataset to be standardized
PATH = '/home/darth/Desktop/pandas/train/5'
# destination path of standardized dataset
WRITE_PATH = '/home/darth/Desktop/preprocessed/train/5'
# number of splits for the dataset
NUM_CHUNKS = 5

# column names of 24 features
col_names = ['duration', 'service', 'src_bytes', 'dest_bytes', 'count', 'same_srv_rate',
			'serror_rate', 'srv_serror_rate', 'dst_host_count', 'dst_host_srv_count',
			'dst_host_same_src_port_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
			'flag', 'ids_detection', 'malware_detection', 'ashula_detection', 'label', 'src_ip_add',
			'src_port_num', 'dst_ip_add', 'dst_port_num', 'start_time', 'protocol']

# column names of continuous and quasi-continuous features
cols_to_std = ['duration', 'src_bytes', 'dest_bytes', 'count',
				'same_srv_rate', 'serror_rate', 'srv_serror_rate',
				'dst_host_count', 'dst_host_srv_count',
				'dst_host_same_src_port_rate', 'dst_host_serror_rate',
				'dst_host_srv_serror_rate', 'src_port_num',
				'dst_port_num', 'start_time']

# column names of categorical data
cols_to_index = ['ashula_detection', 'dst_ip_add', 'flag', 'ids_detection', 'label',
					'malware_detection', 'protocol', 'service', 'src_ip_add']

def main():
	# get all the CSV files in the PATH dir
	files = list_files(path=PATH)

	# create empty df, where dfs shall be appended
	df = pd.DataFrame()

	# append the dfs from each file to the data df
	for file in files:
		# the python engine was used to support mixed data types
		df = df.append(pd.read_csv(filepath_or_buffer=file, names=col_names, engine='python'))
		print('Appending {}'.format(file))

	# drop rows with NaN values
	df[col_names] = df[col_names].dropna(axis=0, how='any')

	# since malware_detection, ashula_detection,
	# and ids_detection col contains string data
	# replace if the string != '0' with int 1
	# otherwise with int 0
	df['malware_detection'] = df['malware_detection'].apply(lambda malware_detection : 1 if malware_detection != '0' else 0)
	df['ashula_detection'] = df['ashula_detection'].apply(lambda ashula_detection : 1 if ashula_detection != '0' else 0)
	df['ids_detection'] = df['ids_detection'].apply(lambda ids_detection : 1 if ids_detection != '0' else 0)

	# convert time to continuous data
	df['start_time'] = df['start_time'].apply(lambda time: int(time.split(':')[0]) +
												(int(time.split(':')[1]) * (1 / 60)) +
												(int(time.split(':')[2]) * (1 / 3600)))

	# index categorical data to [0, n-1] where n is the number of categories per feature
	df[cols_to_index] = df[cols_to_index].apply(preprocessing.LabelEncoder().fit_transform)

	# standardize continuous and quasi-continuous features
	df[cols_to_std] = preprocessing.StandardScaler().fit_transform(df[cols_to_std])

	# split the dataframe to multiple CSV files
	for id, df_i in enumerate(np.array_split(df, NUM_CHUNKS)):
		df_i.to_csv(path_or_buf=os.path.join(WRITE_PATH, '{id}.csv'.format(id=id)), columns=col_names, header=None, index=False)
		print('Saving CSV file : {path}'.format(path=os.path.join(WRITE_PATH, '{id}'.format(id=id))))

	print('Done')

def list_files(path):
	'''Returns the list of files present in the path'''
	file_list = []
	for (dirpath, dirnames, filenames) in walk(path):
		file_list.extend(os.path.join(dirpath, filename) for filename in filenames)
	return file_list

if __name__ == '__main__':
	main()