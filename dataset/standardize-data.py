import numpy as np
import pandas as pd
import os
from os import walk
from sklearn import preprocessing

PATH = '/home/darth/Desktop/pandas'
WRITE_PATH = '/home/darth/Desktop/preprocessed/'
NUM_CHUNKS = 10

col_names = ['duration', 'service', 'src_bytes', 'dest_bytes', 'count', 'same_srv_rate',
			'serror_rate', 'srv_serror_rate', 'dst_host_count', 'dst_host_srv_count',
			'dst_host_same_src_port_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
			'flag', 'ids_detection', 'malware_detection', 'ashula_detection', 'label', 'src_ip_add',
			'src_port_num', 'dst_ip_add', 'dst_port_num', 'start_time', 'protocol']

cols_to_std = ['duration', 'src_bytes', 'dest_bytes', 'count',
				'same_srv_rate', 'serror_rate', 'srv_serror_rate',
				'dst_host_count', 'dst_host_srv_count',
				'dst_host_same_src_port_rate', 'dst_host_serror_rate',
				'dst_host_srv_serror_rate', 'src_port_num',
				'dst_port_num', 'start_time']

cols_to_index = ['ashula_detection', 'dst_ip_add', 'flag', 'ids_detection', 'label',
					'malware_detection', 'protocol', 'service', 'src_ip_add']

def main():
	# get all the CSV files in the PATH dir
	files = list_files(path=PATH)

	# create empty df, where dfs shall be appended
	df = pd.DataFrame(dtype=str)

	# append the dfs from each file to the data df
	# for file in files:
	for index in range(0, 11):
		df = df.append(pd.read_csv(filepath_or_buffer=files[index], names=col_names, engine='python'))
		print('Appending {}'.format(index))

	# df[col_names] = df[col_names].dropna(axis=0, how='any')
	
	print(df)

	df[['malware_detection', 'ashula_detection', 'ids_detection']] = \
	df[['malware_detection', 'ashula_detection', 'ids_detection']].apply(lambda x : 1 if x != '0' else 0)

	df['start_time'] = df['start_time'].apply(lambda time: int(time.split(':')[0]) +
											(int(time.split(':')[1]) * (1 / 60)) +
											(int(time.split(':')[2]) * (1 / 3600)))

	df[cols_to_index] = df[cols_to_index].apply(preprocessing.LabelEncoder().fit_transform)
	df[cols_to_std] = preprocessing.StandardScaler().fit_transform(df[cols_to_std])

	print(df)

	# df.to_csv(path_or_buf=WRITE_PATH, columns=col_names, header=None, index=False)
	[df.to_csv(path_or_buf=WRITE_PATH + '/{}.csv'.format(id=id), columns=col_names, header=None, index=False)
		for id, df in enumerate(np.array_split(df, NUM_CHUNKS))]
	print('Done')

def list_files(path):
	file_list = []
	for (dirpath, dirnames, filenames) in walk(path):
		file_list.extend(os.path.join(dirpath, filename) for filename in filenames)
	return file_list

if __name__ == '__main__':
	main()