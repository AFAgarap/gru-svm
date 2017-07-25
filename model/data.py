import os
import pandas as pd
import sys

col_names = ['duration', 'service', 'src_bytes', 'dest_bytes', 'count', 'same_srv_rate',
			'serror_rate', 'srv_serror_rate', 'dst_host_count', 'dst_host_srv_count',
			'dst_host_same_src_port_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
			'flag', 'ids_detection', 'malware_detection', 'ashula_detection', 'label', 'src_ip_add',
			'src_port_num', 'dst_ip_add', 'dst_port_num', 'start_time', 'protocol']

def load_data(path):
	print(path)
	# create a list to store the filenames
	files = []
	
	# create a dataframe to store the contents of CSV files
	df = pd.DataFrame()

	# get the filenames in the specified PATH
	for (dirpath, dirnames, filenames) in os.walk(path):
		''' Append to the list the filenames under the subdirectories of the <path> '''
		files.extend(os.path.join(dirpath, filename) for filename in filenames)
	
	for index in range(0, 26):
		df = df.append(pd.read_csv(filepath_or_buffer=files[index], names=col_names, engine='python'))
		print('Appending file : {file}'.format(file=files[index]))

	pd.set_option('display.max_colwidth', -1)
	print(df)