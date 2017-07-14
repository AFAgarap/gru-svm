import numpy as np
import pandas as pd
import os
from os import walk
from sklearn import preprocessing

PATH = '/home/darth/Desktop/pandas'

def main():
	# get all the CSV files in the PATH dir
	files = list_files(path=PATH)

	# create empty df, where dfs shall be appended
	data = pd.DataFrame(dtype=str)

	# append the dfs from each file to the data df
	for file in files:
		data = data.append(pd.read_csv(filepath_or_buffer=file, header=None, engine='python'))
	
	data[22] = data[22].apply(lambda time: int(time.split(':')[0]) +
		(int(time.split(':')[1]) * (1 / 60)) +
		(int(time.split(':')[2]) * (1 / 3600)))

	std_scale = preprocessing.StandardScaler().fit(data[0])
	data_std = std_scale.transform(data[0])

	print('Mean after standardization: \ndata[0]={}'.format(data_std.mean()))
	print('Std dev after standardization: \ndata[0]={}'.format(data_std.std()))

	print(data_std)
	# print(data.describe())

def standardize(x):
	mean = sum(x) / len(x)
	std_dev = (1 / len(x) * sum([ (x_i - mean) ** 2 for x_i in x] )) ** 0.5
	z_score = [ (x_i - mean) / std_dev for x_i in x ]
	return z_score

def list_files(path):
	file_list = []
	for (dirpath, dirnames, filenames) in walk(path):
		file_list.extend(os.path.join(dirpath, filename) for filename in filenames)
	return file_list

if __name__ == '__main__':
	main()