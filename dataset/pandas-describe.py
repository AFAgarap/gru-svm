import numpy as np
import pandas as pd
import os
from os import walk

PATH = '/home/darth/Desktop/pandas'

def main():
	# get all the CSV files in the PATH dir
	files = list_files(path=PATH)

	# create empty df, where dfs shall be appended
	data = pd.DataFrame(dtype=str)

	# for file in files:
	for index in range(len(files)):
		print('{}'.format(index), end=' ')
		data = data.append(pd.read_csv(filepath_or_buffer=files[index], header=None, na_values=['--']))
		print()
	
	for index in range(0, 47):
		print(data.iloc[index])

def list_files(path):
	file_list = []
	for (dirpath, dirnames, filenames) in walk(path):
		file_list.extend(os.path.join(dirpath, filename) for filename in filenames)
	return file_list

if __name__ == '__main__':
	main()