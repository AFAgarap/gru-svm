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

	# append the dfs from each file to the data df
	for file in files:
		data = data.append(pd.read_csv(filepath_or_buffer=file, header=None, engine='python'))
	
	print(data.describe())

def list_files(path):
	file_list = []
	for (dirpath, dirnames, filenames) in walk(path):
		file_list.extend(os.path.join(dirpath, filename) for filename in filenames)
	return file_list

if __name__ == '__main__':
	main()