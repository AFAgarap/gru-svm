import csv
import os
from os import walk

PATH = '/home/darth/Documents/Adamson University/CS Research Project/Kyoto 2013 Dataset/CSV'

data = []

def main():
	#
	
def list_files(path):
	file_list = []
	for (dirpath, dirnames, filenames) in walk(path):
		file_list.extend(os.path.join(dirpath, filename) for filename in filenames)
	return file_list

def linear_scale(min, max, x):
	'''Scales integer values [min, max] -> [0.0, 1.0]'''
	if (x >= min and x <= max):
		scaled_value = (((1.0 - 0.0) * (x - min)) / (max - min)) + 0.0
	else:
		scaled_value = -1
	return scaled_value

if __name__ == '__main__':
	main()