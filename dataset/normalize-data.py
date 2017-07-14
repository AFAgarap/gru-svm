import csv
import os
from os import walk

PATH = '/home/darth/Documents/Adamson University/CS Research Project/Kyoto 2013 Dataset/CSV'

data = []

def main():
	# list all CSV files under the PATH
	data = list_files(path=PATH)
	for datum in data:
		# read CSV data
		in_csv = csv.reader(open(datum, 'r'))

	
def list_files(path):
	file_list = []
	for (dirpath, dirnames, filenames) in walk(path):
		file_list.extend(os.path.join(dirpath, filename) for filename in filenames)
	return file_list

def feature_scale(x):
	'''Scales integer values [min, max] -> [0.0, 1.0]'''
	return [ (x_i - min(x)) / (max(x) - min(x)) for x_i in x ]

def scale_boolean_values(value):
	'''Scales the supposed boolean values in the dataset'''
	if (value != 0):
		return 1
	elif (value == 0):
		return 0

def scale_label(label):
	'''Scales the labels [-2 -1 1] -> [0, 1]'''
	if (label == -2 or label == -1):
		return 1	# -2 and -1 represents an attack
	elif (label == 1):
		return 0	# 1 represents normal session

def convert_time_to_integer(time):
	'''Converts time to integer value, preparation for scaling'''
	h, m, s = map(int, time.split(':'))
	parsed_time = h + (m * (1 / 60)) + (s * (1 / 3600))
	return parsed_time

def standardize(x):
	'''Method for normalizing using Student's t-statistic'''
	mean = sum(x) / len(x)
	std_dev = (1 / len(x) * sum([ (x_i - mean) ** 2 for x_i in x] )) ** 0.5
	z_score = [ (x_i - mean) / std_dev for x_i in x ]
	return z_score

if __name__ == '__main__':
	main()