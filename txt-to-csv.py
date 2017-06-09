import csv
import os
from os import walk

path = '/home/darth/Documents/Adamson University/CS Research Project 1/Kyoto 2013 Dataset/Kyoto2016/2013/'

months = []	# list to store the subdirectories (months 01-12) under the <path>
data = []	# list to store the days under the subdirectories

for (dirpath, dirnames, filenames) in walk(path):
	''' append to <months> list the subdirectories under <path> '''
	months.extend(os.path.join(dirpath, dirname) for dirname in dirnames)
	break

for index in range(len(months)):
	''' loops through the <months> list from 01-12 '''
	for (dirpath, dirnames, filenames) in walk(months[index]):
		''' append to <data> list the filenames under each subdirectories of the <path> '''
		data.extend(os.path.join(dirpath, filename) for filename in filenames)
		break


for index in range(len(data)):
	''' Reading the text files delimited with tab, and converts it to CSV '''
	try:
		print('Processing: {}'.format(data[index]))
		in_csv = csv.reader(open(data[index], 'r'), delimiter='\t')
		out_csv = csv.writer(open(csv_data[index], 'w'))
		out_csv.writerows(in_csv)
	except:
		print('File not found: {}'.format(data[index]))