import csv
import os
from os import walk

months = []
data = []

path = '/home/darth/Documents/Adamson University/CS Research Project 1/Kyoto 2013 Dataset/Kyoto2016/2013/'

for (dirpath, dirnames, filenames) in walk(path):
	months.extend(os.path.join(dirpath, dirname) for dirname in dirnames)
	break

for index in range(len(months)):
	for (dirpath, dirnames, filenames) in walk(months[index]):
		data.extend(os.path.join(dirpath, filename) for filename in filenames)
		break


for index in range(len(data)):
	try:
		print('Processing: {}'.format(data[index]))
		in_csv = csv.reader(open(data[index], 'r'), delimiter='\t')
		out_csv = csv.writer(open(csv_data[index], 'w'))
		out_csv.writerows(in_csv)
	except:
		print('File not found: {}'.format(data[index]))