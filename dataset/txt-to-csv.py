import csv
import os
from os import walk

path = '/home/darth/Documents/Adamson University/CS Research Project/Kyoto 2013 Dataset/Kyoto2016/2013/'
csv_path = '/home/darth/Documents/Adamson University/CS Research Project/Kyoto 2013 Dataset/CSV'

data = []	# list to store the filenames under the subdirectories of the <path>
csv_data = []	# list to store the converted CSV files

for (dirpath, dirnames, filenames) in walk(path):
	''' Append to the list the filenames under the subdirectories of the <path> '''
	data.extend(os.path.join(dirpath, filename) for filename in filenames)

# Create the <csv_path> if it does not exist
os.makedirs(csv_path) if not os.path.exists(csv_path) else print('CSV folder exists')

for month in range(12):
	''' Create the subdirectories under the <csv_path> if it does not exist '''
	if next(walk(csv_path))[1].__len__() == 12:
		print('Folders exist')
		break
	print('Creating subdirectories.')
	# get the dirpath from the generator object <walk> (index 0)
	# then joins the dirpath with the month number
	os.makedirs(os.path.join(next(walk(csv_path))[0], '0' + str(month + 1) if month < 9 else str(month + 1)))
	

for index in range(len(data)):
	''' Store the processed CSV filename to <csv_data> list '''
	csv_data.append(os.path.join(csv_path, data[index].split(path)[1].replace('txt', 'csv')))

for index in range(len(data)):
	''' Reading the text files delimited with tab, and converts it to CSV '''
	try:
		print('Processing: {}'.format(data[index]))
		in_csv = csv.reader(open(data[index], 'r'), delimiter='\t')
		out_csv = csv.writer(open(csv_data[index], 'x'))
		out_csv.writerows(in_csv)
	except:
		print('File not found: {}'.format(data[index]))