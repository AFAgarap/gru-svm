import os
import pandas as pd
import standardize_data

# a sample data to describe using pandas
PATH = '/home/darth/GitHub Projects/gru_svm/dataset/train/4/attack'

# get the column names from standardize_data.py module
COL_NAMES = standardize_data.col_names

# list to contain the filenames
files = []

# instantiate a dataframe
df = pd.DataFrame()

# append the filenames to the list
for (dirpath, dirnames, filenames) in os.walk(PATH):
	files.extend(os.path.join(dirpath, filename) for filename in filenames)
	print('Appending file : {}'.format(file))

# append the data from files to df
for file in files:
	df = df.append(pd.read_csv(filepath_or_buffer=file, names=COL_NAMES, engine='python'))

# print the data summary
print(df.describe())