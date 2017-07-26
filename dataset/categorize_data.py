import numpy as np
import os
import pandas as pd
import standardize_data

# path of the normalized data to categorize
FILE_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/train/5'

# path where to save the data with 'under attack' sessions
FILE_WITH_ATTACK = '/home/darth/GitHub Projects/gru_svm/dataset/train/5/attack'

# path where to save the data with normal sessions
FILE_WITH_NORMAL = '/home/darth/GitHub Projects/gru_svm/dataset/train/5/normal'

# number of files to save
NUM_CHUNKS = 4

# column names of the dataset (24 features)
COL_NAMES = standardize_data.col_names

def main():
	# get all the CSV files under the FILE_PATH dir
	files = standardize_data.list_files(path=FILE_PATH)

	# dataframe for the dataset in the FILE_PATH
	df = pd.DataFrame()

	# dataframe for the dataset with attack
	df_attack = pd.DataFrame()

	# dataframe for the dataset with normal
	df_normal = pd.DataFrame()

	# append the contents of the CSV files to the dataframe df
	for file in files:
		df = df.append(pd.read_csv(filepath_or_buffer=file, names=COL_NAMES, engine='python'))
		print('Appending file : {file}'.format(file=file))

	# store the data with attack to df_attack
	df_attack = df[df['label'] == 1]

	# store the data with normal to df_normal
	df_normal = df[df['label'] == 0]

	# delete the df to save memory
	del df

	# split df_attack into NUM_CHUNKS and
	# save the splitted files to CSV
	for id, df_attack_i in enumerate(np.array_split(df_attack, NUM_CHUNKS)):
		df_attack_i.to_csv(path_or_buf=os.path.join(FILE_WITH_ATTACK, '{id}.csv'.format(id=id)),
							columns=COL_NAMES, header=None, index=False)
		print('Saving CSV file : {path}'.format(path=os.path.join(FILE_WITH_ATTACK, '{id}.csv'.format(id=id))))

	# delete the dataframe to save memory
	del df_attack

	# split df_normal into NUM_CHUNKS and
	# save the splitted files to CSV
	for id, df_normal_i in enumerate(np.array_split(df_normal, NUM_CHUNKS)):
		df_normal_i.to_csv(path_or_buf=os.path.join(FILE_WITH_NORMAL, '{id}.csv'.format(id=id)),
							columns=COL_NAMES, header=None, index=False)
		print('Saving CSV file : {path}'.format(path=os.path.join(FILE_WITH_NORMAL, '{id}.csv'.format(id=id))))

	# delete the dataframe to save memory
	del df_normal

if __name__ == '__main__':
	main()