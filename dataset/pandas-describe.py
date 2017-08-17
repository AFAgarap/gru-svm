# Copyright 2017 Abien Fred Agarap. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for getting the dataset description based on sample data"""
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
	

# append the data from files to df
for file in files:
	df = df.append(pd.read_csv(filepath_or_buffer=file, names=COL_NAMES, engine='python'))

# print the data summary
print(df.describe())