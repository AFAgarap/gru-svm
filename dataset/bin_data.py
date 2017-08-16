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

"""Bins continuous data into 10 evenly-spaced intervals"""
import numpy as np
import os
import pandas as pd
import standardize_data as sd
import tensorflow as tf

NUM_CHUNKS = 10
PATH = '/home/darth/GitHub Projects/gru_svm/backup/5'
WRITE_PATH = '/home/darth/GitHub Projects/gru_svm/dataset/train/foo'
column_names = sd.col_names
columns_to_save = list(column_names)
columns_to_save.remove('dst_ip_add')
columns_to_save.remove('src_ip_add')
cols_to_std = sd.cols_to_std
cols_to_std.append('service')
cols_to_std.append('flag')

files = sd.list_files(path=PATH)
df = pd.DataFrame()

for file in files:
    df = df.append(pd.read_csv(filepath_or_buffer=file, names=column_names))
    print('appending : {}'.format(file))

df = df.drop(labels=['dst_ip_add', 'src_ip_add'], axis=1)

for index in range(len(cols_to_std)):
    bins = np.linspace(df[cols_to_std[index]].min(), df[cols_to_std[index]].max(), 10)
    df[cols_to_std[index]] = np.digitize(df[cols_to_std[index]], bins, right=True)

for id, df_i in enumerate(np.array_split(df, NUM_CHUNKS)):
	df_i.to_csv(path_or_buf=os.path.join(WRITE_PATH, '{id}.csv'.format(id=id)), columns=columns_to_save, header=None, index=False)
	print('Saving CSV file : {path}'.format(path=os.path.join(WRITE_PATH, '{id}'.format(id=id))))