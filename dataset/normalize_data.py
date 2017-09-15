# Module for normalizing the Kyoto University 2013 Network Traffic Data
# Copyright (C) 2017  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""Dataset normalization using standardization and indexing"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1'
__author__ = 'Abien Fred Agarap'

import argparse
import numpy as np
import pandas as pd
import os
from os import walk
from sklearn import preprocessing


# column names of 24 features
col_names = ['duration', 'service', 'src_bytes', 'dest_bytes', 'count', 'same_srv_rate',
             'serror_rate', 'srv_serror_rate', 'dst_host_count', 'dst_host_srv_count',
             'dst_host_same_src_port_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
             'flag', 'ids_detection', 'malware_detection', 'ashula_detection', 'label', 'src_ip_add',
             'src_port_num', 'dst_ip_add', 'dst_port_num', 'start_time', 'protocol']

# column names of continuous and quasi-continuous features
cols_to_std = ['duration', 'src_bytes', 'dest_bytes', 'count',
               'same_srv_rate', 'serror_rate', 'srv_serror_rate',
               'dst_host_count', 'dst_host_srv_count',
               'dst_host_same_src_port_rate', 'dst_host_serror_rate',
               'dst_host_srv_serror_rate', 'src_port_num',
               'dst_port_num', 'start_time']

# column names of categorical data
cols_to_index = ['ashula_detection', 'dst_ip_add', 'flag', 'ids_detection', 'label',
                 'malware_detection', 'protocol', 'service', 'src_ip_add']


def normalize_data(path, write_path, num_chunks):
    # get all the CSV files in the PATH dir
    files = list_files(path=path)

    # create empty df, where dfs shall be appended
    df = pd.DataFrame()

    # append the dfs from each file to the data df
    for file in files:
        # the python engine was used to support mixed data types
        df = df.append(pd.read_csv(filepath_or_buffer=file, names=col_names, engine='python'))
        print('Appending {}'.format(file))

    print('Current DataFrame shape: {}'.format(df.shape))

    # drop rows with NaN values
    df[col_names] = df[col_names].dropna(axis=0, how='any')
    print('DataFrame shape after NaN values removal: {}'.format(df.shape))

    # since malware_detection, ashula_detection,
    # and ids_detection col contains string data
    # replace if the string != '0' with int 1
    # otherwise with int 0
    df['malware_detection'] = df['malware_detection'].apply(
        lambda malware_detection: 1 if malware_detection != '0' else 0)
    df['ashula_detection'] = df['ashula_detection'].apply(lambda ashula_detection: 1 if ashula_detection != '0' else 0)
    df['ids_detection'] = df['ids_detection'].apply(lambda ids_detection: 1 if ids_detection != '0' else 0)

    # label indicates there is an attack if
    # it is either -1 or -2, otherwise 1
    # replace -1 & -2 with 1, and 1 with 0
    df['label'] = df['label'].apply(lambda label: 1 if label == -1 or label == -2 else 0)

    # convert time to continuous data
    df['start_time'] = df['start_time'].apply(lambda time: int(time.split(':')[0]) +
                                                           (int(time.split(':')[1]) * (1 / 60)) +
                                                           (int(time.split(':')[2]) * (1 / 3600)))

    # index categorical data to [0, n-1] where n is the number of categories per feature
    df[cols_to_index] = df[cols_to_index].apply(preprocessing.LabelEncoder().fit_transform)

    # standardize continuous and quasi-continuous features
    df[cols_to_std] = preprocessing.StandardScaler().fit_transform(df[cols_to_std])

    # split the dataframe to multiple CSV files
    for id, df_i in enumerate(np.array_split(df, num_chunks)):
        df_i.to_csv(path_or_buf=os.path.join(write_path, '{id}.csv'.format(id=id)), columns=col_names, header=None,
                    index=False)
        print('Saving CSV file : {path}'.format(path=os.path.join(write_path, '{id}'.format(id=id))))

    print('Done')


def list_files(path):
    """Returns the list of files present in the path"""
    file_list = []
    for (dir_path, dir_names, file_names) in walk(path):
        file_list.extend(os.path.join(dir_path, filename) for filename in file_names)
    return file_list


def parse_args():
    parser = argparse.ArgumentParser(
        description='Data normalization script for Kyoto University 2013 Network Traffic Data')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-d', '--dataset', required=True, type=str,
                       help='path of the dataset to be normalized')
    group.add_argument('-w', '--write_path', required=True, type=str,
                       help='path where to save the normalized dataset')
    group.add_argument('-n', '--num_chunks', required=True, type=int,
                       help='number of file splits for the dataset')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()

    normalize_data(args.dataset, args.write_path, args.num_chunks)
