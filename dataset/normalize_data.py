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

__version__ = "0.2.1"
__author__ = "Abien Fred Agarap"

import argparse
import numpy as np
import pandas as pd
import os
from os import walk
from sklearn import preprocessing

# column names of 24 features
COLUMN_NAMES = [
    "duration",
    "service",
    "src_bytes",
    "dest_bytes",
    "count",
    "same_srv_rate",
    "serror_rate",
    "srv_serror_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_src_port_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "flag",
    "ids_detection",
    "malware_detection",
    "ashula_detection",
    "label",
    "src_ip_add",
    "src_port_num",
    "dst_ip_add",
    "dst_port_num",
    "start_time",
    "protocol",
]

# column names of continuous and quasi-continuous features
COLUMN_TO_STANDARDIZE = [
    "duration",
    "src_bytes",
    "dest_bytes",
    "count",
    "same_srv_rate",
    "serror_rate",
    "srv_serror_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_src_port_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "src_port_num",
    "dst_port_num",
    "start_time",
]

# column names of categorical data
COLUMN_TO_INDEX = [
    "ashula_detection",
    "dst_ip_add",
    "flag",
    "ids_detection",
    "label",
    "malware_detection",
    "protocol",
    "service",
    "src_ip_add",
]


def normalize_data(path):
    """Normalizes a given dataset.

    Parameter
    ---------
    path : str
      The path of the dataset to be normalized.
    write_path : str
      The path where to save the normalized dataset.
    num_chunks : int
      The number of file splits for the normalized dataset.

    Returns
    -------
    dataframe : pandas.core.frame.DataFrame
      A Pandas dataframe containing the normalized dataset.

    Example
    -------
    >>> PATH = '/home/data'
    >>> normalize_data(PATH)
    Appending /home/darth/Desktop/data/sample-data.csv
    Current DataFrame shape: (3, 24)
    DataFrame shape after NaN values removal: (3, 24)
       duration  service  src_bytes  dest_bytes  count  same_srv_rate  \
    0 -0.784854        0  -0.707107   -0.707107    0.0            0.0
    1 -0.626398        0  -0.707107   -0.707107    0.0            0.0
    2  1.411251        1   1.414214    1.414214    0.0            0.0

       serror_rate  srv_serror_rate  dst_host_count  dst_host_srv_count    ...     \
    0          0.0        -0.707107        1.414214           -0.687558    ...
    1          0.0         1.414214       -0.707107           -0.726477    ...
    2          0.0        -0.707107       -0.707107            1.414035    ...

       ids_detection  malware_detection  ashula_detection  label  src_ip_add  \
    0              0                  0                 0      1           2
    1              0                  0                 0      1           0
    2              0                  0                 0      0           1

       src_port_num  dst_ip_add  dst_port_num  start_time  protocol
    0      1.391614           0      0.707107         0.0         0
    1     -0.913883           1      0.707107         0.0         0
    2     -0.477731           2     -1.414214         0.0         0

    [3 rows x 24 columns]
    """

    # get all the CSV files in the PATH dir
    files = list_files(path=path)

    # create empty df, where dfs shall be appended
    df = pd.DataFrame()

    # append the dfs from each file to the data df
    for file in files:
        # the python engine was used to support mixed data types
        df = df.append(
            pd.read_csv(filepath_or_buffer=file, names=COLUMN_NAMES, engine="python")
        )
        print("Appending {}".format(file))

    print("Current DataFrame shape: {}".format(df.shape))

    # drop rows with NaN values
    df[COLUMN_NAMES] = df[COLUMN_NAMES].dropna(axis=0, how="any")
    print("DataFrame shape after NaN values removal: {}".format(df.shape))

    # since malware_detection, ashula_detection,
    # and ids_detection col contains string data
    # replace if the string != '0' with int 1
    # otherwise with int 0
    df["malware_detection"] = df["malware_detection"].apply(
        lambda malware_detection: 1 if malware_detection != "0" else 0
    )
    df["ashula_detection"] = df["ashula_detection"].apply(
        lambda ashula_detection: 1 if ashula_detection != "0" else 0
    )
    df["ids_detection"] = df["ids_detection"].apply(
        lambda ids_detection: 1 if ids_detection != "0" else 0
    )

    # label indicates there is an attack if
    # it is either -1 or -2, otherwise 1
    # replace -1 & -2 with 1, and 1 with 0
    df["label"] = df["label"].apply(
        lambda label: 1 if label == -1 or label == -2 else 0
    )

    # convert time to continuous data
    df["start_time"] = df["start_time"].apply(
        lambda time: int(time.split(":")[0])
        + (int(time.split(":")[1]) * (1 / 60))
        + (int(time.split(":")[2]) * (1 / 3600))
    )

    # index categorical data to [0, n-1] where n is the number of categories per feature
    df[COLUMN_TO_INDEX] = df[COLUMN_TO_INDEX].apply(
        preprocessing.LabelEncoder().fit_transform
    )

    # standardize continuous and quasi-continuous features
    df[COLUMN_TO_STANDARDIZE] = preprocessing.StandardScaler().fit_transform(
        df[COLUMN_TO_STANDARDIZE]
    )

    return df


def save_dataframe(dataframe, write_path, num_chunks):
    """Saves the given pandas dataframe to N-number of CSV files.

    Parameter
    ---------
    dataframe : pandas.core.frame.DataFrame
      A Pandas dataframe containing the normalized dataset.
    num_chunks : int
      The number of file splits for the normalized dataset.
    """
    for id, df_i in enumerate(np.array_split(dataframe, num_chunks)):
        df_i.to_csv(
            path_or_buf=os.path.join(write_path, "{id}.csv".format(id=id)),
            columns=COLUMN_NAMES,
            header=None,
            index=False,
        )
        print(
            "Saving CSV file : {path}".format(
                path=os.path.join(write_path, "{id}".format(id=id))
            )
        )


def list_files(path):
    """Returns a list of files

    Parameter
    ---------
    path : str
      A string consisting of a path containing files.

    Returns
    -------
    file_list : list
      A list of the files present in the given directory

    Examples
    --------
    >>> PATH = '/home/data'
    >>> list_files(PATH)
    >>> ['/home/data/file1', '/home/data/file2', '/home/data/file3']
    """

    file_list = []
    for (dir_path, dir_names, file_names) in walk(path):
        file_list.extend(os.path.join(dir_path, filename) for filename in file_names)
    return file_list


def parse_args():
    """Returns user-defined argument values."""
    parser = argparse.ArgumentParser(
        description="Data normalization script for Kyoto University 2013 Network Traffic Data"
    )
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=str,
        help="path of the dataset to be normalized",
    )
    group.add_argument(
        "-w",
        "--write_path",
        required=True,
        type=str,
        help="path where to save the normalized dataset",
    )
    group.add_argument(
        "-n",
        "--num_chunks",
        required=True,
        type=int,
        help="number of file splits for the dataset",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()

    normalized_data = normalize_data(args.dataset)

    save_dataframe(
        dataframe=normalized_data,
        write_path=args.write_path,
        num_chunks=args.num_chunks,
    )
