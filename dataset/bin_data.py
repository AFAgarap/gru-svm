# Module for binning continuous data
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

"""Bins continuous data into 10 evenly-spaced intervals"""
import argparse
import numpy as np
import os
import pandas as pd
import normalize_data as nd

__version__ = "0.1"
__author__ = "Abien Fred Agarap"

column_names = nd.COLUMN_NAMES
columns_to_save = list(column_names)
columns_to_save.remove("dst_ip_add")
columns_to_save.remove("src_ip_add")
cols_to_std = nd.COLUMN_TO_STANDARDIZE
cols_to_std.append("service")
cols_to_std.append("flag")


def bin_data(path, write_path, num_chunks, binning):
    """Bins the continuous features through bucket or quantile binning

    Parameter
    ---------
    path : str
      The path where the dataset to be binned is located.
    write_path : str
      The path where to save the binned dataset.
    num_chunks : int
      The number of file splits to perform on the binned dataset.
    binning : int
      The type of binning to perform on the dataset: 0 if bucket binning, 1 if quantile binning.
    """

    # get the list of files found in PATH
    files = nd.list_files(path=path)

    df = pd.DataFrame()

    for file in files:
        # append the data from CSV files to the dataframe
        df = df.append(pd.read_csv(filepath_or_buffer=file, names=column_names))
        print("appending : {}".format(file))

    # remove dst_ip_add and src_ip_add features
    df = df.drop(labels=["dst_ip_add", "src_ip_add"], axis=1)

    for index in range(len(cols_to_std)):
        if int(binning) == 0:
            # bucket binning
            bins = np.linspace(
                df[cols_to_std[index]].min(), df[cols_to_std[index]].max(), 10
            )
            df[cols_to_std[index]] = np.digitize(
                df[cols_to_std[index]], bins, right=True
            )
            print(
                "min : {}, max : {}".format(
                    df[cols_to_std[index]].min(), df[cols_to_std[index]].max()
                )
            )

        if int(binning) == 1:
            # decile binning
            df[cols_to_std[index]] = pd.qcut(
                df[cols_to_std[index]], 10, labels=False, duplicates="drop"
            )
            print(
                "min : {}, max : {}".format(
                    df[cols_to_std[index]].min(), df[cols_to_std[index]].max()
                )
            )

    for id, df_i in enumerate(np.array_split(df, num_chunks)):
        # split and save the dataframe to CSV files
        df_i.to_csv(
            path_or_buf=os.path.join(write_path, "{id}.csv".format(id=id)),
            columns=columns_to_save,
            header=None,
            index=False,
        )
        print(
            "Saving CSV file : {path}".format(
                path=os.path.join(write_path, "{id}".format(id=id))
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Module for binning the Kyoto University 2013 dataset"
    )
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=str,
        help="path of the dataset to be binned",
    )
    group.add_argument(
        "-w",
        "--write_path",
        required=True,
        type=str,
        help="path where the binned dataset will be stored",
    )
    group.add_argument(
        "-n",
        "--num_chunks",
        required=True,
        type=int,
        help="number of chunks of CSV files to save",
    )
    group.add_argument(
        "-b",
        "--binning",
        action="store",
        help="set to 0 for bucket binning; set 1 for decile binning",
    )
    arguments = parser.parse_args()
    return arguments


def main(arguments):
    bin_data(
        arguments.dataset, arguments.write_path, arguments.num_chunks, arguments.binning
    )


if __name__ == "__main__":
    args = parse_args()

    main(args)
