# Module for converting CSV files to NPY files
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

"""Converts CSV files to NPY files"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

import argparse
from normalize_data import list_files
import numpy as np
import os
import pandas as pd


def csv_to_npy(csv_path, npy_path, npy_filename):
    files = list_files(path=csv_path)

    df = pd.DataFrame()

    for file in files:
        df = df.append(pd.read_csv(filepath_or_buffer=file, header=None))
        print("Appending file : {}".format(file))

    df = df.drop_duplicates(subset=df, keep="first", inplace=False)

    data = np.array(df)

    np.save(file=os.path.join(npy_path, npy_filename), arr=data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Module for converting CSV to NPY files"
    )
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-c",
        "--csv_path",
        required=True,
        type=str,
        help="path of the CSV files to be converted",
    )
    group.add_argument(
        "-n",
        "--npy_path",
        required=True,
        type=str,
        help="path where converted NPY files will be stored",
    )
    group.add_argument(
        "-f",
        "--npy_filename",
        required=True,
        type=str,
        help="filename of the NPY file to save",
    )
    arguments = parser.parse_args()
    return arguments


def main(arguments):
    csv_to_npy(arguments.csv_path, arguments.npy_path, arguments.npy_filename)


if __name__ == "__main__":
    args = parse_args()

    main(args)
