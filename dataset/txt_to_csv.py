# Converts the Kyoto University 2013 network traffic data from TXT to CSV
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

"""
Converts the original Kyoto University dataset from
 Text files to CSV files
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.1"
__author__ = "Abien Fred Agarap"

import argparse
import csv
from normalize_data import list_files
import os
from os import walk


def convert_txt_to_csv(txt_path, csv_path):
    """Converts the Kyoto University dataset TXT files to CSV files

    Parameter
    ---------
    txt_path : str
      The path where the TXT files are located.
    csv_path : str
      The path where to save the CSV-converted files.
    """

    # list to store the filenames under the subdirectories of the <path>
    data = list_files(path=txt_path)

    csv_data = []  # list to store the converted CSV files

    # Create the <csv_path> if it does not exist
    os.makedirs(csv_path) if not os.path.exists(csv_path) else print(
        "CSV folder exists"
    )

    for month in range(12):
        """ Create the subdirectories under the <csv_path> if it does not exist """
        if next(walk(csv_path))[1].__len__() == 12:
            print("Folders exist")
            break
        print("Creating subdirectories.")
        # get the dirpath from the generator object <walk> (index 0)
        # then joins the dirpath with the month number
        os.makedirs(
            os.path.join(
                next(walk(csv_path))[0],
                "0" + str(month + 1) if month < 9 else str(month + 1),
            )
        )

    for index in range(len(data)):
        """ Store the processed CSV filename to <csv_data> list """
        csv_data.append(
            os.path.join(csv_path, data[index].split(csv_path)[1].replace("txt", "csv"))
        )

    for index in range(len(data)):
        """ Reading the text files delimited with tab, and converts it to CSV """
        try:
            print("Processing: {}".format(data[index]))
            in_csv = csv.reader(open(data[index], "r"), delimiter="\t")
            out_csv = csv.writer(open(csv_data[index], "x"))
            out_csv.writerows(in_csv)
        except FileNotFoundError:
            print("File not found: {}".format(data[index]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Module for converting the Kyoto University 2013 honeypot system dataset TXT to CSV"
    )
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-t",
        "--txt_path",
        required=True,
        type=str,
        help="path of the dataset in TXT format",
    )
    group.add_argument(
        "-c",
        "--csv_path",
        required=True,
        type=str,
        help="path where the dataset in CSV format will be stored",
    )
    arguments = parser.parse_args()
    return arguments


def main(arguments):
    convert_txt_to_csv(arguments.txt_path, arguments.csv_path)


if __name__ == "__main__":
    args = parse_args()

    main(arguments=args)
