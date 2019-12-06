# Module for getting a dataset description
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

"""Module for getting the dataset description based on sample data"""
import os
import pandas as pd
import normalize_data

# a sample data to describe using pandas
PATH = "/home/darth/GitHub Projects/gru_svm/dataset/train/4/attack"

# get the column names from normalize_data.py module
COL_NAMES = normalize_data.COLUMN_NAMES

# list to contain the filenames
files = []

# instantiate a dataframe
df = pd.DataFrame()

# append the filenames to the list
for (dirpath, dirnames, filenames) in os.walk(PATH):
    files.extend(os.path.join(dirpath, filename) for filename in filenames)

# append the data from files to df
for file in files:
    df = df.append(
        pd.read_csv(filepath_or_buffer=file, names=COL_NAMES, engine="python")
    )

# print the data summary
print(df.describe())
