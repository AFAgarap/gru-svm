# A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and
# Support Vector Machine (SVM) for Intrusion Detection in Network Traffic Data
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

"""Displays the summary of results"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.2.0'
__author__ = 'Abien Fred Agarap'

import argparse
from dataset.normalize_data import list_files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf


def view_results(phase, path, class_names):
    files = list_files(path=path)

    df = pd.DataFrame()

    for file in files:
        df = df.append(pd.read_csv(filepath_or_buffer=file, header=None))

    print('Done appending CSV files.')

    results = np.array(df)

    predictions = results[:, :2]

    actual = results[:, 2:]

    with tf.Session() as sess:
        predictions = sess.run(tf.argmax(predictions, 1))
        actual = sess.run(tf.argmax(actual, 1))

    conf = confusion_matrix(y_true=actual, y_pred=predictions)

    plt.imshow(conf, cmap=plt.cm.Purples, interpolation='nearest')
    plt.title('Confusion Matrix for {} Phase'.format(phase))

    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.show()

    # display the findings from the confusion matrix
    print('True negative : {}'.format(conf[0][0]))
    print('False negative : {}'.format(conf[1][0]))
    print('True positive : {}'.format(conf[1][1]))
    print('False positive : {}'.format(conf[0][1]))

    accuracy = (conf[0][0] + conf[1][1]) / results.shape[0]

    print('{} accuracy : {}'.format(phase, accuracy))


def parse_args():
    parser = argparse.ArgumentParser(description='Confusion Matrix for Intrusion Detection')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-t', '--training_results_path', required=True, type=str,
                       help='path where the results of model training are stored')
    group.add_argument('-v', '--validation_results_path', required=True, type=str,
                       help='path where the results of model validation are stored')
    arguments = parser.parse_args()
    return arguments


def main(argv):
    view_results(phase='Training', path=argv.training_results_path, class_names=['normal', 'under attack'])
    view_results(phase='Validation', path=argv.validation_results_path, class_names=['normal', 'under attack'])


if __name__ == '__main__':
    args = parse_args()

    main(args)
