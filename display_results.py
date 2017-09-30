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

"""Implementation of the GRU+SVM model for Intrusion Detection"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1'
__author__ = 'Abien Fred Agarap'

from dataset.normalize_data import list_files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf

TRAINING_RESULTS_PATH = '/home/darth/GitHub Projects/gru_svm/results/gru_svm/training'
VALIDATION_RESULTS_PATH = '/home/darth/GitHub Projects/gru_svm/results/gru_svm/validation'


def view_results(phase, path):
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

    plt.imshow(conf, cmap='binary', interpolation='None')

    plt.show()

    # display the findings from the confusion matrix
    print('True negative : {}'.format(conf[0][0]))
    print('False negative : {}'.format(conf[1][0]))
    print('True positive : {}'.format(conf[1][1]))
    print('False positive : {}'.format(conf[0][1]))

    accuracy = (conf[0][0] + conf[1][1]) / results.shape[0]

    print('{} accuracy : {}'.format(phase, accuracy))


def main():
    view_results(phase='Training', path=TRAINING_RESULTS_PATH)
    view_results(phase='Validation', path=VALIDATION_RESULTS_PATH)


if __name__ == '__main__':
    main()
