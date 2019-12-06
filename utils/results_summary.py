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

__version__ = "0.3.0"
__author__ = "Abien Fred Agarap"

import argparse
from utils.data import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Confusion Matrix for Intrusion Detection"
    )
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-t",
        "--training_results_path",
        required=True,
        type=str,
        help="path where the results of model training are stored",
    )
    group.add_argument(
        "-v",
        "--validation_results_path",
        required=True,
        type=str,
        help="path where the results of model validation are stored",
    )
    arguments = parser.parse_args()
    return arguments


def main(argv):
    training_confusion_matrix = plot_confusion_matrix(
        phase="Training",
        path=argv.training_results_path,
        class_names=["normal", "under attack"],
    )
    validation_confusion_matrix = plot_confusion_matrix(
        phase="Validation",
        path=argv.validation_results_path,
        class_names=["normal", "under attack"],
    )
    # display the findings from the confusion matrix
    print("True negative : {}".format(training_confusion_matrix[0][0][0]))
    print("False negative : {}".format(training_confusion_matrix[0][1][0]))
    print("True positive : {}".format(training_confusion_matrix[0][1][1]))
    print("False positive : {}".format(training_confusion_matrix[0][0][1]))
    print("training accuracy : {}".format(training_confusion_matrix[1]))

    # display the findings from the confusion matrix
    print("True negative : {}".format(validation_confusion_matrix[0][0][0]))
    print("False negative : {}".format(validation_confusion_matrix[0][1][0]))
    print("True positive : {}".format(validation_confusion_matrix[0][1][1]))
    print("False positive : {}".format(validation_confusion_matrix[0][0][1]))
    print("validation accuracy : {}".format(validation_confusion_matrix[1]))


if __name__ == "__main__":
    args = parse_args()

    main(args)
