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

"""An implementation of the L2-SVM class for Intrusion Detection"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.2"
__author__ = "Abien Fred Agarap"

import argparse
from models.svm.svm import Svm

# Hyper-parameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-5
N_CLASSES = 2
SEQUENCE_LENGTH = 21


def parse_args():
    parser = argparse.ArgumentParser(description="SVM for Intrusion Detection")
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-o",
        "--operation",
        required=True,
        type=str,
        help='the operation to perform: "train" or "test"',
    )
    group.add_argument(
        "-t",
        "--train_dataset",
        required=False,
        type=str,
        help="the NumPy array training dataset (*.npy) to be used",
    )
    group.add_argument(
        "-v",
        "--validation_dataset",
        required=True,
        type=str,
        help="the NumPy array validation dataset (*.npy) to be used",
    )
    group.add_argument(
        "-c",
        "--checkpoint_path",
        required=True,
        type=str,
        help="path where to save the trained model",
    )
    group.add_argument(
        "-l",
        "--log_path",
        required=False,
        type=str,
        help="path where to save the TensorBoard logs",
    )
    group.add_argument(
        "-m",
        "--model_name",
        required=False,
        type=str,
        help="filename for the trained model",
    )
    group.add_argument(
        "-r",
        "--result_path",
        required=True,
        type=str,
        help="path where to save the actual and predicted labels",
    )
    arguments = parser.parse_args()
    return arguments


def main(arguments):

    if arguments.operation == "train":
        train_features, train_labels = data.load_data(dataset=arguments.train_dataset)
        validation_features, validation_labels = data.load_data(
            dataset=arguments.validation_dataset
        )

        train_size = train_features.shape[0]
        validation_size = validation_features.shape[0]

        model = Svm(
            alpha=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            svm_c=arguments.svm_c,
            num_classes=N_CLASSES,
            num_features=SEQUENCE_LENGTH,
        )

        model.train(
            checkpoint_path=arguments.checkpoint_path,
            log_path=arguments.log_path,
            model_name=arguments.model_name,
            epochs=arguments.num_epochs,
            result_path=arguments.result_path,
            train_data=[train_features, train_labels],
            train_size=train_size,
            validation_data=[validation_features, validation_labels],
            validation_size=validation_size,
        )
    elif arguments.operation == "test":
        test_features, test_labels = data.load_data(
            dataset=arguments.validation_dataset
        )

        test_size = test_features.shape[0]

        test_features = test_features[: test_size - (test_size % BATCH_SIZE)]
        test_labels = test_labels[: test_size - (test_size % BATCH_SIZE)]

        test_size = test_features.shape[0]

        Svm.predict(
            batch_size=BATCH_SIZE,
            num_classes=N_CLASSES,
            test_data=[test_features, test_labels],
            test_size=test_size,
            checkpoint_path=arguments.checkpoint_path,
            result_path=arguments.result_path,
        )


if __name__ == "__main__":
    args = parse_args()

    main(args)
