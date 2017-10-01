A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and Support Vector Machine (SVM) for Intrusion Detection
===

![](https://img.shields.io/badge/DOI-cs.NE%2F1709.03082-blue.svg)
[![AUR](https://img.shields.io/aur/license/yaourt.svg)]()
[![JIRA sprint completion](https://img.shields.io/badge/completion-90-orange.svg)]()

The full paper on this project may be read at the following sites: [ResearchGate](https://goo.gl/muZP5A), [arXiv.org](https://arxiv.org/abs/1709.03082), [Academia.edu](https://goo.gl/8aBXpX).

## Abstract
Recurrent neural networks (RNNs) is a kind of machine learning algorithm used for processing data that are sequential in nature. Hence, it is the most-commonly used neural network architecture for natural language processing, speech recognition, and sequence (such as words, sentences, even logs) classification among others. Conventionally, like other neural networks, RNNs implement the softmax activation function, which outputs a probability distribution over its target classes, as its last layer in a classification task. In this paper, I present a modification on the Gated Recurrent Unit (GRU) variant of RNN, where a Support Vector Machine (SVM) will be used instead of softmax as the last layer in a classification task. This proposed architecture was specifically intended for binary (non-probabilistic) classification. The choice of SVM over softmax is of paramount importance due to its computational efficiency. An SVM has a constant time complexity (O(1)) for its predictor function, while a softmax has a linear time complexity (O(n)), since it must satisfy a probability distribution for its prediction. For this study, the proposed architecture and the conventional architecture were trained to detect intrusions in a network traffic data, using the 2013 iteration of the Kyoto University honeypot systems dataset. The said task is a binary classification problem, i.e. to detect whether there is an attack or none. It was found out that not only was the GRU-SVM model had higher training and validation accuracy than its comparator, but also, it was faster in terms of training time.

## Citation
```
@online{agarap2017/online,
	author={Abien Fred Agarap}
	title={A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and Support Vector Machine (SVM) for Intrusion Detection in Network Traffic Data},
	date={2017-09-10},
	year={2017},
	eprintclass={cs.NE},
	eprinttype={arXiV},
	eprint={cs.NE/1709.03082},
}
```

## Usage

First, clone this repository:

```
~$ git clone https://github.com/AFAgarap/gru-svm.git/
```

Then, use the sample data for training the proposed GRU+SVM:

```
~$ cd gru-svm/model/
~/gru-svm/model$ python3 gru_svm.py --train_dataset "/home/gru-svm/dataset/train" \
--validation_dataset "/home/gru-svm/dataset/test" \
--checkpoint_path "/home/gru-svm/model/checkpoint/" \
--log_path "/home/gru-svm/model/logs/" \
--model_name "gru_svm.ckpt"
```

After training, use the trained model for the classifier:

```
~/gru-svm/model/$ python3 gru_svm_classifier.py --dataset "/home/gru-svm/dataset/test" \
--model "/home/gru-svm/model/checkpoint/" \
--result "gru+svm_results.csv"

```

Sample output of the classifier may be found [here](results/gru_svm_results.txt).

## License

	A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and
	Support Vector Machine (SVM) for Intrusion Detection in Network Traffic Data
	Copyright (C) 2017  Abien Fred Agarap

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU Affero General Public License as published
	by the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU Affero General Public License for more details.

	You should have received a copy of the GNU Affero General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
