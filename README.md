A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and Support Vector Machine (SVM) for Intrusion Detection
===

![](https://img.shields.io/badge/DOI-cs.NE%2F1709.03082-blue.svg)
[![AUR](https://img.shields.io/aur/license/yaourt.svg)]()
[![JIRA sprint completion](https://img.shields.io/badge/completion-90-orange.svg)]()

The full paper on this project may be read at the following sites: [ResearchGate](https://goo.gl/muZP5A), [arXiv.org](https://arxiv.org/abs/1709.03082), [Academia.edu](https://goo.gl/8aBXpX).

## Abstract
Gated Recurrent Unit (GRU) is a recently-developed variation of the long short-term memory (LSTM) unit, both of which
are types of recurrent neural network (RNN). Through empirical evidence, both models have been proven to be effective
in a wide variety of machine learning tasks such as natural language processing (Wen et al., 2015), speech
recognition (Chorowski et al., 2015), and text classification (Yang et al., 2016). Conventionally, like most
neural networks, both of the aforementioned RNN variants employ the Softmax function as its final output layer for its
prediction, and the cross-entropy function for computing its loss. In this paper, we present an amendment to this norm
by introducing linear support vector machine (SVM) as the replacement for Softmax in the final output layer of a GRU 
model. Furthermore, the cross-entropy function shall be replaced with a margin-based function. While there have been
similar studies (Alalshekmubarak & Smith, 2013; Tang, 2013), this proposal is primarily intended for binary
classification on intrusion detection using the 2013 network traffic data from the honeypot systems of Kyoto University.
Results show that the GRU-SVM model performs relatively higher than the conventional GRU-Softmax model. The proposed
model reached a training accuracy of ~81.54% and a testing accuracy of ~84.15%, while the latter was able to reach a
training accuracy of ~63.07% and a testing accuracy of ~70.75%. In addition, the juxtaposition of these two final output
layers indicate that the SVM would outperform Softmax in prediction time - a theoretical implication which was supported
by the actual training and testing time in the study.

## Citation
```
@article{agarap2017neural,
  title={A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and Support Vector Machine (SVM) for Intrusion Detection in Network Traffic Data},
  author={Agarap, Abien Fred},
  journal={arXiv preprint arXiv:1709.03082},
  year={2017}
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
