Intrusion Detection using Support Vector Machine (SVM) as Classifier in a Deep Recurrent Neural Network (RNN)
===

The full paper on this proposal may be read at [ResearchGate](https://goo.gl/muZP5A)

## Abstract
Gated Recurrent Unit (GRU) is a recently published variant of the Long Short-Term Memory (LSTM) network, designed to solve the vanishing gradient and exploding gradient problems. However, its main objective is to solve the long-term dependency problem in Recurrent Neural Networks (RNNs), which prevents the network to connect an information from previous iteration with the current iteration. This study proposes a modification on the GRU model, having Support Vector Machine (SVM) as its classifier instead of the Softmax function. The classifier is responsible for the output of a network in a classification problem. SVM was chosen over Softmax for its efficiency in learning. The proposed model will then be used for intrusion detection, with the dataset from Kyoto University's honeypot system in 2013 which will serve as both its training and testing data.

## Citation
```
@article{afagarap2017grusvm,
	author={Agarap, Abien Fred},
	title={Intrusion Detection using Support Vector Machine (SVM) as Classifier in a Deep Recurrent Neural Network (RNN)},
	school={Adamson University},
	year={2017},
	note={unpublished thesis}
}
```

## Introduction
The annual cost to the global economy due to cybercrime could be as high as $575 billion, which includes both the gain to criminals and the costs to companies for defense and recovery <a href="https://goo.gl/CHFpgF">(Center for Strategic and International Studies, 2014)</a>. It is even projected that the said cost will reach $2 trillion by 2019 (<a href="https://www.juniperresearch.com/press/press-releases/cybercrime-cost-businesses-over-2trillion">Juniper, 2015</a>; <a href="https://www.forbes.com/sites/stevemorgan/2016/01/17/cyber-crime-costs-projected-to-reach-2-trillion-by-2019">Morgan, 2016</a>).

![](figures/optimal-hyperplane.png)

Figure 1: Image from <a href="http://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html">(OpenCV Dev Team, 2017)</a>. The SVM algorithm outputs a hyperplane which categorizes the data, usually into two classes.

![](figures/ann.jpg)

Figure 2: Image from <a href="https://www.youtube.com/watch?v=OmJ-4B-mS-Y">(Domain of Science, 2017)</a>, showing a neural network that classifies an image as "cat". The image data is converted to a series of numbers, which then gets processed by the hidden layers (violet circles in the image), and outputs the probability of to which class does the image belong.
<!-- ![](figures/gru.png) from [Chris Olah's Blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
![](figures/data.png)
![](figures/svm.png) -->