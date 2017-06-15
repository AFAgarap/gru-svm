## The Proposed GRU-SVM Model
Similar to the work done by <a href="http://ieeexplore.ieee.org/abstract/document/6544391/">Alalshekmubarak & Smith, 2013</a>, the present study proposes to use SVM as the classification function in an RNN. The difference being instead of ESN, the RNN class to be used in the study is the GRU model (see <a href='figures/gru.png'>Figure 4</a>).

![](../figures/gru.png)

Figure 4: Image from [Chris Olah's Blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). The GRU model combines the "forget" gate and "input" gate into a single "update" gate, making it simpler than the LSTM model (see [LSTM model](figures/rnn-lstm.png)).
<!-- ![](figures/data.png)
![](figures/svm.png) -->