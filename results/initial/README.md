## Initial Results

The proposed GRU+SVM neural network architecture and GRU+Softmax was trained to detect intrusions using the [Kyoto University 2013 honeypot systems dataset](http://www.takakura.com/Kyoto_data/). The following were the hyperparameters used for the initial experiment:

Hyperparameters|GRU+SVM|GRU+Softmax|
|--------------|------|-----------|
|BATCH_SIZE|256|256|
|CELL_SIZE|256|256|
|DROPOUT_RATE|0.85|0.8|
|EPOCHS|2|2|
|LEARNING RATE|1e-5|1e-6
|SVM_C|0.5|n/a|

Both models were trained using `tf.train.AdamOptimizer`, with total steps of `116064` (`14856316 mod 256 * EPOCHS`) since there are `14856316` lines of data in the preprocessed 25% of the Kyoto University dataset.

The following were the results of the training:

|Accuracy|GRU+SVM|GRU+Softmax|
|-----|-------|-----------|
|Training|Average of 93.29%|Average of 71.39%|
|Validation|Average of 80.53%|Average of 71.41%|

With a considerable gap between the training and validation accuracy of the proposed GRU+SVM architecture, I posit that the hyperparameters used were sub-optimal. Otherwise, it might have something to do with the preprocessing of the dataset. On the other hand, GRU+Softmax had a small gap between its training and validation accuracy, I suppose it is safe to assume that the hyperparameters are almost optimal. However, the problem lies in its low accuracy. Hence, my hypothesis of dataset having a problem with its preprocessing.
