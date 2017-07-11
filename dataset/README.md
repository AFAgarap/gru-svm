2013 Dataset from Kyoto University
===

### About ###

The [traffic data](http://www.takakura.com/Kyoto_data/ext_old_data201704/) to be used for the training and evaluation of the proposed GRU-SVM model is from the honeypot systems of Kyoto University. You may read the document describing the data features [here](http://www.takakura.com/Kyoto_data/BenchmarkData-Description-v5.pdf).

### Data Pre-processing ###

For the GRU-SVM computational model to utilize the dataset, it must be normalized first. That is, to scale or index non-integer values to a format which can be read by the neural network. Based from [this study](http://scholarworks.rit.edu/cgi/viewcontent.cgi?article=9241&context=theses), the pre-processing will be as follows:

* Split the dataset into multiple files, separated which pertains to (1) logs with an attack detected, and (2) logs with no attack detected,
* Map symbolic features like `service`, `flag`, and `protocol` to [0, n-1] where n is the number of symbols in a feature.
* Linear scaling to [0.0, 1.0] of integer values like `duration`, `src_bytes`, `dest_bytes`, `dst_host_count`, `dst_host_srv_count`, and `start_time`.
* Boolean values and continuous values shall be left as is.