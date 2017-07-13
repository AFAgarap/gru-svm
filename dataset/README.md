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

### Data Summary ###

The data summary using `df.describe()` of `pandas` is an aid for deciding which values in the dataset must be standardized. The inquiry I wrote regarding it may be found [here](https://stats.stackexchange.com/questions/291081/standardization-or-normalization), while the script I used for getting the data summary may be found [here](pandas-describe.py). The data summary is as follows:

```

                 0             2             3             4             5   \
count  4.656124e+06  4.656124e+06  4.656124e+06  4.656124e+06  4.656124e+06   
mean   5.629711e+00  7.620859e+03  8.519262e+03  1.819146e+00  2.604244e-01   
std    1.747110e+02  3.446973e+06  3.346247e+06  8.298907e+00  4.228922e-01   
min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
25%    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
50%    2.866777e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
75%    3.419435e+00  1.280000e+02  2.090000e+02  1.000000e+00  5.000000e-01   
max    8.097288e+04  2.133443e+09  2.116371e+09  1.000000e+02  1.000000e+00   

                 6             7             8             9             10  \
count  4.656124e+06  4.656124e+06  4.656124e+06  4.656124e+06  4.656124e+06   
mean   5.863585e-02  4.008782e-01  1.056284e+01  2.819324e+01  3.256177e-02   
std    2.289534e-01  4.238607e-01  2.233580e+01  2.824031e+01  1.718919e-01   
min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
25%    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
50%    0.000000e+00  3.300000e-01  0.000000e+00  3.000000e+01  0.000000e+00   
75%    0.000000e+00  9.500000e-01  5.000000e+00  5.000000e+01  0.000000e+00   
max    1.000000e+00  1.000000e+00  1.000000e+02  1.000000e+02  1.000000e+00   

                 11            12            17            19            21  \
count  4.656124e+06  4.656124e+06  4.656124e+06  4.656124e+06  4.656124e+06   
mean   1.405558e-01  2.122936e-01 -3.552655e-01  2.292569e+04  1.648552e+03   
std    3.280353e-01  3.864984e-01  9.440645e-01  2.250753e+04  6.820971e+03   
min    0.000000e+00  0.000000e+00 -2.000000e+00  0.000000e+00  0.000000e+00   
25%    0.000000e+00  0.000000e+00 -1.000000e+00  3.028000e+03  2.500000e+01   
50%    0.000000e+00  0.000000e+00 -1.000000e+00  6.000000e+03  8.000000e+01   
75%    0.000000e+00  0.000000e+00  1.000000e+00  4.522200e+04  4.450000e+02   
max    1.000000e+00  1.000000e+00  1.000000e+00  6.553500e+04  6.553500e+04   

                 22  
count  4.656124e+06  
mean   1.216933e+01  
std    7.080260e+00  
min    0.000000e+00  
25%    5.916667e+00  
50%    1.228806e+01  
75%    1.844000e+01  
max    2.399944e+01

```

The values that belong to the missing indices `[1], [13-16], [18], [20], [23]` are either symbolic features or boolean feature or mixed.