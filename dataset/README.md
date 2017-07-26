2013 Dataset from Kyoto University
===

### About ###

The [traffic data](http://www.takakura.com/Kyoto_data/ext_old_data201704/) to be used for the training and evaluation of the proposed GRU-SVM model is from the honeypot systems of Kyoto University. You may read the document describing the data features [here](http://www.takakura.com/Kyoto_data/BenchmarkData-Description-v5.pdf).

### Data Pre-processing ###

For the GRU-SVM computational model to utilize the dataset, it must be normalized first. That is, to scale or index non-integer values to a format which can be read by the neural network. Based from [this study](http://scholarworks.rit.edu/cgi/viewcontent.cgi?article=9241&context=theses), the pre-processing will be as follows:

* Split the dataset into multiple files, separated which pertains to (1) logs with detected attack(s), and (2) logs with no detected attack(s).
* Map symbolic features like `service`, `flag`, and `protocol` to `[0, n-1]` where n is the number of symbols in a feature.
* Linear scaling to `[0.0, 1.0]` of integer values like `duration`, `src_bytes`, `dest_bytes`, `dst_host_count`, `dst_host_srv_count`, and `start_time`.
* Boolean values and continuous values shall be left as is.

Data normalization was done using [standardize_data.py](https://github.com/AFAgarap/gru-svm/blob/master/dataset/standardize_data.py), while splitting of the dataset into two: (1) logs with detected attack(s), and (2) logs with no detected attack(s) was done using [categorize_data.py](https://github.com/AFAgarap/gru-svm/blob/master/dataset/categorize_data.py). To check if the data pre-processing was successful, the normalization and categorization that is, the module [check_standardized_data.py](https://github.com/AFAgarap/gru-svm/blob/master/dataset/check_standardized_data.py) was used.

Sample data with detected attack(s):

```
0 : example [ -3.66881303e-02   3.00000000e+00  -2.20432188e-02  -2.52959970e-03
  -4.15846437e-01   8.17247152e-01   3.45117354e+00   1.87133086e+00
  -9.11360681e-01   1.38428557e+00  -1.90726683e-01  -4.77430880e-01
   2.23967052e+00   6.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   3.08720000e+04  -9.29602027e-01   3.06000000e+02
   7.55433226e+00   1.35430086e+00   2.00000000e+00], label 1.0
1 : example [ -4.13453244e-02   3.00000000e+00  -2.64402162e-02  -2.52959970e-03
  -6.23901546e-01  -1.28207588e+00  -3.01768869e-01   1.87133086e+00
   4.71016735e-01   6.56612664e-02  -7.57413357e-02   2.05189323e+00
   1.96527612e+00   6.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   4.45600000e+04  -1.17409921e+00   2.12500000e+03
  -7.04813078e-02   1.35434210e+00   1.00000000e+00], label 1.0
2 : example [ -4.13453244e-02   3.00000000e+00  -2.64402162e-02  -2.52959970e-03
  -6.23901546e-01  -1.28207588e+00  -3.01768869e-01   1.36253881e+00
  -9.11360681e-01  -1.30678451e+00  -1.90726683e-01  -4.77430880e-01
  -5.04273891e-01   6.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   1.58530000e+04   5.96221209e-01   2.83000000e+03
   1.88292623e+00   1.35434210e+00   1.00000000e+00], label 1.0
```

Sample data with no detected attack(s):

```
0 : example [ -2.43421942e-02   9.00000000e+00   3.89906368e-03  -5.28765901e-04
   6.24429166e-01   4.60362226e-01  -3.01768869e-01  -6.72629297e-01
   8.50492895e-01   4.42411065e-01  -1.90726683e-01  -4.77430880e-01
  -5.04273891e-01   1.00000000e+01   0.00000000e+00   0.00000000e+00
   0.00000000e+00   3.27220000e+04   1.38504553e+00   6.64000000e+02
  -1.78705454e-01   2.04480633e-01   1.00000000e+00], label 0.0
1 : example [ -2.57230941e-02   9.00000000e+00   4.68075182e-03  -5.28765901e-04
   2.08318934e-01   5.65328360e-01  -3.01768869e-01  -6.72629297e-01
   8.50492895e-01   4.42411065e-01  -1.90726683e-01  -4.77430880e-01
  -5.04273891e-01   1.00000000e+01   0.00000000e+00   0.00000000e+00
   0.00000000e+00   3.27220000e+04   1.41305792e+00   6.64000000e+02
  -1.78705454e-01   2.04521850e-01   1.00000000e+00], label 0.0
2 : example [ -2.53630150e-02   9.00000000e+00   3.89906368e-03  -5.28765901e-04
   3.12346488e-01   5.86321592e-01  -3.01768869e-01  -6.72629297e-01
   8.50492895e-01   4.42411065e-01  -1.90726683e-01  -4.77430880e-01
  -5.04273891e-01   1.00000000e+01   0.00000000e+00   0.00000000e+00
   0.00000000e+00   3.27220000e+04   1.41508842e+00   6.64000000e+02
  -1.78705454e-01   2.04521850e-01   1.00000000e+00], label 0.0
```

You may notice the labels for each state: `1` when there is a detected attack, and `0` when there is no detected attack. The labels were converted from `-1` and `-2` for detected attack(s) to `1`, and from `1` for no detected attack(s) to `0`. This was done in [standardize_data.py line #60](https://github.com/AFAgarap/gru-svm/blob/master/dataset/standardize_data.py#L60):

```
df['label'] = df['label'].apply(lambda label : 1 if label == -1 or label == -2 else 0)
```

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


After normalizing the dataset, a sample data was described again using [pandas-describe.py](https://github.com/AFAgarap/gru-svm/blob/master/dataset/pandas-describe.py). It can be noticed that the `mean` and `std` of the data is `0` or approaching zero and `1` or rounded off to 1 respectively, for the continuous and quasi-continuous data. The same cannot be said of course on categorical (symbolic or boolean or mixed) data: indices `[1], [13-16], [18], [20], [23]`.

```
           duration       service     src_bytes    dest_bytes         count  \
count  3.420651e+06  3.420651e+06  3.420651e+06  3.420651e+06  3.420651e+06   
mean  -8.760009e-03  3.822125e+00 -1.714331e-04  4.372183e-04  8.966806e-02   
std    1.131989e+00  2.172168e+00  1.122048e+00  1.143075e+00  1.126134e+00   
min   -2.696939e-02  0.000000e+00 -1.250530e-03 -1.799158e-03 -3.709263e-01   
25%   -2.696939e-02  4.000000e+00 -1.250530e-03 -1.799158e-03 -3.709263e-01   
50%   -2.525046e-02  4.000000e+00 -1.250530e-03 -1.799158e-03 -3.709263e-01   
75%   -5.513818e-03  4.000000e+00 -1.221449e-03 -1.767468e-03 -2.564268e-01   
max    6.255615e+02  1.100000e+01  1.516127e+03  1.091797e+03  5.354051e+00   

       same_srv_rate   serror_rate  srv_serror_rate  dst_host_count  \
count   3.420651e+06  3.420651e+06     3.420651e+06    3.420651e+06   
mean   -4.465839e-02  1.028700e-01     2.239037e-01    1.720746e-01   
std     1.006911e+00  1.119093e+00     9.952793e-01    1.078151e+00   
min    -9.068619e-01 -3.467402e-01    -9.202415e-01   -6.255217e-01   
25%    -9.068619e-01 -3.467402e-01    -9.202415e-01   -6.255217e-01   
50%    -9.068619e-01 -3.467402e-01     2.608652e-01   -1.613522e-01   
75%     1.132909e+00 -3.467402e-01     1.441972e+00    3.337620e-01   
max     1.132909e+00  3.018346e+00     1.441972e+00    2.468942e+00   

       dst_host_srv_count      ...       ids_detection  malware_detection  \
count        3.420651e+06      ...        3.420651e+06       3.420651e+06   
mean        -2.347640e-01      ...        3.414525e-02       2.975165e-03   
std          9.571986e-01      ...        1.816022e-01       5.446388e-02   
min         -1.172377e+00      ...        0.000000e+00       0.000000e+00   
25%         -1.172377e+00      ...        0.000000e+00       0.000000e+00   
50%         -4.223133e-01      ...        0.000000e+00       0.000000e+00   
75%          3.277501e-01      ...        0.000000e+00       0.000000e+00   
max          1.414049e+00      ...        1.000000e+00       1.000000e+00   

       ashula_detection      label    src_ip_add  src_port_num    dst_ip_add  \
count      3.420651e+06  3420651.0  3.420651e+06  3.420651e+06  3.420651e+06   
mean       8.052561e-03        1.0  1.277138e+05 -2.451679e-01  9.988296e+03   
std        8.937404e-02        0.0  6.964637e+04  9.086642e-01  4.442578e+03   
min        0.000000e+00        1.0  0.000000e+00 -1.015630e+00  0.000000e+00   
25%        0.000000e+00        1.0  7.146300e+04 -8.924692e-01  6.733000e+03   
50%        0.000000e+00        1.0  1.295630e+05 -8.015208e-01  9.721000e+03   
75%        0.000000e+00        1.0  1.850620e+05  4.588618e-01  1.479600e+04   
max        1.000000e+00        1.0  2.474770e+05  1.991585e+00  1.668800e+04   

       dst_port_num    start_time      protocol  
count  3.420651e+06  3.420651e+06  3.420651e+06  
mean   6.384958e-02 -2.523717e-02  1.144730e+00  
std    1.131659e+00  1.007345e+00  4.056236e-01  
min   -2.180595e-01 -1.644528e+00  0.000000e+00  
25%   -2.089779e-01 -9.366630e-01  1.000000e+00  
50%   -1.418084e-01 -6.922149e-02  1.000000e+00  
75%   -1.418084e-01  8.811915e-01  1.000000e+00  
max    1.101124e+01  1.729711e+00  2.000000e+00
```