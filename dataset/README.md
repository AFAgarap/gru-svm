2013 Dataset from Kyoto University
===

### About ###

The [network traffic data](http://www.takakura.com/Kyoto_data/ext_old_data201704/) to be used for the training and
evaluation of the proposed GRU-SVM model is from the honeypot systems of Kyoto University. You may read the document
describing the data features [here](http://www.takakura.com/Kyoto_data/BenchmarkData-Description-v5.pdf).

The dataset contains logs for 360 days of the year 2013. Only the logs for the following dates are non-existing:
(1) March 2-4, and (2) October 13-14 -- totalling to 5 days.

Only the 25% of the whole 16.1 GB dataset was used for the experiment, which was further partitioned to (1) 80% for
training dataset, and (2) 20% for testing dataset. The following is the class distribution of the said partition:

|Class|Training data|Testing data|
|-----|-------------|------------|
|Normal|794,512|157,914|
|Intrusion detected|1,103,728|262,694|

A total of 1,898,240 instances of training data; a total of 420,608 instances of testing data. The specified [training
dataset](https://github.com/AFAgarap/gru-svm/tree/master/dataset/train) and
[testing dataset](https://github.com/AFAgarap/gru-svm/tree/master/dataset/test) are available in this repository.

### Data Pre-processing ###

For the computational models in this study to utilize the dataset, it must be normalized first. That is, to standardize
continuous features, and to index categorical features.
The pre-processing done in this study were as follows:

* Map categorical features like `service`, `flag`, and `protocol` to `[0, n-1]` where n is the number of symbols in a
feature.
* Standardize continuous features like `duration`, `src_bytes`, `dest_bytes`, `dst_host_count`, `dst_host_srv_count`,
and `start_time` among others.
* After the preceding processes, the normalized data was binned using decile binning (for one-hot encoding purposes).

### Script Parameters

The following are the parameters for each script used in pre-processing the dataset.

* For converting the raw TXT files of Kyoto University dataset to CSV files.
```buildoutcfg
usage: txt_to_csv.py [-h] -t TXT_PATH -c CSV_PATH

Module for converting the Kyoto University 2013 honeypot system dataset TXT to
CSV

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -t TXT_PATH, --txt_path TXT_PATH
                        path of the dataset in TXT format
  -c CSV_PATH, --csv_path CSV_PATH
                        path where the dataset in CSV format will be stored
```

* For normalization of the dataset.
```buildoutcfg
usage: normalize_data.py [-h] -d DATASET -w WRITE_PATH -n NUM_CHUNKS

Data normalization script for Kyoto University 2013 Network Traffic Data

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -d DATASET, --dataset DATASET
                        path of the dataset to be normalized
  -w WRITE_PATH, --write_path WRITE_PATH
                        path where to save the normalized dataset
  -n NUM_CHUNKS, --num_chunks NUM_CHUNKS
                        number of file splits for the dataset
```

* For binning (discretization / quantization) of continuous features in the dataset.
```buildoutcfg
usage: bin_data.py [-h] -d DATASET -w WRITE_PATH -n NUM_CHUNKS [-b BINNING]

Module for binning the Kyoto University 2013 dataset

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -d DATASET, --dataset DATASET
                        path of the dataset to be binned
  -w WRITE_PATH, --write_path WRITE_PATH
                        path where the binned dataset will be stored
  -n NUM_CHUNKS, --num_chunks NUM_CHUNKS
                        number of chunks of CSV files to save
  -b BINNING, --binning BINNING
                        set to 0 for bucket binning; set 1 for decile binning
```

* For converting the binned CSV files to NPY files.
```buildoutcfg
usage: csv_to_npy.py [-h] -c CSV_PATH -n NPY_PATH -f NPY_FILENAME

Module for converting CSV to NPY files

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -c CSV_PATH, --csv_path CSV_PATH
                        path of the CSV files to be converted
  -n NPY_PATH, --npy_path NPY_PATH
                        path where converted NPY files will be stored
  -f NPY_FILENAME, --npy_filename NPY_FILENAME
                        filename of the NPY file to save
```

### Usage

First, convert the raw dataset TXT files to CSV files using `txt_to_csv.py`:
```buildoutcfg
cd gru-svm/dataset
python3 txt_to_csv.py --txt_path gru-svm/dataset/raw/train --csv_path gru-svm/dataset/csv/train
python3 txt_to_csv.py --txt_path gru-svm/dataset/raw/test --csv_path gru-svm/dataset/csv/test
```

After converting the TXT files to CSV files, the dataset is ready for normalization. Use the `normalize_data.py` to do
so.
```buildoutcfg
python3 normalize_data.py --dataset gru-svm/dataset/csv/train --write_path gru-svm/dataset/train --num_chunks 24
python3 normalize_data.py --dataset gru-svm/dataset/csv/test --write_path gru-svm/dataset/test --num_chunks 24
```

After normalization, perform quantile binning on the dataset. Therefore preparing the dataset for one-hot encoding.
```buildoutcfg
python3 bin_data.py --dataset gru-svm/dataset/train --write_path gru-svm/dataset/train/binned --num_chunks 24 --binning 1
python3 bin_data.py --dataset gru-svm/dataset/test --write_path gru-svm/dataset/test/binned --num_chunks 24 --binning 1
```

Instead of using the TensorFlow Queues for feeding data from CSV files, NumPy arrays are saved from the loaded CSV
files. In other words, the CSV files are converted to NPY files.
```buildoutcfg
python3 csv_to_npy.py --csv_path gru-svm/dataset/train --npy_path gru-svm/dataset/train --npy_filename train.npy
python3 csv_to_npy.py --csv_path gru-svm/dataset/test --npy_path gru-svm/dataset/test --npy_filename test.npy
```

The sub-directories specified in the sample module usages are only hypothetical; you may have different sub-directories
from these. Lastly, as the dataset is too large (i.e. 16.1 GB when uncompressed), it cannot be uploaded in this GitHub
repository. So, you may download the dataset from the
[official website of Kyoto University network traffic data](http://www.takakura.com/Kyoto_data/).
