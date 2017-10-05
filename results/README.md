## Results

All experiments in this study were conducted on a laptop computer with Intel Core(TM) i5-6300HQ CPU @ 2.30GHz x 4, 16GB
of DDR3 RAM, and NVIDIA GeForce GTX 960M 4GB DDR5 GPU. The hyper-parameters used for both the proposed and the
conventional models were assigned by hand, and through hyper-parameter optimization/tuning.

#### Hyper-parameters used in both neural networks
|Hyperparameters|GRU+SVM|GRU+Softmax|
|--------------|------|-----------|
|BATCH_SIZE|256|256|
|CELL_SIZE|256|256|
|DROPOUT_RATE|0.85|0.8|
|EPOCHS|2|2|
|LEARNING RATE|1e-5|1e-6
|SVM_C|0.5|n/a|

#### GRU+Softmax binary classification statistical measures

|Variable | Training results | Validation results|
|---------|------------------|-------------------|
|False positive|3017548|32255|
|False negative|487175|582105|
|True positive|5031465|731365|
|True negative|955012|757315|
Accuracy|63.073973786244097%|70.78705112598904%|


#### GRU+SVM binary classification statistical measures

|Variable | Training results | Validation results|
|---------|------------------|-------------------|
|False positive|889327|192635|
|False negative|862419|140535|
|True positive|4656221|1172935|
|True negative|3083233|596935|
Accuracy|81.54347184760621%|84.15769552647596%|
