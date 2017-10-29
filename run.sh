#!/bin/bash
python3 gru_svm_main.py --train_dataset dataset/train/train_data.npy \
--validation_dataset dataset/test/test_data.npy \
--checkpoint_path models/checkpoint/gru_svm \
--model_name gru_svm.ckpt \
--log_path models/logs/gru_svm \
--result_path results/gru_svm