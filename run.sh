#!/bin/bash
figlet "GRU-SVM"
printf "Operation: [1] Train [2] Test\n>> "
read choice
if [ "$choice" -eq "1" ] &>/dev/null; then
	echo "Training GRU-SVM for intrusion detection"
	if [ ! -d "./models/checkpoint" ] &>/dev/null; then
		mkdir models/checkpoint_path
	fi
	python3 gru_svm_main.py --operation "train" \
	--train_dataset dataset/train/train_data.npy \
	--validation_dataset dataset/test/test_data.npy \
	--checkpoint_path models/checkpoint/gru_svm \
	--model_name gru_svm.ckpt \
	--log_path models/logs/gru_svm \
	--result_path results/gru_svm
elif [ "$choice" -eq "2" ] &>/dev/null; then
	if [ -d "./models/checkpoint/gru_svm" ] &>/dev/null; then
		echo "Testing GRU-SVM for intrusion detection"
		python3 gru_svm_main.py --operation "test" \
		--validation_dataset dataset/test/test_data.npy \
		--checkpoint_path models/checkpoint/gru_svm \
		--result_path results/gru_svm
	else
		echo "Train the model first!"
		exit
	fi
else
	echo "Invalid input"
	exit
fi