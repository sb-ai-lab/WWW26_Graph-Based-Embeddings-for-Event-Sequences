#!/usr/bin/env bash

mkdir data
cd data

curl -OL 'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_train.csv.gz'
curl -OL 'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_test.csv.gz'
curl -OL 'https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/train_target.csv'

gunzip -f *.csv.gz
