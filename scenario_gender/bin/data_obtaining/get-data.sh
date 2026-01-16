#!/usr/bin/env bash

mkdir data
cd data

curl -OL 'https://huggingface.co/datasets/dllllb/transactions-gender/resolve/main/gender_train.csv'
curl -OL 'https://huggingface.co/datasets/dllllb/transactions-gender/resolve/main/transactions.csv.gz'

gunzip -f *.csv.gz
