#!/usr/bin/env bash

mkdir data
cd data
mkdir original_format_data
cd original_format_data

curl -OL 'This/Url/Is/Removed/Until/The/Review/Process/Finishes'
curl -OL 'This/Url/Is/Removed/Until/The/Review/Process/Finishes'

gunzip -f *.pqt.gz
tar -xzf *.tar.gz
rm *.tar.gz


# # Loop through all .pqt files in the current directory
# for file in *.pqt; do
#   # Check if any .pqt files exist
#   if [ -e "$file" ]; then
#     # Rename the file to .parquet
#     mv -- "$file" "${file%.pqt}.parquet"
#   fi
# done

mv competition_data_final_pqt competition_data_final.parquet
mv public_train.pqt public_train.parquet