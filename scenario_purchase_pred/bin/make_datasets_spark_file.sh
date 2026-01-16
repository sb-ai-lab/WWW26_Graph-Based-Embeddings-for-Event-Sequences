#!/usr/bin/env bash

export PYTHONPATH="../../"
SPARK_LOCAL_IP="127.0.0.1" spark-submit \
    --master local[8] \
    --name "Purchase Make Dataset" \
    --driver-memory 8G \
    --conf spark.sql.shuffle.partitions=100 \
    --conf spark.sql.parquet.compression.codec="snappy" \
    --conf spark.ui.port=4041 \
    --conf spark.local.dir="data/.spark_local_dir" \
    make_dataset.py \
    --data_path data/mbd_dataset \
    --trx_files trx_train.parquet trx_test.parquet \
    --col_client_id "client_id" \
    --cols_event_time "#datetime" "event_time" \
    --cols_category "event_type" "event_subtype" "currency" "src_type11" "src_type12" "dst_type11" "dst_type12" "src_type21" "src_type22" "src_type31" "src_type32" \
    --cols_log_norm "amount" \
    --test_size 0.1 \
    --output_train_path "data/train_trx_file.parquet" \
    --output_test_path "data/test_trx_file.parquet" \
    --output_test_ids_path "data/test_ids_file.csv" \
    --log_file "results/dataset_purchase_pred_file.txt"
    # --target_files train_target.parquet test_target_b.parquet \
    # --col_target TODO:Fill me \
    
    # !!!! the target is sequential. It would make sense to add it to the trx files and add the tgt columns to cols_category
    # but in this case we have to check that for each timestamp from the target file we have the corresponding trx in the trx file.
    # Aslo tgt values have to be unchanged (category values are mapped to new integers; we have to avoid this for tgt values)
    # If tgt values have different timestamps (which is likely) we have to add target columns with target sequences  and also 
    # a column with a sequence of timestamps for target values.

# 654 sec with    --print_dataset_info
# 144 sec without --print_dataset_info
