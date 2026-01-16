#!/bin/bash

python pretrain_gnn.py \
    --n_users 14160 \
    --n_items 184 \
    --data_path "data/" \
    --graph_path "data/graphs/weighted/" \
    --save_model_path "data/models_gnn/" \
    --device "cpu" \
    --save_every 10 \
    --weight_decay 0 \
    --residual True \
    --weight_loss_alpha 0.5 \
    --gnn_name "GraphSAGE" \
    --num_layers 2 \
    --num_heads 4 \
    --output_size 48 \
    --embedding_dim 48 \
    --link_predictor_name "one_layer" \
    --link_predictor_add_sigmoid True \
    --use_edge_weights True \
    --weight_add_sigmoid True \
    --n_epochs 200 \
    --learning_rate 0.002 \
    --neg_items_per_pos 1