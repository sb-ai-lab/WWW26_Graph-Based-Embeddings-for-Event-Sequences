#!/bin/bash

python -m ptls.pl_train_module --config-dir conf --config-name barlow_twins_params \
    +ckpt_path="./checkpoints/${full_experiment_name}/epoch\=${checkpoint_n}.ckpt"

python -m ptls.pl_inference --config-dir conf --config-name barlow_twins_params
