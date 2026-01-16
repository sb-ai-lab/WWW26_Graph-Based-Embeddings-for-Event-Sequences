#!/bin/bash

# CHECKPOINTS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149)
CHECKPOINTS=($(seq 9 10 149)) 


PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'

MAX_EPOCHES=150


declare -A model_epoch_map

# If an epoch from range(1,201) was not used it will cause an error, but will just move on tho the next one until a proper value is met.
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.0__emb_size_32"]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200"
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.0__emb_size_128"]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200"
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.0__emb_size_64"]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200"



for model_dir in "${!model_epoch_map[@]}"; do
  IFS=' ' read -r -a EPOCHES <<< "${model_epoch_map[$model_dir]}"

  for pretrain_epoch in "${EPOCHES[@]}"; do

    f_name="${pretrain_epoch}.pt"
    embeddings_path="${PRETRAINED_MODELS_ROOT}/${model_dir}/${f_name}"

    experiment_name_without_max_epochs="coles_gnn__pretrained_${model_dir}__pretrain_epoches_${pretrain_epoch}"
    full_experiment_name="${experiment_name_without_max_epochs}__epoches_${MAX_EPOCHES}"


    for checkpoint_n in "${CHECKPOINTS[@]}"; do
      N_ECPOCHES_FROM_ONE=$((checkpoint_n+1))
      MODEL_NAME="${experiment_name_without_max_epochs}__epoches_${N_ECPOCHES_FROM_ONE}"
      MODEL_PATH="models/${MODEL_NAME}.p"

      PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
        --config-dir conf --config-name coles_gnn_pretrained_params \
        data_module.train_data.splitter.split_count=2 \
        data_module.valid_data.splitter.split_count=2 \
        pl_module.validation_metric.K=1 \
        pl_module.lr_scheduler_partial.step_size=60 \
        model_path=${MODEL_PATH} \
        logger_name="${experiment_name_without_max_epochs}" \
        data_module.train_batch_size=64 \
        data_module.train_num_workers=4 \
        data_module.valid_batch_size=64 \
        data_module.valid_num_workers=4 \
        \
        pl_module.seq_encoder.trx_encoder.custom_embeddings.mcc_code.embeddings.f="${embeddings_path}" \
        \
        +ckpt_path="./checkpoints/${full_experiment_name}/epoch\=${checkpoint_n}.ckpt" \
        # device="cpu" \


      # Collect embeddings
      PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
          model_path=${MODEL_PATH} \
          embed_file_name="${MODEL_NAME}_embeddings" \
          inference.batch_size=256 \
          --config-dir conf --config-name coles_gnn_pretrained_params
          # +inference.devices=0 \
    done
  done
done
