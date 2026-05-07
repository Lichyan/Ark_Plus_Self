#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="/home/pcl/data/lxx/mimic_multi_stage_v2_with_ehr_added"
PRETRAIN="./Ark6_swinLarge768_ep50.pth.tar"
EXP_SUFFIX="_mimic_tabonly_mlp_seed42"
CUDA_VISIBLE_DEVICES=0,1 python main_classification.py \
  --mode train --data_set advCheX_hyp_grade_stage_tab_only --data_dir "${DATA_DIR}" \
  --train_list "${DATA_DIR}/train_feat.csv" --val_list "${DATA_DIR}/valid_feat.csv" --test_list "${DATA_DIR}/test_feat.csv" \
  --model swin_large_384 --init ark_plus --pretrained_weights "${PRETRAIN}" --exp_name "${EXP_SUFFIX}" \
  --epochs 50 --batch_size 128 --opt adamw --lr 5e-4 --weight-decay 1e-4 --patience 8 --freeze_encoder False \
  --ordinal_mode CORN --tab_dim 5 --tab_hidden_dim 32 --tab_out_dim 64 --task_hidden_dim 128 --dropout_tab 0.1 \
  --lambda_cond 0.0 --lambda_joint_soft 0.0 --decodermode non --decoder_objective qwk --skip_test True --workers 4
