#!/usr/bin/env bash
set -euo pipefail
MODEL_TYPE="${1:-lr}"
ROOT="Finetuning/result_tmp/baseline_comparison/clinical_only_${MODEL_TYPE}"
python Finetuning/baselines/train_clinical_only_ml.py \
  --train_csv /home/pcl/data/lxx/mimic_multi_stage_v2_with_ehr_added/train_feat.csv \
  --valid_csv /home/pcl/data/lxx/mimic_multi_stage_v2_with_ehr_added/valid_feat.csv \
  --test_csv /home/pcl/data/lxx/mimic_multi_stage_v2_with_ehr_added/test_feat.csv \
  --external_csvs \
    /home/pcl/data/lxx/advCheX_Hyp_multi_grade_stage_feat_handan_testonly/test_feat.csv \
    /home/pcl/data/lxx/advCheX_Hyp_multi_grade_stage_feat_hebei_testonly/test_feat.csv \
    /home/pcl/data/lxx/advCheX_Hyp_multi_grade_stage_feat_hfirstall_testonly/test_feat.csv \
    /home/pcl/data/lxx/advCheX_Hyp_multi_grade_stage_feat_hfirstall_have_bmi_testonly/test_feat.csv \
    /home/pcl/data/lxx/advCheX_Hyp_multi_grade_stage_feat_handan_have_bmi_testonly/test_feat.csv \
  --external_names handan hebei hfirstALL hfirstALL_have_bmi handan_have_bmi \
  --model_type "${MODEL_TYPE}" --out_root "${ROOT}"
