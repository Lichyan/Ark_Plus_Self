#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"
DATASET_NAME="${1:?dataset_name}"; EXP_SUFFIX="${2:?exp_suffix}"; MODEL_NAME="${3:-swin_large_384}"; INIT_NAME="${4:-ark_plus}"
TRAIN_LIST="/home/pcl/data/lxx/mimic_multi_stage_v2_with_ehr_added/train_feat.csv"
VAL_LIST="/home/pcl/data/lxx/mimic_multi_stage_v2_with_ehr_added/valid_feat.csv"
VAL_DATA_DIR="/home/pcl/data/lxx/mimic_multi_stage_v2_with_ehr_added"
OUTPUTS_DIR="Outputs/Classification/${DATASET_NAME}"
declare -A COHORTS=(
  [internal]="/home/pcl/data/lxx/mimic_multi_stage_v2_with_ehr_added"
  [handan]="/home/pcl/data/lxx/advCheX_Hyp_multi_grade_stage_feat_handan_testonly"
  [hebei]="/home/pcl/data/lxx/advCheX_Hyp_multi_grade_stage_feat_hebei_testonly"
  [hfirstALL]="/home/pcl/data/lxx/advCheX_Hyp_multi_grade_stage_feat_hfirstall_testonly"
  [hfirstALL_have_bmi]="/home/pcl/data/lxx/advCheX_Hyp_multi_grade_stage_feat_hfirstall_have_bmi_testonly"
  [handan_have_bmi]="/home/pcl/data/lxx/advCheX_Hyp_multi_grade_stage_feat_handan_have_bmi_testonly"
)
for combo in non:qwk temp_ev:macro_f1 temp_threshold:composite; do
  mode="${combo%%:*}"; obj="${combo##*:}"
  for name in "${!COHORTS[@]}"; do
    data_dir="${COHORTS[$name]}"
    CUDA_VISIBLE_DEVICES=0 python main_classification.py --mode test --data_set "${DATASET_NAME}" --data_dir "${data_dir}" \
      --exp_name "${EXP_SUFFIX}" --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" --test_list "${data_dir}/test_feat.csv" \
      --val_data_dir "${VAL_DATA_DIR}" --model "${MODEL_NAME}" --init "${INIT_NAME}" --batch_size 64 --freeze_encoder False \
      --ordinal_mode CORN --decodermode "${mode}" --decoder_objective "${obj}" --decoder_bins 101 \
      --decoder_use_saved_thresholds False --decoder_save_debug True --decoder_keep_raw_metrics True \
      --temperature_init 1.0 --temperature_min 0.5 --temperature_max 5.0 --temperature_grid_size 91 \
      --workers 4 --return_path True

    save_dir="Finetuning/result_tmp/baseline_comparison/${DATASET_NAME}/decoder_${mode}_${obj}/${name}"; mkdir -p "${save_dir}"
    if [[ -d "${OUTPUTS_DIR}" ]]; then
      find "${OUTPUTS_DIR}" -maxdepth 1 -type f \( -name "*.png" -o -name "*.json" -o -name "*.csv" -o -name "*results.txt" -o -name "result.txt" \) -exec cp -f {} "${save_dir}/" \;
      [[ -f "${save_dir}/${EXP_SUFFIX}_results.txt" ]] && mv -f "${save_dir}/${EXP_SUFFIX}_results.txt" "${save_dir}/result.txt" || true
    fi
    cat > "${save_dir}/run_info.txt" <<EOF
mode=${mode}
objective=${obj}
dataset=${name}
data_set=${DATASET_NAME}
exp_name=${EXP_SUFFIX}
data_dir=${data_dir}
train_list=${TRAIN_LIST}
val_list=${VAL_LIST}
test_list=${data_dir}/test_feat.csv
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
EOF
  done
done
