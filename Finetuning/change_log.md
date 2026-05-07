# Change Log

## 2026-05-06 baseline comparison extension

- Added three new PyTorch baseline dataset/model routes:
  - `advCheX_hyp_grade_stage_tab_only`
  - `advCheX_hyp_grade_stage_imgemb_only`
  - `advCheX_hyp_grade_stage_simple_concat_fusion`
- Added model classes:
  - `TabularOnlyOrdinalModel`
  - `EmbeddingOnlyOrdinalModel`
  - `SimpleConcatFusionOrdinalModel`
- Extended embtab dataset to support `load_img_emb=False` for tabular-only speed-up.
- Extended engine multi-head evaluation whitelist to include the new baselines so existing decoder/result pipeline remains reusable.
- Added clinical traditional ML baseline script:
  - `Finetuning/baselines/train_clinical_only_ml.py`
- Added baseline scripts:
  - `Finetuning/scripts/baseline_comparison/train_tabular_only_mlp.sh`
  - `Finetuning/scripts/baseline_comparison/test_baseline_multi_cohort.sh`
  - `Finetuning/scripts/baseline_comparison/run_clinical_ml.sh`

Compatibility note:
- Existing modes (`embtab-base`, `embtab-v2lite`, `v1`, `sep_v1`) are untouched in their original dataset names, model classes, and script interface.

## 2026-05-07 baseline output completeness update

- Added dedicated training scripts for all new neural baselines:
  - `train_tabular_only_mlp.sh`
  - `train_image_only_mlp.sh`
  - `train_simple_concat_fusion_mlp.sh`
- Added dedicated auto-test scripts for each new mode:
  - `test_tabular_only_mlp_auto.sh`
  - `test_image_only_mlp_auto.sh`
  - `test_simple_concat_fusion_mlp_auto.sh`
  - `test_clinical_ml_auto.sh`
- Upgraded `test_baseline_multi_cohort.sh` to copy full artifacts (`png/json/csv/txt`) and generate `run_info.txt` per cohort/decoder.
- Enhanced clinical ML baseline outputs to include:
  - AUROC/QWK metrics in `metrics.json` and `result.txt`
  - `Confmat_grade.png`, `Confmat_stage.png`, `invalid_type_hist.png`
  - `predictions.csv`
  - structured invalid-type count section in `result.txt`
- Added LightGBM option with automatic fallback to HGB if LightGBM is unavailable.
