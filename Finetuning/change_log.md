# Change Log

## 2026-05-07 quick bugfix (multi-head dict plain branch)

- Fixed `test_classification`/`test_model` missing plain dict multi-head route for outputs with only `grade_logits` + `stage_ind_logits`.
- Added fail-fast guard when multi-head targets exist but `p_grade/p_stage` were not collected.
- This resolves tab-only test-time evaluation input assembly failures and keeps existing embtab/v1/sep/v2 routes unchanged.

## 2026-05-07 ordinal evaluation stability + fairness hardening (ABCD)

- Reworked ordinal shape handling to avoid silent semantic drift:
  - Added `_validate_ordinal_k(...)` to enforce expected ordinal channel count.
  - `evaluate_grade_stage_sep` now validates:
    - `y_grade` as `[N,3]`
    - `y_stage` as `[N,2]`
    - `p_ge_grade` as `[N,3]`
    - `p_ge_stage` as `[N,2]`
- Hardened `ordinal_probs_to_class_probs(...)`:
  - Explicitly supports only `K=2` (stage) and `K=3` (grade).
  - Raises clear errors for unexpected `K` instead of falling through and indexing invalid channels.
- Added test-time aggregation safety in `trainer.test_classification(...)`:
  - New `_ensure_2d_batch_tensor(...)` ensures per-batch multi-head outputs remain 2D before concat.
  - Applied to all multi-head test branches (coarse-fine dict, v2 dict, tuple branch), reducing squeeze-related runtime instability for:
    - tab-only MLP
    - image-only projector MLP
    - simple-concat fusion MLP
    - while preserving existing embtab/v1/v2/sep code paths unless shape is invalid.

Compatibility note:
- This is a strict validation + robustness update; old modes are not rerouted and their training/testing interfaces stay unchanged.

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
