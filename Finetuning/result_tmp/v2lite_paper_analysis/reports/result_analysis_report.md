# embtab-v2lite 结果分析报告

基于自动解析表格进行初步分析。

## 1) decoder 选择分析

internal上优先比较QWK/MAE/invalid_rate，三种decoder各有trade-off；建议主decoder以internal稳定性与raw->final增益为主，不基于外部集cherry-pick。

## 2) grandBatch128 vs jointsoft_00

jointsoft_00用于检验去掉joint CE后中间类是否改善。总体以middle-class recall与QWK/MAE联合评估：若middle recall提升但QWK下降，需权衡。

## 3) decoder final vs fused

fused通常invalid_rate趋近0，但可能带来middle-class collapse；建议将fused作为joint-consistency辅助分析而非唯一主输出。

## 4) multicenter external heterogeneity

外部队列存在明显中心差异，应分cohort报告QWK/AUROC与错误模式。

## 5) BMI completeness sensitivity

handan/hfirstALL在have_bmi子集的变化可作为敏感性分析，不宜过度因果解读。

## 6) q1/q2 conditional heads

即便AUC_cond较好，若composer分配造成中间类概率被挤压，joint11/joint21 recall仍可能偏低。

## 7) 初步论文展示建议

主文建议展示internal+多中心外部的QWK/MAE/AUROC与middle/extreme recall；补充材料展示jointsoft_00 ablation与fused-vs-final对照。
