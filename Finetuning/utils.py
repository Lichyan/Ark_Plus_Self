from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score, f1_score, matthews_corrcoef, recall_score, confusion_matrix, brier_score_loss
import torch
import numpy as np
import json
import os

class MetricLogger(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressLogger(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def read_from_csv(csv_file):
    arr = []
    lines = open(csv_file).readlines()
    for line in lines[1:]:
        row = line.strip().split(",")
        row = [float(v) for v in row]
        arr.append(row)
    return np.array(arr)

def get_classwise_mean_std(data):
    data = np.array(data)
    class_wise_mean, class_wise_std = [],[]
    _, n_class = data.shape
    for ic in range(n_class):
        class_wise_mean.append(np.mean(data[:,ic]))
        class_wise_std.append(np.std(data[:,ic]))
    return [class_wise_mean, class_wise_std]

def meanMCC(ground_truth, predictions):
    thresholds_all = []
    ap_scores = []
    for i in range(ground_truth.shape[1]):
        if np.any(ground_truth[:, i]):
            fpr, tpr, thresholds = roc_curve(ground_truth[:, i], predictions[:, i])
            youden_j = tpr - fpr
            optimal_threshold = thresholds[np.argmax(youden_j)]
            thresholds_all.append(optimal_threshold)
            binary_predictions = (predictions[:, i] > optimal_threshold).astype(int)
            ap = matthews_corrcoef(ground_truth[:, i], binary_predictions)
            ap_scores.append(ap)

    map_score = np.mean(ap_scores)
    print(thresholds_all)
    return map_score, ap_scores

def meanAP(ground_truth, predictions):
    # Compute mean Average Precision (mAP)
    ap_scores = []
    for i in range(ground_truth.shape[1]):
        if np.any(ground_truth[:, i]):
            ap = average_precision_score(ground_truth[:, i], predictions[:, i])
            ap_scores.append(ap)

    map_score = np.mean(ap_scores)
    return map_score, ap_scores

def meanAUC(ground_truth, predictions):
    # Compute mean Area Under the ROC Curve (mAUC)
    auc_scores = []
    for i in range(ground_truth.shape[1]):
        if np.any(ground_truth[:, i]):
            auc = roc_auc_score(ground_truth[:, i], predictions[:, i])
            auc_scores.append(auc)

    mauc_score = np.mean(auc_scores)
    return mauc_score, auc_scores
def meanF1(ground_truth, predictions):
    # Compute mean F1 score (mF1)
    f1_scores = []
    optimal_thresholds = []
    recall_scores = []
    for i in range(ground_truth.shape[1]):
        if np.any(ground_truth[:, i]):
            fpr, tpr, thresholds = roc_curve(ground_truth[:, i], predictions[:, i])
            youden_j = tpr - fpr
            optimal_threshold = thresholds[np.argmax(youden_j)]
            binary_predictions = (predictions[:, i] > optimal_threshold).astype(int)
            f1 = f1_score(ground_truth[:, i], binary_predictions)
            recall = recall_score(ground_truth[:, i], binary_predictions)
            f1_scores.append(f1)
            optimal_thresholds.append(optimal_threshold)
            recall_scores.append(recall)

    mf1_score = np.mean(f1_scores)
    return mf1_score, f1_scores, optimal_thresholds, recall_scores


def grade_to_ordinal_targets(grade: int, k: int = 3):
    """(0~k) -> k 个是否>=k 标签"""
    return [1 if grade >= idx else 0 for idx in range(1, k + 1)]


def ordinal_targets_to_grade(row):
    """[>=1, >=2(, >=3)] -> 离散等级"""
    row = [int(v) for v in row]
    if len(row) == 2:
        if row == [0, 0]:
            return 0
        if row == [1, 0]:
            return 1
        return 2
    if row == [0, 0, 0]:
        return 0
    if row == [1, 0, 0]:
        return 1
    if row == [1, 1, 0]:
        return 2
    return 3


def decode_ordinal_probs(p_ge, thresholds=None):
    """按照默认或传入阈值把概率解码成等级"""
    p_ge = np.asarray(p_ge)
    k = p_ge.shape[1]
    if k == 2:
        thresholds = thresholds or {"ge1": 0.5, "ge2": 0.5}
        ge1, ge2 = thresholds.get("ge1", 0.5), thresholds.get("ge2", 0.5)
        preds = []
        for p1, p2 in p_ge:
            if p1 < ge1:
                preds.append(0)
            elif p2 < ge2:
                preds.append(1)
            else:
                preds.append(2)
        return preds

    thresholds = thresholds or {"ge1": 0.5, "ge2": 0.5, "ge3": 0.5}
    ge1, ge2, ge3 = thresholds.get("ge1", 0.5), thresholds.get("ge2", 0.5), thresholds.get("ge3", 0.5)
    preds = []
    for p1, p2, p3 in p_ge:
        if p1 < ge1:
            preds.append(0)
        elif p2 < ge2:
            preds.append(1)
        elif p3 < ge3:
            preds.append(2)
        else:
            preds.append(3)
    return preds


def compute_threshold_by_metric(y_true, scores, metric="youden"):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    if metric == "youden":
        youden_j = tpr - fpr
        return thresholds[np.argmax(youden_j)]
    elif metric == "f1":
        best_f1, best_t = -1, 0.5
        for thr in thresholds:
            pred = (scores >= thr).astype(int)
            f1 = f1_score(y_true, pred)
            if f1 > best_f1:
                best_f1, best_t = f1, thr
        return best_t
    return 0.5


def compute_ordinal_thresholds(y_ord, p_ge):
    y_ord = np.asarray(y_ord)
    p_ge = np.asarray(p_ge)
    k = y_ord.shape[1]
    thresholds = {"youden": {}, "f1": {}}
    for idx in range(k):
        key = f"ge{idx + 1}"
        y_true = y_ord[:, idx]
        thresholds["youden"][key] = compute_threshold_by_metric(y_true, p_ge[:, idx], metric="youden")
        thresholds["f1"][key] = compute_threshold_by_metric(y_true, p_ge[:, idx], metric="f1")
    return thresholds


def compute_stage2_thresholds(y_ord, p_ge, default=0.5):
    y_ord = np.asarray(y_ord)
    p_ge = np.asarray(p_ge)
    thresholds = {}
    thresholds["ge1"] = compute_threshold_by_metric(y_ord[:, 0], p_ge[:, 0], metric="youden")
    thresholds["ge2"] = compute_threshold_by_metric(y_ord[:, 1], p_ge[:, 1], metric="youden")

    grades = np.array([ordinal_targets_to_grade(row) for row in y_ord])
    p1 = np.clip(p_ge[:, 0] - p_ge[:, 1], 0, 1)
    p2 = np.clip(p_ge[:, 1], 0, 1)

    mask_stage1 = np.isin(grades, [0, 1])
    if mask_stage1.any() and (~mask_stage1).any():
        labels_stage1 = (grades[mask_stage1] == 1).astype(int)
        if len(np.unique(labels_stage1)) >= 2:
            thresholds["stage1_vs_non"] = compute_threshold_by_metric(labels_stage1, p1[mask_stage1], metric="youden")
        else:
            thresholds["stage1_vs_non"] = default
    else:
        thresholds["stage1_vs_non"] = default

    mask_stage2 = np.isin(grades, [0, 2])
    if mask_stage2.any() and (~mask_stage2).any():
        labels_stage2 = (grades[mask_stage2] == 2).astype(int)
        if len(np.unique(labels_stage2)) >= 2:
            thresholds["stage2_vs_non"] = compute_threshold_by_metric(labels_stage2, p2[mask_stage2], metric="youden")
        else:
            thresholds["stage2_vs_non"] = default
    else:
        thresholds["stage2_vs_non"] = default

    return thresholds


def build_ordinal_task_views(y_ord, p_ge):
    """返回各二分类任务的标签、分数和索引掩码"""
    y_ord = np.asarray(y_ord)
    p_ge = np.asarray(p_ge)
    grades = np.array([ordinal_targets_to_grade(row) for row in y_ord])
    k = p_ge.shape[1]
    idxs = np.arange(len(y_ord))

    if k == 2:
        p_stage1 = np.clip(p_ge[:, 0] - p_ge[:, 1], 0, 1)
        p_stage2 = np.clip(p_ge[:, 1], 0, 1)
        task_views = {
            "ge1": (y_ord[:, 0], p_ge[:, 0], idxs),
            "ge2": (y_ord[:, 1], p_ge[:, 1], idxs),
        }
        mask_stage1 = np.isin(grades, [0, 1])
        if mask_stage1.any() and (~mask_stage1).any():
            task_views["stage1_vs_non"] = (grades[mask_stage1] == 1, p_stage1[mask_stage1], np.nonzero(mask_stage1)[0])
        mask_stage2 = np.isin(grades, [0, 2])
        if mask_stage2.any() and (~mask_stage2).any():
            task_views["stage2_vs_non"] = (grades[mask_stage2] == 2, p_stage2[mask_stage2], np.nonzero(mask_stage2)[0])
        return task_views

    p_lv1 = np.clip(p_ge[:, 0] - p_ge[:, 1], 0, 1)
    p_lv2 = np.clip(p_ge[:, 1] - p_ge[:, 2], 0, 1)
    p_lv3 = np.clip(p_ge[:, 2], 0, 1)

    task_views = {
        "hasHTN": (y_ord[:, 0], p_ge[:, 0], idxs),
        "severe": (y_ord[:, 1], p_ge[:, 1], idxs),
        "very_severe": (y_ord[:, 2], p_ge[:, 2], idxs),
        "hypertension_vs_non": (grades >= 1, p_ge[:, 0], idxs),
    }

    mask_lv1 = np.isin(grades, [0, 1])
    if mask_lv1.any() and (~mask_lv1).any():
        task_views["lv1_vs_non"] = (grades[mask_lv1] == 1, p_lv1[mask_lv1], np.nonzero(mask_lv1)[0])
    mask_lv2 = np.isin(grades, [0, 2])
    if mask_lv2.any() and (~mask_lv2).any():
        task_views["lv2_vs_non"] = (grades[mask_lv2] == 2, p_lv2[mask_lv2], np.nonzero(mask_lv2)[0])
    mask_lv3 = np.isin(grades, [0, 3])
    if mask_lv3.any() and (~mask_lv3).any():
        task_views["lv3_vs_non"] = (grades[mask_lv3] == 3, p_lv3[mask_lv3], np.nonzero(mask_lv3)[0])

    return task_views


def compute_task_thresholds(task_views, metric="youden", default=0.5):
    thresholds = {}
    for name, (labels, scores, _) in task_views.items():
        thresholds[name] = compute_threshold_by_metric(labels, scores, metric=metric) if len(np.unique(labels)) >= 2 else default
    return thresholds


def binary_metrics_at_threshold(labels, scores, threshold):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    denom_rec = tp + fn
    denom_spec = tn + fp
    recall = tp / denom_rec if denom_rec > 0 else None
    spec = tn / denom_spec if denom_spec > 0 else None
    f1 = f1_score(labels, preds, zero_division=0)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "spec": spec,
        "f1": f1,
    }


def evaluate_tasks_with_thresholds(task_views, task_thresholds):
    metrics = {}
    for name, (labels, scores, _) in task_views.items():
        thr = task_thresholds.get(name, 0.5)
        metrics[name] = binary_metrics_at_threshold(labels, scores, thr)
    return metrics


def collect_confusion_examples(task_views, task_thresholds, paths):
    if paths is None:
        return {}
    examples = {}
    for name, (labels, scores, idxs) in task_views.items():
        thr = task_thresholds.get(name, 0.5)
        labels = np.asarray(labels).astype(int)
        scores = np.asarray(scores)
        preds = (scores >= thr).astype(int)
        candidates = {
            "TP": np.where((preds == 1) & (labels == 1))[0],
            "FP": np.where((preds == 1) & (labels == 0))[0],
            "TN": np.where((preds == 0) & (labels == 0))[0],
            "FN": np.where((preds == 0) & (labels == 1))[0],
        }
        row = {"task": name, "threshold": float(thr)}
        for tag, arr in candidates.items():
            if len(arr) > 0:
                global_idx = idxs[arr[0]]
                row[f"{tag}_path"] = paths[global_idx]
                row[f"{tag}_pred"] = int(preds[arr[0]])
                row[f"{tag}_gt"] = int(labels[arr[0]])
            else:
                row[f"{tag}_path"] = None
                row[f"{tag}_pred"] = None
                row[f"{tag}_gt"] = None
        examples.setdefault("rows", []).append(row)
    return examples


def ordinal_binary_task_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return None
    return roc_auc_score(labels, scores)


def evaluate_ordinal_tasks(y_ord, p_ge, thresholds=None):
    y_ord = np.asarray(y_ord)
    p_ge = np.asarray(p_ge)
    grades_true = [ordinal_targets_to_grade(row) for row in y_ord]
    k = p_ge.shape[1]

    metrics = {}
    if k == 2:
        metrics["AUROC_ge1"] = ordinal_binary_task_auc(y_ord[:, 0], p_ge[:, 0])
        metrics["AUROC_ge2"] = ordinal_binary_task_auc(y_ord[:, 1], p_ge[:, 1])
        p0 = 1 - p_ge[:, 0]
        p1 = np.clip(p_ge[:, 0] - p_ge[:, 1], 0, 1)
        p2 = np.clip(p_ge[:, 1], 0, 1)
        probs = np.stack([p0, p1, p2], axis=1)
        grade_pred = probs.argmax(axis=1).tolist()
        metrics["macro_f1"] = f1_score(grades_true, grade_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        metrics["macro_f1_stage3"] = metrics["macro_f1"]
        confmat = confusion_matrix(grades_true, grade_pred, labels=[0, 1, 2])
        metrics["confmat_stage3"] = confmat.astype(int).tolist()
        grades = np.array(grades_true)
        mask_midlow = np.isin(grades, [0, 1])
        if mask_midlow.any() and (~mask_midlow).any():
            labels_midlow = (grades[mask_midlow] == 1).astype(int)
            scores_midlow = p1[mask_midlow]
            if len(np.unique(labels_midlow)) >= 2:
                metrics["AUROC_midlow_vs_non"] = roc_auc_score(labels_midlow, scores_midlow)
                metrics["AUPRC_midlow_vs_non"] = average_precision_score(labels_midlow, scores_midlow)
            else:
                metrics["AUROC_midlow_vs_non"] = np.nan
                metrics["AUPRC_midlow_vs_non"] = np.nan
        else:
            metrics["AUROC_midlow_vs_non"] = np.nan
            metrics["AUPRC_midlow_vs_non"] = np.nan

        mask_high_midlow = np.isin(grades, [1, 2])
        if mask_high_midlow.any() and (~mask_high_midlow).any():
            labels_high_midlow = (grades[mask_high_midlow] == 2).astype(int)
            scores_high_midlow = p2[mask_high_midlow]
            if len(np.unique(labels_high_midlow)) >= 2:
                metrics["AUROC_high_vs_midlow"] = roc_auc_score(labels_high_midlow, scores_high_midlow)
                metrics["AUPRC_high_vs_midlow"] = average_precision_score(labels_high_midlow, scores_high_midlow)
            else:
                metrics["AUROC_high_vs_midlow"] = np.nan
                metrics["AUPRC_high_vs_midlow"] = np.nan
        else:
            metrics["AUROC_high_vs_midlow"] = np.nan
            metrics["AUPRC_high_vs_midlow"] = np.nan

        mask_high = np.isin(grades, [0, 2])
        if mask_high.any() and (~mask_high).any():
            labels_high = (grades[mask_high] == 2).astype(int)
            scores_high = p2[mask_high]
            if len(np.unique(labels_high)) >= 2:
                metrics["AUROC_high_vs_non"] = roc_auc_score(labels_high, scores_high)
                metrics["AUPRC_high_vs_non"] = average_precision_score(labels_high, scores_high)
            else:
                metrics["AUROC_high_vs_non"] = np.nan
                metrics["AUPRC_high_vs_non"] = np.nan
        else:
            metrics["AUROC_high_vs_non"] = np.nan
            metrics["AUPRC_high_vs_non"] = np.nan
        return metrics, grades_true, grade_pred

    # 基础三条任务
    metrics["AUROC_hasHTN"] = ordinal_binary_task_auc(y_ord[:, 0], p_ge[:, 0])
    metrics["AUROC_severe"] = ordinal_binary_task_auc(y_ord[:, 1], p_ge[:, 1])
    metrics["AUROC_very_severe"] = ordinal_binary_task_auc(y_ord[:, 2], p_ge[:, 2])

    # 分级概率（近似）
    p_lv1 = np.clip(p_ge[:, 0] - p_ge[:, 1], 0, 1)
    p_lv2 = np.clip(p_ge[:, 1] - p_ge[:, 2], 0, 1)
    p_lv3 = np.clip(p_ge[:, 2], 0, 1)

    grades = np.array(grades_true)
    mask_lv1 = np.isin(grades, [0, 1])
    if mask_lv1.any() and (~mask_lv1).any():
        metrics["AUROC_lv1_vs_non"] = ordinal_binary_task_auc(grades[mask_lv1], p_lv1[mask_lv1])
    else:
        metrics["AUROC_lv1_vs_non"] = None

    mask_lv2 = np.isin(grades, [0, 2])
    if mask_lv2.any() and (~mask_lv2).any():
        metrics["AUROC_lv2_vs_non"] = ordinal_binary_task_auc(grades[mask_lv2] == 2, p_lv2[mask_lv2])
    else:
        metrics["AUROC_lv2_vs_non"] = None

    mask_lv3 = np.isin(grades, [0, 3])
    if mask_lv3.any() and (~mask_lv3).any():
        metrics["AUROC_lv3_vs_non"] = ordinal_binary_task_auc(grades[mask_lv3] == 3, p_lv3[mask_lv3])
    else:
        metrics["AUROC_lv3_vs_non"] = None

    metrics["AUROC_hypertension_vs_non"] = ordinal_binary_task_auc(grades >= 1, p_ge[:, 0])

    grade_pred = decode_ordinal_probs(p_ge, thresholds)
    return metrics, grades_true, grade_pred


JOINT_LABELS = [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2)]
JOINT_LABEL_TO_INDEX = {pair: idx for idx, pair in enumerate(JOINT_LABELS)}


def safe_roc_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, scores)


def safe_auprc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return np.nan
    return average_precision_score(labels, scores)


def ordinal_logits_to_probs(logits):
    logits = np.asarray(logits)
    return 1.0 / (1.0 + np.exp(-logits))


def ordinal_probs_to_class_probs(p_ge):
    p_ge = np.asarray(p_ge)
    k = p_ge.shape[1]
    if k == 2:
        p0 = 1 - p_ge[:, 0]
        p1 = np.clip(p_ge[:, 0] - p_ge[:, 1], 0, 1)
        p2 = np.clip(p_ge[:, 1], 0, 1)
        return np.stack([p0, p1, p2], axis=1)
    p0 = 1 - p_ge[:, 0]
    p1 = np.clip(p_ge[:, 0] - p_ge[:, 1], 0, 1)
    p2 = np.clip(p_ge[:, 1] - p_ge[:, 2], 0, 1)
    p3 = np.clip(p_ge[:, 2], 0, 1)
    return np.stack([p0, p1, p2, p3], axis=1)


def corn_marginal_ge_probs(q):
    if torch.is_tensor(q):
        q = q.clamp(1e-6, 1 - 1e-6)
        return torch.cumprod(q, dim=1)
    q = np.asarray(q)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.cumprod(q, axis=1)


def build_joint_prior_mimic(grades, stages, eps=1e-3):
    grades = np.asarray(grades)
    stages = np.asarray(stages)
    prior = np.zeros((4, 3), dtype=np.float32)
    for g in range(4):
        mask = grades == g
        if mask.any():
            counts = np.bincount(stages[mask], minlength=3).astype(np.float32)
        else:
            counts = np.zeros(3, dtype=np.float32)
        counts = counts + eps
        prior[g] = counts / counts.sum()
    return prior


def _normalize_prior(prior, eps=1e-6):
    prior = np.asarray(prior, dtype=np.float32)
    prior = np.maximum(prior, eps)
    prior = prior / prior.sum(axis=1, keepdims=True)
    return prior


def compute_joint_distribution(pG, pS, prior=None, alpha=0.2, eps=1e-12):
    pG = np.asarray(pG)
    pS = np.asarray(pS)
    if prior is None:
        prior = np.ones((4, 3), dtype=np.float32)
    prior = _normalize_prior(prior)
    scores = []
    for g, s in JOINT_LABELS:
        score = pG[:, g] * pS[:, s] * (prior[g, s] ** alpha)
        scores.append(score)
    scores = np.stack(scores, axis=1)
    denom = np.sum(scores, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return scores / denom


def adjust_joint_predictions(grade_pred, stage_pred, pG, pS, prefer_high_stage=True):
    grade_pred = np.asarray(grade_pred)
    stage_pred = np.asarray(stage_pred)
    pG = np.asarray(pG)
    pS = np.asarray(pS)
    joint_pred = np.zeros_like(grade_pred)
    grade_adj = np.zeros_like(grade_pred)
    stage_adj = np.zeros_like(stage_pred)
    base_scores = []
    for g, s in JOINT_LABELS:
        bias = 1e-6 * s if prefer_high_stage else 0.0
        base_scores.append(pG[:, g] * pS[:, s] + bias)
    base_scores = np.stack(base_scores, axis=1)
    best_joint = base_scores.argmax(axis=1)
    for i, (g, s) in enumerate(zip(grade_pred, stage_pred)):
        idx = JOINT_LABEL_TO_INDEX.get((int(g), int(s)))
        if idx is None:
            idx = int(best_joint[i])
        joint_pred[i] = idx
        grade_adj[i], stage_adj[i] = JOINT_LABELS[int(idx)]
    return joint_pred, grade_adj, stage_adj


def joint_soft_accuracy(grade_gt, stage_gt, grade_pred, stage_pred, gamma_over=0.5):
    grade_gt = np.asarray(grade_gt)
    stage_gt = np.asarray(stage_gt)
    grade_pred = np.asarray(grade_pred)
    stage_pred = np.asarray(stage_pred)
    scores = np.zeros_like(stage_gt, dtype=np.float32)
    match_grade = grade_pred == grade_gt
    for i in range(len(stage_gt)):
        if not match_grade[i]:
            scores[i] = 0.0
            continue
        if stage_pred[i] == stage_gt[i]:
            scores[i] = 1.0
        elif stage_gt[i] == 1 and stage_pred[i] == 2:
            scores[i] = gamma_over
        elif stage_pred[i] < stage_gt[i]:
            scores[i] = 0.0
        else:
            scores[i] = 1.0
    return float(np.mean(scores)) if len(scores) > 0 else np.nan


def under_triage_metrics(stage_gt, stage_pred):
    stage_gt = np.asarray(stage_gt)
    stage_pred = np.asarray(stage_pred)
    if len(stage_gt) == 0:
        return np.nan, np.nan
    under_rate = float(np.mean(stage_pred < stage_gt))
    mask_high = stage_gt == 2
    if mask_high.any():
        under_high = float(np.mean(stage_pred[mask_high] < stage_gt[mask_high]))
    else:
        under_high = np.nan
    return under_rate, under_high


def evaluate_grade_stage_joint(
    y_grade,
    y_stage,
    p_ge_grade,
    p_ge_stage,
    prior=None,
    prior_alpha=0.2,
    softacc_gamma_over=0.5,
    ordinal_mode="default",
):
    y_grade = np.asarray(y_grade)
    y_stage = np.asarray(y_stage)
    p_ge_grade = np.asarray(p_ge_grade)
    p_ge_stage = np.asarray(p_ge_stage)

    grades_true = np.array([ordinal_targets_to_grade(row) for row in y_grade])
    stages_true = np.array([ordinal_targets_to_grade(row) for row in y_stage])

    pG = ordinal_probs_to_class_probs(p_ge_grade)
    pS = ordinal_probs_to_class_probs(p_ge_stage)

    metrics = {}
    metrics["ordinal_mode"] = ordinal_mode
    if ordinal_mode == "default":
        metrics["AUROC_ge1"] = safe_roc_auc(y_grade[:, 0], p_ge_grade[:, 0])
        metrics["AUROC_ge2"] = safe_roc_auc(y_grade[:, 1], p_ge_grade[:, 1])
        metrics["AUROC_ge3"] = safe_roc_auc(y_grade[:, 2], p_ge_grade[:, 2])

    grade_pred = pG.argmax(axis=1)
    metrics["macro_f1"] = f1_score(grades_true, grade_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    metrics["confmat_grade4"] = confusion_matrix(grades_true, grade_pred, labels=[0, 1, 2, 3]).astype(int).tolist()

    for g in range(4):
        labels = (grades_true == g).astype(int)
        metrics[f"AUROC_grade{g}"] = safe_roc_auc(labels, pG[:, g])
        metrics[f"AUPRC_grade{g}"] = safe_auprc(labels, pG[:, g])

    if ordinal_mode == "default":
        metrics["AUROC_stage_ge1"] = safe_roc_auc(y_stage[:, 0], p_ge_stage[:, 0])
        metrics["AUROC_stage_ge2"] = safe_roc_auc(y_stage[:, 1], p_ge_stage[:, 1])
        metrics["AUROC_ge1_stage"] = metrics["AUROC_stage_ge1"]
        metrics["AUROC_ge2_stage"] = metrics["AUROC_stage_ge2"]

    stage_pred = pS.argmax(axis=1)
    metrics["macro_f1_stage3"] = f1_score(stages_true, stage_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    metrics["confmat_stage3"] = confusion_matrix(stages_true, stage_pred, labels=[0, 1, 2]).astype(int).tolist()

    mask_midlow = np.isin(stages_true, [0, 1])
    if mask_midlow.any() and (~mask_midlow).any():
        labels_midlow = (stages_true[mask_midlow] == 1).astype(int)
        metrics["AUROC_midlow_vs_non"] = safe_roc_auc(labels_midlow, pS[mask_midlow, 1])
        metrics["AUPRC_midlow_vs_non"] = safe_auprc(labels_midlow, pS[mask_midlow, 1])
    else:
        metrics["AUROC_midlow_vs_non"] = np.nan
        metrics["AUPRC_midlow_vs_non"] = np.nan

    mask_high = np.isin(stages_true, [0, 2])
    if mask_high.any() and (~mask_high).any():
        labels_high = (stages_true[mask_high] == 2).astype(int)
        metrics["AUROC_high_vs_non"] = safe_roc_auc(labels_high, pS[mask_high, 2])
        metrics["AUPRC_high_vs_non"] = safe_auprc(labels_high, pS[mask_high, 2])
    else:
        metrics["AUROC_high_vs_non"] = np.nan
        metrics["AUPRC_high_vs_non"] = np.nan

    mask_high_midlow = np.isin(stages_true, [1, 2])
    if mask_high_midlow.any() and (~mask_high_midlow).any():
        labels_high_midlow = (stages_true[mask_high_midlow] == 2).astype(int)
        metrics["AUROC_high_vs_midlow"] = safe_roc_auc(labels_high_midlow, pS[mask_high_midlow, 2])
        metrics["AUPRC_high_vs_midlow"] = safe_auprc(labels_high_midlow, pS[mask_high_midlow, 2])
    else:
        metrics["AUROC_high_vs_midlow"] = np.nan
        metrics["AUPRC_high_vs_midlow"] = np.nan

    joint_gt = np.array([JOINT_LABEL_TO_INDEX.get((int(g), int(s)), -1) for g, s in zip(grades_true, stages_true)])
    mask_valid_joint = joint_gt >= 0
    if mask_valid_joint.any():
        joint_gt_valid = joint_gt[mask_valid_joint]
        grades_true = grades_true[mask_valid_joint]
        stages_true = stages_true[mask_valid_joint]
        pG = pG[mask_valid_joint]
        pS = pS[mask_valid_joint]
    else:
        joint_gt_valid = joint_gt

    joint_pred_hard_raw = pG.argmax(axis=1)
    stage_pred_hard_raw = pS.argmax(axis=1)
    joint_pred_hard, grade_pred_hard, stage_pred_hard = adjust_joint_predictions(
        joint_pred_hard_raw, stage_pred_hard_raw, pG, pS
    )

    P_joint = compute_joint_distribution(pG, pS, prior=prior, alpha=prior_alpha)
    joint_pred_pjoint = P_joint.argmax(axis=1)
    grade_pred_pjoint = np.array([JOINT_LABELS[idx][0] for idx in joint_pred_pjoint])
    stage_pred_pjoint = np.array([JOINT_LABELS[idx][1] for idx in joint_pred_pjoint])

    if len(joint_gt_valid) == 0:
        metrics["joint_exact_acc_hard"] = np.nan
        metrics["joint_exact_acc_pjoint"] = np.nan
        metrics["joint_macro_f1_hard"] = np.nan
        metrics["joint_macro_f1_pjoint"] = np.nan
        metrics["joint_confmat6_hard"] = []
        metrics["joint_confmat6_pjoint"] = []
    else:
        metrics["joint_exact_acc_hard"] = float(np.mean(joint_pred_hard == joint_gt_valid))
        metrics["joint_exact_acc_pjoint"] = float(np.mean(joint_pred_pjoint == joint_gt_valid))
        metrics["joint_macro_f1_hard"] = f1_score(joint_gt_valid, joint_pred_hard, labels=list(range(6)),
                                                  average="macro", zero_division=0)
        metrics["joint_macro_f1_pjoint"] = f1_score(joint_gt_valid, joint_pred_pjoint, labels=list(range(6)),
                                                    average="macro", zero_division=0)
        metrics["joint_confmat6_hard"] = confusion_matrix(
            joint_gt_valid, joint_pred_hard, labels=list(range(6))
        ).astype(int).tolist()
        metrics["joint_confmat6_pjoint"] = confusion_matrix(
            joint_gt_valid, joint_pred_pjoint, labels=list(range(6))
        ).astype(int).tolist()

    for idx, (g, s) in enumerate(JOINT_LABELS):
        labels = (joint_gt_valid == idx).astype(int) if len(joint_gt_valid) > 0 else np.array([])
        key = f"AUROC_joint_{g}{s}"
        metrics[key] = safe_roc_auc(labels, P_joint[:, idx]) if len(joint_gt_valid) > 0 else np.nan

    metrics["joint_softacc_hard"] = joint_soft_accuracy(grades_true, stages_true, grade_pred_hard, stage_pred_hard,
                                                        gamma_over=softacc_gamma_over)
    metrics["joint_softacc_pjoint"] = joint_soft_accuracy(grades_true, stages_true, grade_pred_pjoint, stage_pred_pjoint,
                                                          gamma_over=softacc_gamma_over)

    under_rate_hard, under_high_hard = under_triage_metrics(stages_true, stage_pred_hard)
    under_rate_pjoint, under_high_pjoint = under_triage_metrics(stages_true, stage_pred_pjoint)
    metrics["under_triage_rate_hard"] = under_rate_hard
    metrics["under_triage_rate_pjoint"] = under_rate_pjoint
    metrics["under_triage_on_high_hard"] = under_high_hard
    metrics["under_triage_on_high_pjoint"] = under_high_pjoint

    return metrics, pG, pS, P_joint


def multiclass_macro_auc(labels, probs, num_classes):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    aucs = []
    for cls in range(num_classes):
        y_true = (labels == cls).astype(int)
        aucs.append(safe_roc_auc(y_true, probs[:, cls]))
    return float(np.nanmean(aucs)) if len(aucs) > 0 else np.nan, aucs


def _plot_confusion_matrix(cm, labels, title, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel="True label",
           xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_roc_curve(y_true, y_score, title, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return auc


def _plot_roc_comparison(curves, title, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_true, y_score in curves:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_risk_distribution(scores, labels, title, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for lab in sorted(np.unique(labels)):
        ax.hist(scores[labels == lab], bins=30, alpha=0.5, label=f"class {lab}")
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_calibration_curve(y_true, y_prob, title, save_path, n_bins=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    if len(np.unique(y_true)) < 2:
        return None
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _hosmer_lemeshow(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return np.nan
    bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf
    hl = 0.0
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if not mask.any():
            continue
        obs = y_true[mask].sum()
        exp = y_prob[mask].sum()
        n = mask.sum()
        exp = np.clip(exp, 1e-6, None)
        hl += (obs - exp) ** 2 / (exp * (1 - exp / n))
    return float(hl)


def _plot_dca_curve(y_true, y_prob, title, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return None
    thresholds = np.linspace(0.01, 0.99, 99)
    n = len(y_true)
    net_benefits = []
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (thr / (1 - thr))
        net_benefits.append(nb)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresholds, net_benefits, label="Model")
    ax.plot(thresholds, np.zeros_like(thresholds), linestyle="--", color="gray", label="Treat None")
    prevalence = np.mean(y_true)
    treat_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, treat_all, linestyle="--", color="red", label="Treat All")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def compute_modethese_outputs(grades_true, stages_true, pG, pS, P_joint, output_dir):
    grades_true = np.asarray(grades_true)
    stages_true = np.asarray(stages_true)
    pG = np.asarray(pG)
    pS = np.asarray(pS)
    P_joint = np.asarray(P_joint)

    metrics = {}
    # Joint macro AUROC (OvR)
    joint_aurocs = []
    for idx in range(len(JOINT_LABELS)):
        y_true = (np.array([JOINT_LABEL_TO_INDEX.get((int(g), int(s)), -1)
                            for g, s in zip(grades_true, stages_true)]) == idx).astype(int)
        joint_aurocs.append(safe_roc_auc(y_true, P_joint[:, idx]))
    metrics["joint_macro_auroc_ovr"] = float(np.nanmean(joint_aurocs)) if joint_aurocs else np.nan
    metrics["joint_weighted_f1"] = f1_score(
        np.array([JOINT_LABEL_TO_INDEX.get((int(g), int(s)), -1) for g, s in zip(grades_true, stages_true)]),
        P_joint.argmax(axis=1),
        labels=list(range(len(JOINT_LABELS))),
        average="weighted",
        zero_division=0,
    )

    # Grade metrics
    grade_macro_auc, grade_auc_list = multiclass_macro_auc(grades_true, pG, num_classes=4)
    metrics["grade_macro_auc"] = grade_macro_auc
    metrics["grade_auc_per_class"] = grade_auc_list
    grade_pred = pG.argmax(axis=1)
    metrics["grade_f1_per_class"] = f1_score(grades_true, grade_pred, labels=[0, 1, 2, 3], average=None, zero_division=0).tolist()
    metrics["grade_acc"] = accuracy_score(grades_true, grade_pred)

    # Binary ROC for grade
    y_any_htn = (grades_true >= 1).astype(int)
    y_severe = (grades_true >= 2).astype(int)
    metrics["grade_any_htn_auc"] = safe_roc_auc(y_any_htn, pG[:, 1:].sum(axis=1))
    metrics["grade_severe_auc"] = safe_roc_auc(y_severe, pG[:, 2:].sum(axis=1))

    # Stage metrics
    stage_macro_auc, stage_auc_list = multiclass_macro_auc(stages_true, pS, num_classes=3)
    metrics["stage_macro_auc"] = stage_macro_auc
    metrics["stage_auc_per_class"] = stage_auc_list
    stage_pred = pS.argmax(axis=1)
    metrics["stage_acc"] = accuracy_score(stages_true, stage_pred)
    # High vs non-high
    y_high = (stages_true == 2).astype(int)
    pred_high = (stage_pred == 2).astype(int)
    tp = np.sum((pred_high == 1) & (y_high == 1))
    fn = np.sum((pred_high == 0) & (y_high == 1))
    tn = np.sum((pred_high == 0) & (y_high == 0))
    fp = np.sum((pred_high == 1) & (y_high == 0))
    metrics["stage_high_sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    metrics["stage_high_specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # Calibration metrics (high risk)
    metrics["brier_high"] = brier_score_loss(y_high, pS[:, 2]) if len(np.unique(y_high)) >= 2 else np.nan
    metrics["hosmer_lemeshow_high"] = _hosmer_lemeshow(y_high, pS[:, 2])
    metrics["nri"] = np.nan
    metrics["idi"] = np.nan

    # Figures
    fig_paths = {}
    fig_paths["Figure1_ROC_grade_any_htn"] = os.path.join(output_dir, "Figure1_ROC_grade_any_htn.png")
    _plot_roc_curve(y_any_htn, pG[:, 1:].sum(axis=1), "Any HTN vs None (Grade)", fig_paths["Figure1_ROC_grade_any_htn"])
    fig_paths["Figure2_Confmat_grade"] = os.path.join(output_dir, "Figure2_Confmat_grade.png")
    _plot_confusion_matrix(confusion_matrix(grades_true, grade_pred, labels=[0, 1, 2, 3]),
                            labels=["0", "1", "2", "3"], title="Grade Confusion Matrix",
                            save_path=fig_paths["Figure2_Confmat_grade"])

    fig_paths["Figure3_ROC_comparison"] = os.path.join(output_dir, "Figure3_ROC_comparison.png")
    _plot_roc_comparison(
        [
            ("Any HTN", y_any_htn, pG[:, 1:].sum(axis=1)),
            ("Severe", y_severe, pG[:, 2:].sum(axis=1)),
            ("High Risk", y_high, pS[:, 2]),
        ],
        title="ROC Comparison",
        save_path=fig_paths["Figure3_ROC_comparison"],
    )

    fig_paths["Figure4_DCA_high"] = os.path.join(output_dir, "Figure4_DCA_high.png")
    _plot_dca_curve(y_high, pS[:, 2], "Decision Curve (High Risk)", fig_paths["Figure4_DCA_high"])

    fig_paths["Figure5_Calibration_high"] = os.path.join(output_dir, "Figure5_Calibration_high.png")
    _plot_calibration_curve(y_high, pS[:, 2], "Calibration (High Risk)", fig_paths["Figure5_Calibration_high"])

    fig_paths["Figure_stage_risk_dist"] = os.path.join(output_dir, "Figure_stage_risk_dist.png")
    _plot_risk_distribution(pS[:, 2], stages_true, "Stage High-Risk Distribution", fig_paths["Figure_stage_risk_dist"])

    metrics["figure_paths"] = fig_paths
    return metrics




def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += np.abs(acc - conf) * (mask.sum() / max(n, 1))
    return float(ece)


def _joint_name_from_pred(g, s):
    idx = JOINT_LABEL_TO_INDEX.get((int(g), int(s)))
    if idx is None:
        return "INV"
    return f"{int(g)}{int(s)}"


def _safe_kappa(y_true, y_pred):
    from sklearn.metrics import cohen_kappa_score
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return np.nan


def _normalize_decoder_objective_name(objective):
    objective = str(objective or "qwk").lower()
    if objective == "marco_f1":
        return "macro_f1"
    return objective


def _decoder_objective_value(y_true, y_pred, objective, head="grade"):
    objective = _normalize_decoder_objective_name(objective)
    qwk = _safe_kappa(y_true, y_pred)
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    if head == "grade":
        rec_mid = 0.5 * (
            float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)) +
            float(recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0))
        )
    else:
        rec_mid = float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0))
    if objective == "qwk":
        return qwk
    if objective == "macro_f1":
        return macro
    if objective == "balanced_acc":
        return bal
    if objective == "mid_recall":
        return rec_mid
    if objective == "composite":
        return 0.50 * qwk + 0.30 * macro + 0.20 * rec_mid
    return qwk


def _decode_threshold(p_ge, thresholds):
    th = np.asarray(thresholds, dtype=float).reshape(1, -1)
    return np.sum(np.asarray(p_ge) >= th, axis=1).astype(int)


def _search_thresholds_for_decoder(y_true, p_ge, bins=101, objective="qwk", head="grade"):
    grid = np.linspace(0.0, 1.0, int(max(3, bins)))
    best_score, best_th = -np.inf, None
    if p_ge.shape[1] == 3:
        for t1 in grid:
            for t2 in grid[grid <= t1]:
                for t3 in grid[grid <= t2]:
                    pred = _decode_threshold(p_ge, [t1, t2, t3])
                    score = _decoder_objective_value(y_true, pred, objective, head=head)
                    if score > best_score:
                        best_score, best_th = score, [float(t1), float(t2), float(t3)]
    else:
        for u1 in grid:
            for u2 in grid[grid <= u1]:
                pred = _decode_threshold(p_ge, [u1, u2])
                score = _decoder_objective_value(y_true, pred, objective, head=head)
                if score > best_score:
                    best_score, best_th = score, [float(u1), float(u2)]
    return best_th, float(best_score)


def _expected_value_from_pge(p_ge):
    p_cls = ordinal_probs_to_class_probs(np.asarray(p_ge))
    values = np.arange(p_cls.shape[1], dtype=float)
    return p_cls @ values


def _decode_ev(ev_score, cutpoints):
    return np.digitize(ev_score, bins=np.asarray(cutpoints, dtype=float), right=False).astype(int)


def _search_cutpoints_for_ev(y_true, ev_score, n_classes, bins=101, objective="qwk", head="grade"):
    grid = np.linspace(0.0, float(n_classes - 1), int(max(3, bins)))
    best_score, best_cp = -np.inf, None
    if n_classes == 4:
        for c1 in grid:
            for c2 in grid[grid >= c1]:
                for c3 in grid[grid >= c2]:
                    pred = _decode_ev(ev_score, [c1, c2, c3])
                    score = _decoder_objective_value(y_true, pred, objective, head=head)
                    if score > best_score:
                        best_score, best_cp = score, [float(c1), float(c2), float(c3)]
    else:
        for d1 in grid:
            for d2 in grid[grid >= d1]:
                pred = _decode_ev(ev_score, [d1, d2])
                score = _decoder_objective_value(y_true, pred, objective, head=head)
                if score > best_score:
                    best_score, best_cp = score, [float(d1), float(d2)]
    return best_cp, float(best_score)


def _fit_temperature_grid(y_ord, p_ge, t_min=0.5, t_max=5.0, grid_size=91, t_init=1.0):
    y_ord = np.asarray(y_ord).astype(float)
    p_ge = np.clip(np.asarray(p_ge).astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(p_ge / (1.0 - p_ge))
    ts = np.linspace(float(t_min), float(t_max), int(max(3, grid_size)))
    if float(t_init) not in ts:
        ts = np.sort(np.unique(np.concatenate([ts, [float(t_init)]])))
    best_t, best_nll = float(t_init), np.inf
    for t in ts:
        p = 1.0 / (1.0 + np.exp(-(logits / t)))
        nll = -np.mean(y_ord * np.log(np.clip(p, 1e-8, 1.0)) + (1.0 - y_ord) * np.log(np.clip(1.0 - p, 1e-8, 1.0)))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t, best_nll


def _apply_temp(p_ge, temp):
    p_ge = np.clip(np.asarray(p_ge), 1e-6, 1 - 1e-6)
    logits = np.log(p_ge / (1 - p_ge))
    return 1.0 / (1.0 + np.exp(-(logits / temp)))


def extract_saved_thresholds_for_sep(thresholds_src):
    if not isinstance(thresholds_src, dict):
        return None, None
    gsrc = thresholds_src.get("grade") if isinstance(thresholds_src.get("grade"), dict) else None
    ssrc = thresholds_src.get("stage") if isinstance(thresholds_src.get("stage"), dict) else None
    if not (gsrc and ssrc):
        return None, None
    gobj = gsrc.get("youden", gsrc) if isinstance(gsrc.get("youden", gsrc), dict) else gsrc
    sobj = ssrc.get("youden", ssrc) if isinstance(ssrc.get("youden", ssrc), dict) else ssrc
    keys_ok = all(k in gobj for k in ["ge1", "ge2", "ge3"]) and all(k in sobj for k in ["ge1", "ge2"])
    if not keys_ok:
        return None, None
    thr_grade = [float(gobj["ge1"]), float(gobj["ge2"]), float(gobj["ge3"])]
    thr_stage = [float(sobj["ge1"]), float(sobj["ge2"])]
    return thr_grade, thr_stage


def _extract_saved_thresholds_for_sep(thresholds_src):
    # backward-compatible alias for older internal references
    return extract_saved_thresholds_for_sep(thresholds_src)


def evaluate_grade_stage_sep(y_grade, y_stage, p_ge_grade, p_ge_stage, output_dir, path_list=None, modethese=False,
                             decodermode="non", decoder_objective="qwk", decoder_bins=101,
                             decoder_use_saved_thresholds=True, decoder_save_debug=True,
                             temperature_init=1.0, temperature_min=0.5, temperature_max=5.0, temperature_grid_size=91,
                             decoder_keep_raw_metrics=True, thresholds_src=None,
                             val_y_grade=None, val_y_stage=None, val_p_ge_grade=None, val_p_ge_stage=None,
                             sep_head_mode="flat", aux_scores=None, loss_w_anyhtn=1.0, pos_weight_anyhtn=None,
                             coarse_auc_loss_mode="none", loss_w_anyhtn_auc=0.0, auc_margin=1.0, auc_pair_subsample=256,
                             fine_soft_label_mode="none", grade_soft_center=0.85, stage_label_smoothing=0.05,
                             loss_w_grade_soft=0.2, loss_w_stage_soft=0.1, loss_w_stage_smooth=1.0,
                             dataset_tag="sep_v1", v1_soft_label_mode="none", grade_soft_scheme="asym_v1", stage_soft_scheme="asym_v1",
                             lambda_incomp=0.0, lambda_joint=0.0, joint_gate="htn_only", joint_detach="both", incomp_mode="mask_sum"):
    y_grade = np.asarray(y_grade)
    y_stage = np.asarray(y_stage)
    p_ge_grade = np.asarray(p_ge_grade)
    p_ge_stage = np.asarray(p_ge_stage)
    grades_true = np.array([ordinal_targets_to_grade(row) for row in y_grade])
    stages_true = np.array([ordinal_targets_to_grade(row) for row in y_stage])
    pG = ordinal_probs_to_class_probs(p_ge_grade)
    pS = ordinal_probs_to_class_probs(p_ge_stage)
    grade_pred_raw = pG.argmax(axis=1)
    stage_pred_raw = pS.argmax(axis=1)
    grade_pred = grade_pred_raw.copy()
    stage_pred = stage_pred_raw.copy()

    mode = str(decodermode or "non").lower()
    decoder_objective = _normalize_decoder_objective_name(decoder_objective)
    saved_thr_grade, saved_thr_stage = extract_saved_thresholds_for_sep(thresholds_src) if decoder_use_saved_thresholds else (None, None)
    has_complete_saved_thresholds = (saved_thr_grade is not None and saved_thr_stage is not None)

    need_val = mode in {"ev", "temp_threshold", "temp_ev"}
    if mode == "threshold":
        need_val = not has_complete_saved_thresholds

    missing_val = (val_y_grade is None or val_y_stage is None or val_p_ge_grade is None or val_p_ge_stage is None)
    if need_val and missing_val:
        if mode == "threshold" and decoder_use_saved_thresholds and not has_complete_saved_thresholds:
            raise ValueError(
                "Decoder mode=threshold: 未找到可用的已保存阈值（grade.ge1/2/3, stage.ge1/2），且当前无法使用验证集重搜；"
                "请提供有效 --val_list，或提供完整 _thresholds.json，或改用 --decodermode non。"
            )
        raise ValueError(
            f"Decoder mode={mode} requires validation predictions for parameter search. "
            "Please provide valid --val_list (dataset_val must be available in test flow)."
        )

    used_saved = False
    val_used = False
    temp_grade = None
    temp_stage = None
    p_ge_grade_dec = p_ge_grade.copy()
    p_ge_stage_dec = p_ge_stage.copy()
    val_p_ge_grade_dec = np.asarray(val_p_ge_grade).copy() if val_p_ge_grade is not None else None
    val_p_ge_stage_dec = np.asarray(val_p_ge_stage).copy() if val_p_ge_stage is not None else None
    decoder_summary = {"decoder_mode": mode, "decoder_objective": decoder_objective}
    decoder_search = {}

    if mode in {"temp_threshold", "temp_ev"}:
        temp_grade, nll_g = _fit_temperature_grid(val_y_grade, val_p_ge_grade_dec, temperature_min, temperature_max, temperature_grid_size, temperature_init)
        temp_stage, nll_s = _fit_temperature_grid(val_y_stage, val_p_ge_stage_dec, temperature_min, temperature_max, temperature_grid_size, temperature_init)
        val_p_ge_grade_dec = _apply_temp(val_p_ge_grade_dec, temp_grade)
        val_p_ge_stage_dec = _apply_temp(val_p_ge_stage_dec, temp_stage)
        p_ge_grade_dec = _apply_temp(p_ge_grade_dec, temp_grade)
        p_ge_stage_dec = _apply_temp(p_ge_stage_dec, temp_stage)
        decoder_search["temperature"] = {"grade_nll": nll_g, "stage_nll": nll_s, "objective": "nll"}
        val_used = True

    if mode in {"threshold", "temp_threshold"}:
        thr_grade = saved_thr_grade if decoder_use_saved_thresholds else None
        thr_stage = saved_thr_stage if decoder_use_saved_thresholds else None
        if thr_grade is not None and thr_stage is not None:
            used_saved = True
        if thr_grade is None or thr_stage is None:
            val_grade_true = np.array([ordinal_targets_to_grade(row) for row in val_y_grade])
            val_stage_true = np.array([ordinal_targets_to_grade(row) for row in val_y_stage])
            thr_grade, best_g = _search_thresholds_for_decoder(val_grade_true, val_p_ge_grade_dec, bins=decoder_bins, objective=decoder_objective, head="grade")
            thr_stage, best_s = _search_thresholds_for_decoder(val_stage_true, val_p_ge_stage_dec, bins=decoder_bins, objective=decoder_objective, head="stage")
            decoder_search["threshold"] = {"grade_best": best_g, "stage_best": best_s}
            val_used = True
        grade_pred = _decode_threshold(p_ge_grade_dec, thr_grade)
        stage_pred = _decode_threshold(p_ge_stage_dec, thr_stage)
        decoder_summary["thresholds"] = {"grade": thr_grade, "stage": thr_stage}
    elif mode in {"ev", "temp_ev"}:
        ev_grade = _expected_value_from_pge(p_ge_grade_dec)
        ev_stage = _expected_value_from_pge(p_ge_stage_dec)
        val_ev_grade = _expected_value_from_pge(val_p_ge_grade_dec)
        val_ev_stage = _expected_value_from_pge(val_p_ge_stage_dec)
        val_grade_true = np.array([ordinal_targets_to_grade(row) for row in val_y_grade])
        val_stage_true = np.array([ordinal_targets_to_grade(row) for row in val_y_stage])
        cp_grade, best_g = _search_cutpoints_for_ev(val_grade_true, val_ev_grade, 4, decoder_bins, decoder_objective, "grade")
        cp_stage, best_s = _search_cutpoints_for_ev(val_stage_true, val_ev_stage, 3, decoder_bins, decoder_objective, "stage")
        grade_pred = _decode_ev(ev_grade, cp_grade)
        stage_pred = _decode_ev(ev_stage, cp_stage)
        decoder_summary["cutpoints"] = {"grade": cp_grade, "stage": cp_stage}
        decoder_search["ev"] = {"grade_best": best_g, "stage_best": best_s}
        val_used = True
    elif mode != "non":
        raise ValueError(f"Unsupported decodermode={decodermode}")

    prob_grade_any = 1.0 - pG[:, 0]
    prob_stage_any = 1.0 - pS[:, 0]

    metrics = {
        "MAE_grade": float(np.mean(np.abs(grade_pred - grades_true))),
        "MAE_stage": float(np.mean(np.abs(stage_pred - stages_true))),
        "QWK_grade": _safe_kappa(grades_true, grade_pred),
        "QWK_stage": _safe_kappa(stages_true, stage_pred),
    }
    metrics["AUROC_grade_any_htn"] = safe_roc_auc((grades_true > 0).astype(int), prob_grade_any)
    metrics["AUROC_grade_ge1"] = safe_roc_auc((grades_true >= 1).astype(int), p_ge_grade[:, 0])
    metrics["AUROC_grade_ge2"] = safe_roc_auc((grades_true >= 2).astype(int), p_ge_grade[:, 1])
    metrics["AUROC_grade_ge3"] = safe_roc_auc((grades_true >= 3).astype(int), p_ge_grade[:, 2])
    metrics["AUROC_stage_any_htn"] = safe_roc_auc((stages_true > 0).astype(int), prob_stage_any)
    metrics["AUROC_stage_ge1"] = safe_roc_auc((stages_true >= 1).astype(int), p_ge_stage[:, 0])
    metrics["AUROC_stage_ge2"] = safe_roc_auc((stages_true >= 2).astype(int), p_ge_stage[:, 1])

    _plot_roc_curve((grades_true > 0).astype(int), prob_grade_any, "ROC grade any HTN", os.path.join(output_dir, "roc_grade_any_htn.png"))
    _plot_roc_comparison([("grade>=1", (grades_true >= 1).astype(int), p_ge_grade[:, 0]), ("grade>=2", (grades_true >= 2).astype(int), p_ge_grade[:, 1]), ("grade>=3", (grades_true >= 3).astype(int), p_ge_grade[:, 2])], "ROC Grade Comparison", os.path.join(output_dir, "roc_grade_comparison.png"))
    cm_grade = confusion_matrix(grades_true, grade_pred, labels=[0, 1, 2, 3])
    _plot_confusion_matrix(cm_grade, ["0", "1", "2", "3"], "Grade Confmat", os.path.join(output_dir, "Confmat_grade.png"))
    _plot_roc_curve((stages_true > 0).astype(int), prob_stage_any, "ROC stage any HTN", os.path.join(output_dir, "roc_stage_any_htn.png"))
    _plot_roc_comparison([("stage>=1", (stages_true >= 1).astype(int), p_ge_stage[:, 0]), ("stage>=2", (stages_true >= 2).astype(int), p_ge_stage[:, 1])], "ROC Stage Comparison", os.path.join(output_dir, "roc_stage_comparison.png"))
    cm_stage = confusion_matrix(stages_true, stage_pred, labels=[0, 1, 2])
    _plot_confusion_matrix(cm_stage, ["0", "1", "2"], "Stage Confmat", os.path.join(output_dir, "Confmat_stage.png"))

    y_any_grade = (grades_true > 0).astype(int)
    _plot_calibration_curve(y_any_grade, prob_grade_any, "Calibration any HTN (grade)", os.path.join(output_dir, "calib_any_htn_grade.png"), n_bins=10)
    metrics["ECE_grade_any_htn"] = expected_calibration_error(y_any_grade, prob_grade_any, n_bins=10)
    metrics["Brier_grade_any_htn"] = float(brier_score_loss(y_any_grade, prob_grade_any)) if len(np.unique(y_any_grade)) >= 2 else np.nan
    y_any_stage = (stages_true > 0).astype(int)
    _plot_calibration_curve(y_any_stage, prob_stage_any, "Calibration any HTN (stage)", os.path.join(output_dir, "calib_any_htn_stage.png"), n_bins=10)
    metrics["ECE_stage_any_htn"] = expected_calibration_error(y_any_stage, prob_stage_any, n_bins=10)
    metrics["Brier_stage_any_htn"] = float(brier_score_loss(y_any_stage, prob_stage_any)) if len(np.unique(y_any_stage)) >= 2 else np.nan

    joint_gt_name = [_joint_name_from_pred(g, s) for g, s in zip(grades_true, stages_true)]
    joint_pred_name = [_joint_name_from_pred(g, s) for g, s in zip(grade_pred, stage_pred)]
    joint_pred_raw = [_joint_name_from_pred(g, s) for g, s in zip(grade_pred_raw, stage_pred_raw)]
    invalid_flag = np.array([v == "INV" for v in joint_pred_name])
    invalid_flag_raw = np.array([v == "INV" for v in joint_pred_raw])
    invalid_type = [f"g{g}_s{s}" if inv else "" for g, s, inv in zip(grade_pred, stage_pred, invalid_flag)]
    invalid_type_raw = [f"g{g}_s{s}" if inv else "" for g, s, inv in zip(grade_pred_raw, stage_pred_raw, invalid_flag_raw)]
    metrics["invalid_rate"] = float(invalid_flag.mean()) if len(invalid_flag) > 0 else np.nan
    metrics["sep_head_mode"] = str(sep_head_mode)
    metrics["coarse_to_fine_enabled"] = bool(str(sep_head_mode).lower() == "coarse_fine")
    metrics["loss_w_anyhtn"] = float(loss_w_anyhtn)
    metrics["coarse_auc_loss_mode"] = str(coarse_auc_loss_mode)
    metrics["loss_w_anyhtn_auc"] = float(loss_w_anyhtn_auc)
    metrics["auc_margin"] = float(auc_margin)
    metrics["auc_pair_subsample"] = int(auc_pair_subsample)
    metrics["fine_soft_label_mode"] = str(fine_soft_label_mode)
    metrics["grade_soft_center"] = float(grade_soft_center)
    metrics["stage_label_smoothing"] = float(stage_label_smoothing)
    metrics["loss_w_grade_soft"] = float(loss_w_grade_soft)
    metrics["loss_w_stage_soft"] = float(loss_w_stage_soft)
    metrics["loss_w_stage_smooth"] = float(loss_w_stage_smooth)
    metrics["v1_soft_label_mode"] = str(v1_soft_label_mode)
    metrics["grade_soft_scheme"] = str(grade_soft_scheme)
    metrics["stage_soft_scheme"] = str(stage_soft_scheme)
    metrics["lambda_incomp"] = float(lambda_incomp)
    metrics["lambda_joint"] = float(lambda_joint)
    metrics["joint_gate"] = str(joint_gate)
    metrics["joint_detach"] = str(joint_detach)
    metrics["incomp_mode"] = str(incomp_mode)
    metrics["decoder_mode"] = str(mode)
    metrics["decoder_objective"] = str(decoder_objective)
    metrics["decoder_keep_raw_metrics"] = bool(decoder_keep_raw_metrics)
    if pos_weight_anyhtn is not None:
        metrics["pos_weight_anyhtn"] = pos_weight_anyhtn
    if isinstance(aux_scores, dict) and "p_anyhtn_coarse" in aux_scores:
        p_any = np.asarray(aux_scores["p_anyhtn_coarse"]).reshape(-1)
        y_any = (grades_true > 0).astype(int)
        metrics["AUROC_anyhtn_coarse"] = safe_roc_auc(y_any, p_any)
        metrics["ECE_anyhtn_coarse"] = expected_calibration_error(y_any, p_any, n_bins=10)
        metrics["Brier_anyhtn_coarse"] = float(brier_score_loss(y_any, p_any)) if len(np.unique(y_any)) >= 2 else np.nan
        metrics["coarse_head_positive_rate_pred"] = float((p_any >= 0.5).mean())
        metrics["coarse_head_positive_rate_true"] = float(y_any.mean())
    metrics.update({"decoder_mode": mode, "decoder_objective": decoder_objective, "decoder_used_saved_thresholds": bool(used_saved), "decoder_has_val_search": bool(val_used)})
    if temp_grade is not None:
        metrics["temperature_grade"] = float(temp_grade)
    if temp_stage is not None:
        metrics["temperature_stage"] = float(temp_stage)
    if decoder_keep_raw_metrics:
        metrics.update({
            "MAE_grade_raw": float(np.mean(np.abs(grade_pred_raw - grades_true))),
            "MAE_stage_raw": float(np.mean(np.abs(stage_pred_raw - stages_true))),
            "QWK_grade_raw": _safe_kappa(grades_true, grade_pred_raw),
            "QWK_stage_raw": _safe_kappa(stages_true, stage_pred_raw),
            "macro_f1_grade_raw": float(f1_score(grades_true, grade_pred_raw, average="macro", zero_division=0)),
            "macro_f1_stage_raw": float(f1_score(stages_true, stage_pred_raw, average="macro", zero_division=0)),
            "invalid_rate_raw": float(invalid_flag_raw.mean()) if len(invalid_flag_raw) > 0 else np.nan,
        })

    labels7 = ["00", "11", "12", "21", "22", "32", "INV"]
    map7 = {k: i for i, k in enumerate(labels7)}
    y7 = np.array([map7.get(v, 6) for v in joint_gt_name])
    p7 = np.array([map7.get(v, 6) for v in joint_pred_name])
    cm7 = confusion_matrix(y7, p7, labels=list(range(7)))
    _plot_confusion_matrix(cm7, labels7, "Grade+Stage Confmat", os.path.join(output_dir, "Confmat_grade_stage.png"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cnt = {}
    for t in invalid_type:
        if t:
            cnt[t] = cnt.get(t, 0) + 1
    fig, ax = plt.subplots(figsize=(6, 4))
    if cnt:
        ax.bar(list(cnt.keys()), [cnt[k] for k in cnt.keys()])
        ax.tick_params(axis='x', rotation=45)
    ax.set_title("Invalid type histogram")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "invalid_type_hist.png"), dpi=200)
    plt.close(fig)

    ev_grade_score = _expected_value_from_pge(p_ge_grade_dec)
    ev_stage_score = _expected_value_from_pge(p_ge_stage_dec)
    rows = []
    for i in range(len(grades_true)):
        row = {
            "Path": path_list[i] if path_list is not None and i < len(path_list) else "",
            "grade_gt": int(grades_true[i]), "stage_gt": int(stages_true[i]),
            "grade_pred": int(grade_pred[i]), "stage_pred": int(stage_pred[i]),
            "grade_pred_raw": int(grade_pred_raw[i]), "stage_pred_raw": int(stage_pred_raw[i]),
            "grade_pred_final": int(grade_pred[i]), "stage_pred_final": int(stage_pred[i]),
            "prob_grade_any_htn": float(prob_grade_any[i]), "prob_stage_any_htn": float(prob_stage_any[i]),
            "joint_gt": joint_gt_name[i], "joint_pred": joint_pred_name[i],
            "joint_pred_raw": joint_pred_raw[i], "joint_pred_final": joint_pred_name[i],
            "invalid_flag": int(invalid_flag[i]), "invalid_type": invalid_type[i],
            "invalid_flag_raw": int(invalid_flag_raw[i]), "invalid_flag_final": int(invalid_flag[i]),
            "invalid_type_raw": invalid_type_raw[i], "invalid_type_final": invalid_type[i],
            "p_grade_ge1": float(p_ge_grade[i, 0]), "p_grade_ge2": float(p_ge_grade[i, 1]), "p_grade_ge3": float(p_ge_grade[i, 2]),
            "p_stage_ge1": float(p_ge_stage[i, 0]), "p_stage_ge2": float(p_ge_stage[i, 1]),
            "grade_ge1_score": float(p_ge_grade_dec[i, 0]), "grade_ge2_score": float(p_ge_grade_dec[i, 1]), "grade_ge3_score": float(p_ge_grade_dec[i, 2]),
            "stage_ge1_score": float(p_ge_stage_dec[i, 0]), "stage_ge2_score": float(p_ge_stage_dec[i, 1]),
            "decoder_mode": mode,
            "sep_head_mode": str(sep_head_mode),
        }
        if isinstance(aux_scores, dict) and "p_anyhtn_coarse" in aux_scores:
            row["p_anyhtn_coarse"] = float(np.asarray(aux_scores["p_anyhtn_coarse"]).reshape(-1)[i])
            row["coarse_pred_raw"] = int(row["p_anyhtn_coarse"] >= 0.5)
        if isinstance(aux_scores, dict) and "grade_pos_probs" in aux_scores:
            gpp = np.asarray(aux_scores["grade_pos_probs"])
            row["grade_pos_cond_prob_1"] = float(gpp[i, 0])
            row["grade_pos_cond_prob_2"] = float(gpp[i, 1])
            row["grade_pos_cond_prob_3"] = float(gpp[i, 2])
        if isinstance(aux_scores, dict) and "stage_pos_probs" in aux_scores:
            spp = np.asarray(aux_scores["stage_pos_probs"])
            row["stage_pos_cond_prob_1"] = float(spp[i, 0])
            row["stage_pos_cond_prob_2"] = float(spp[i, 1])
        if mode in {"ev", "temp_ev"}:
            row["grade_ev_score"] = float(ev_grade_score[i]); row["stage_ev_score"] = float(ev_stage_score[i])
        if mode in {"threshold", "temp_threshold"} and "thresholds" in decoder_summary:
            row["grade_threshold_t1"], row["grade_threshold_t2"], row["grade_threshold_t3"] = decoder_summary["thresholds"]["grade"]
            row["stage_threshold_u1"], row["stage_threshold_u2"] = decoder_summary["thresholds"]["stage"]
        rows.append(row)

    report_lines = [
        f"[{dataset_tag} test summary]", f"N={len(grades_true)}", "", "[scalar metrics]",
        f"MAE_grade={metrics['MAE_grade']:.6f}", f"QWK_grade={metrics['QWK_grade']:.6f}",
        f"MAE_stage={metrics['MAE_stage']:.6f}", f"QWK_stage={metrics['QWK_stage']:.6f}",
        f"AUROC_grade_any_htn={metrics['AUROC_grade_any_htn']}", f"AUROC_grade_ge1={metrics['AUROC_grade_ge1']}",
        f"AUROC_grade_ge2={metrics['AUROC_grade_ge2']}", f"AUROC_grade_ge3={metrics['AUROC_grade_ge3']}",
        f"AUROC_stage_any_htn={metrics['AUROC_stage_any_htn']}", f"AUROC_stage_ge1={metrics['AUROC_stage_ge1']}",
        f"AUROC_stage_ge2={metrics['AUROC_stage_ge2']}",
        f"ECE_grade_any_htn={metrics['ECE_grade_any_htn']}", f"Brier_grade_any_htn={metrics['Brier_grade_any_htn']}",
        f"ECE_stage_any_htn={metrics['ECE_stage_any_htn']}", f"Brier_stage_any_htn={metrics['Brier_stage_any_htn']}",
        f"invalid_rate={metrics['invalid_rate']}", "", "[Decoder Summary]",
        json.dumps({**decoder_summary, "used_saved_thresholds": used_saved, "has_val_search": val_used, "temperature_grade": temp_grade, "temperature_stage": temp_stage}, ensure_ascii=False),
        "", "[Coarse-to-Fine Summary]", f"sep_head_mode={sep_head_mode}", f"AUROC_anyhtn_coarse={metrics.get('AUROC_anyhtn_coarse')}",
        f"coarse_auc_loss_mode={metrics.get('coarse_auc_loss_mode')}", f"loss_w_anyhtn_auc={metrics.get('loss_w_anyhtn_auc')}",
        f"auc_margin={metrics.get('auc_margin')}", f"auc_pair_subsample={metrics.get('auc_pair_subsample')}",
        f"fine_soft_label_mode={metrics.get('fine_soft_label_mode')}", f"grade_soft_center={metrics.get('grade_soft_center')}",
        f"stage_label_smoothing={metrics.get('stage_label_smoothing')}", f"loss_w_grade_soft={metrics.get('loss_w_grade_soft')}",
        f"loss_w_stage_soft={metrics.get('loss_w_stage_soft')}", f"loss_w_stage_smooth={metrics.get('loss_w_stage_smooth')}",
        "", "[v1 Soft-Label Summary]", f"v1_soft_label_mode={metrics.get('v1_soft_label_mode')}",
        f"grade_soft_scheme={metrics.get('grade_soft_scheme')}", f"stage_soft_scheme={metrics.get('stage_soft_scheme')}",
        f"loss_w_grade_soft={metrics.get('loss_w_grade_soft')}", f"loss_w_stage_soft={metrics.get('loss_w_stage_soft')}",
        "", "[v1 Joint/Incomp Summary]", f"lambda_incomp={metrics.get('lambda_incomp')}", f"lambda_joint={metrics.get('lambda_joint')}",
        f"joint_gate={metrics.get('joint_gate')}", f"joint_detach={metrics.get('joint_detach')}", f"incomp_mode={metrics.get('incomp_mode')}",
        "", "[Confmat_grade labels=0,1,2,3]", np.array2string(cm_grade, separator=', '), "", "[Confmat_stage labels=0,1,2]", np.array2string(cm_stage, separator=', '), "",
        "[Confmat_grade_stage labels=00,11,12,21,22,32,INV]", np.array2string(cm7, separator=', '), "", "[invalid_type_count]", json.dumps(cnt, ensure_ascii=False, sort_keys=True), "", "[generated_figures]",
        "roc_grade_any_htn.png", "roc_grade_comparison.png", "Confmat_grade.png", "roc_stage_any_htn.png", "roc_stage_comparison.png", "Confmat_stage.png", "calib_any_htn_grade.png", "calib_any_htn_stage.png", "Confmat_grade_stage.png", "invalid_type_hist.png",
    ]
    if decoder_keep_raw_metrics:
        report_lines.extend(["", "[Raw vs Final Comparison]", f"QWK_grade_raw={metrics.get('QWK_grade_raw')}; QWK_grade_final={metrics.get('QWK_grade')}", f"QWK_stage_raw={metrics.get('QWK_stage_raw')}; QWK_stage_final={metrics.get('QWK_stage')}", f"MAE_grade_raw={metrics.get('MAE_grade_raw')}; MAE_grade_final={metrics.get('MAE_grade')}", f"MAE_stage_raw={metrics.get('MAE_stage_raw')}; MAE_stage_final={metrics.get('MAE_stage')}", f"invalid_rate_raw={metrics.get('invalid_rate_raw')}; invalid_rate_final={metrics.get('invalid_rate')}"])

    if decoder_save_debug:
        with open(os.path.join(output_dir, "decoder_config.json"), "w", encoding="utf-8") as f:
            json.dump({**decoder_summary, "used_saved_thresholds": used_saved, "has_val_search": val_used, "temperature_grade": temp_grade, "temperature_stage": temp_stage}, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, "decoder_search_summary.json"), "w", encoding="utf-8") as f:
            json.dump(decoder_search, f, indent=2, ensure_ascii=False)
        if decoder_keep_raw_metrics:
            with open(os.path.join(output_dir, "raw_vs_final_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"raw": {k: v for k, v in metrics.items() if k.endswith("_raw")}, "final": {"MAE_grade": metrics["MAE_grade"], "QWK_grade": metrics["QWK_grade"], "MAE_stage": metrics["MAE_stage"], "QWK_stage": metrics["QWK_stage"], "invalid_rate": metrics["invalid_rate"]}}, f, indent=2, ensure_ascii=False)

    return metrics, rows, report_lines


def save_thresholds_json(path, thresholds):
    # 将 numpy 标量转换为 Python float，避免 json 序列化报错
    def _to_float_dict(d):
        return {k: float(v) if v is not None else None for k, v in d.items()}

    if isinstance(thresholds, dict):
        thresholds = {k: _to_float_dict(v) if isinstance(v, dict) else v for k, v in thresholds.items()}
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)


def load_thresholds_json(path):
    with open(path, "r") as f:
        return json.load(f)

def metric_AUROC(target, output, nb_classes=14):
    outAUROC = []

    target = target.cpu().numpy()
    output = output.cpu().numpy()

    for i in range(nb_classes):
        outAUROC.append(roc_auc_score(target[:, i], output[:, i]))

    return outAUROC


def vararg_callback_bool(option, opt_str, value, parser):
    assert value is None

    arg = parser.rargs[0]
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        value = True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        value = False

    del parser.rargs[:1]
    setattr(parser.values, option.dest, value)


def vararg_callback_int(option, opt_str, value, parser):
    assert value is None
    value = []

    def intable(str):
        try:
            int(str)
            return True
        except ValueError:
            return False

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        # stop on -a, but not on -3 or -3.0
        if arg[:1] == "-" and len(arg) > 1 and not intable(arg):
            break
        value.append(int(arg))

    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

def step_decay(step, lr, epochs):

    progress = (step - 20) / float(epochs - 20)
    progress = np.clip(progress, 0.0, 1.0)
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))

    lr = lr * np.minimum(1., step / 20)

    return lr

def cosine_anneal_schedule(t,epochs,learning_rate):
    T=epochs
    M=1
    alpha_zero = learning_rate

    cos_inner = np.pi * (t % (T // M))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    return float(alpha_zero / 2 * cos_out)

def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1 > 0.5).astype(np.bool)
    im2 = np.asarray(im2 > 0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def mean_dice_coef(y_true,y_pred):
    sum=0
    for i in range (y_true.shape[0]):
        sum += dice(y_true[i,:,:,:],y_pred[i,:,:,:])
    return sum/y_true.shape[0]

def load_swin_pretrained(ckpt, model):
    state_dict = ckpt
    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            print(f"Error in loading {k}, passing......", file=writter)
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero

    # if 'head.bias' in state_dict:
    #     head_bias_pretrained = state_dict['head.bias']
    #     Nc1 = head_bias_pretrained.shape[0]
    # else:
    #     Nc1 = -1
    # Nc2 = model.head.bias.shape[0]
    #
    # if (Nc1 != Nc2):
    #     if Nc1 == 21841 and Nc2 == 1000:
    #         print("loading ImageNet-22K weight to ImageNet-1K ......", file=writter)
    #         map22kto1k_path = f'data/map22kto1k.txt'
    #         with open(map22kto1k_path) as f:
    #             map22kto1k = f.readlines()
    #         map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
    #         state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
    #         state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
    #     else:
    #         torch.nn.init.constant_(model.head.bias, 0.)
    #         torch.nn.init.constant_(model.head.weight, 0.)
    #         if Nc1 != -1:
    #             del state_dict['head.weight']
    #             del state_dict['head.bias']
    #         print(f"Error in loading classifier head, re-init classifier head to 0", file=writter)



    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    del ckpt
    torch.cuda.empty_cache()


def corn_probs_to_class_probs_torch(p_ge):
    if p_ge.shape[1] == 3:
        p0 = 1.0 - p_ge[:, 0]
        p1 = np_or_torch_clamp(p_ge[:, 0] - p_ge[:, 1])
        p2 = np_or_torch_clamp(p_ge[:, 1] - p_ge[:, 2])
        p3 = np_or_torch_clamp(p_ge[:, 2])
        p = stack_like(p_ge, [p0, p1, p2, p3])
    else:
        s0 = 1.0 - p_ge[:, 0]
        s1 = np_or_torch_clamp(p_ge[:, 0] - p_ge[:, 1])
        s2 = np_or_torch_clamp(p_ge[:, 1])
        p = stack_like(p_ge, [s0, s1, s2])
    return p / sum_like(p, axis=1, keepdims=True).clip(min=1e-8)


def np_or_torch_clamp(x, minv=0.0, maxv=1.0):
    if hasattr(x, 'clamp'):
        return x.clamp(minv, maxv)
    return np.clip(x, minv, maxv)


def stack_like(ref, arrs):
    if isinstance(ref, torch.Tensor):
        return torch.stack(arrs, dim=1)
    return np.stack(arrs, axis=1)


def sum_like(x, axis=1, keepdims=True):
    if isinstance(x, torch.Tensor):
        return x.sum(dim=axis, keepdim=keepdims)
    return x.sum(axis=axis, keepdims=keepdims)


def build_v2_joint_graph_distance_matrix_numpy(joint_graph_w_00_11=1.0, joint_graph_w_11_21=0.6,
                                             joint_graph_w_11_12=1.2, joint_graph_w_21_22=0.8,
                                             joint_graph_w_12_22=0.7, joint_graph_w_22_32=1.5):
    inf = 1e9
    D = np.full((6, 6), inf, dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    edges = [
        (0, 1, joint_graph_w_00_11),
        (1, 3, joint_graph_w_11_21),
        (1, 2, joint_graph_w_11_12),
        (3, 4, joint_graph_w_21_22),
        (2, 4, joint_graph_w_12_22),
        (4, 5, joint_graph_w_22_32),
    ]
    for i, j, w in edges:
        D[i, j] = min(D[i, j], float(w))
        D[j, i] = min(D[j, i], float(w))
    for k in range(6):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D


def compose_v2_joint_predictions(p_ge_grade, p_ge_stage, q1_logit, q2_logit, alpha_gate_min=0.15, alpha_gate_max=0.65,
                                 joint_beta_stage=0.5, joint_gamma_cond=0.5, eps=1e-8, use_entropy_alpha=True):
    pG = corn_probs_to_class_probs_torch(p_ge_grade)
    pS = corn_probs_to_class_probs_torch(p_ge_stage)
    if isinstance(pG, torch.Tensor):
        if use_entropy_alpha:
            H = -(pG.clamp_min(eps) * torch.log(pG.clamp_min(eps))).sum(dim=1) / np.log(4.0)
            alpha = alpha_gate_min + (alpha_gate_max - alpha_gate_min) * H
        else:
            alpha = torch.ones((pG.shape[0],), dtype=pG.dtype, device=pG.device)
        q1 = torch.sigmoid(q1_logit.view(-1))
        q2 = torch.sigmoid(q2_logit.view(-1))
        pg = pG.clamp(eps, 1.0); ps = pS.clamp(eps, 1.0); q1c = q1.clamp(eps, 1.0 - eps); q2c = q2.clamp(eps, 1.0 - eps)
        joint_logits = torch.stack([
            torch.log(pg[:, 0]) + alpha * joint_beta_stage * torch.log(ps[:, 0]),
            torch.log(pg[:, 1]) + alpha * joint_beta_stage * torch.log(ps[:, 1]) + alpha * joint_gamma_cond * torch.log(q1c),
            torch.log(pg[:, 1]) + alpha * joint_beta_stage * torch.log(ps[:, 2]) + alpha * joint_gamma_cond * torch.log(1.0 - q1c),
            torch.log(pg[:, 2]) + alpha * joint_beta_stage * torch.log(ps[:, 1]) + alpha * joint_gamma_cond * torch.log(q2c),
            torch.log(pg[:, 2]) + alpha * joint_beta_stage * torch.log(ps[:, 2]) + alpha * joint_gamma_cond * torch.log(1.0 - q2c),
            torch.log(pg[:, 3]) + alpha * joint_beta_stage * torch.log(ps[:, 2]),
        ], dim=1)
        P_joint6 = torch.softmax(joint_logits, dim=1)
        pG_fused = torch.stack([P_joint6[:, 0], P_joint6[:, 1] + P_joint6[:, 2], P_joint6[:, 3] + P_joint6[:, 4], P_joint6[:, 5]], dim=1)
        pS_fused = torch.stack([P_joint6[:, 0], P_joint6[:, 1] + P_joint6[:, 3], P_joint6[:, 2] + P_joint6[:, 4] + P_joint6[:, 5]], dim=1)
        return {"pG_raw4": pG, "pS_ind3": pS, "q1": q1.unsqueeze(1), "q2": q2.unsqueeze(1), "alpha": alpha.unsqueeze(1), "P_joint6": P_joint6, "pG_fused4": pG_fused, "pS_fused3": pS_fused}
    raise TypeError('compose_v2_joint_predictions expects torch tensors')


def evaluate_grade_stage_v2(y_grade, y_stage, p_ge_grade, p_ge_stage, output_dir, path_list=None, modethese=False,
                            decodermode='non', decoder_objective='qwk', decoder_bins=101,
                            decoder_use_saved_thresholds=True, decoder_save_debug=True,
                            temperature_init=1.0, temperature_min=0.5, temperature_max=5.0, temperature_grid_size=91,
                            decoder_keep_raw_metrics=True, thresholds_src=None,
                            val_y_grade=None, val_y_stage=None, val_p_ge_grade=None, val_p_ge_stage=None,
                            dataset_tag='dataset', aux_scores=None, **kwargs):
    sep_kwargs = {
        "sep_head_mode": kwargs.get("sep_head_mode", "flat"),
        "loss_w_anyhtn": kwargs.get("loss_w_anyhtn", 1.0),
        "pos_weight_anyhtn": kwargs.get("pos_weight_anyhtn", None),
        "coarse_auc_loss_mode": kwargs.get("coarse_auc_loss_mode", "none"),
        "loss_w_anyhtn_auc": kwargs.get("loss_w_anyhtn_auc", 0.0),
        "auc_margin": kwargs.get("auc_margin", 1.0),
        "auc_pair_subsample": kwargs.get("auc_pair_subsample", 256),
        "fine_soft_label_mode": kwargs.get("fine_soft_label_mode", "none"),
        "grade_soft_center": kwargs.get("grade_soft_center", 0.85),
        "stage_label_smoothing": kwargs.get("stage_label_smoothing", 0.05),
        "loss_w_grade_soft": kwargs.get("loss_w_grade_soft", 0.2),
        "loss_w_stage_soft": kwargs.get("loss_w_stage_soft", 0.1),
        "loss_w_stage_smooth": kwargs.get("loss_w_stage_smooth", 1.0),
        "dataset_tag": dataset_tag,
        "v1_soft_label_mode": kwargs.get("v1_soft_label_mode", "none"),
        "grade_soft_scheme": kwargs.get("grade_soft_scheme", "asym_v1"),
        "stage_soft_scheme": kwargs.get("stage_soft_scheme", "asym_v1"),
        "lambda_incomp": kwargs.get("lambda_incomp", 0.0),
        "lambda_joint": kwargs.get("lambda_joint", 0.0),
        "joint_gate": kwargs.get("joint_gate", "htn_only"),
        "joint_detach": kwargs.get("joint_detach", "both"),
        "incomp_mode": kwargs.get("incomp_mode", "mask_sum"),
    }
    metrics, rows, report_lines = evaluate_grade_stage_sep(
        y_grade, y_stage, p_ge_grade, p_ge_stage, output_dir, path_list=path_list, modethese=modethese,
        decodermode=decodermode, decoder_objective=decoder_objective, decoder_bins=decoder_bins,
        decoder_use_saved_thresholds=decoder_use_saved_thresholds, decoder_save_debug=decoder_save_debug,
        temperature_init=temperature_init, temperature_min=temperature_min, temperature_max=temperature_max,
        temperature_grid_size=temperature_grid_size, decoder_keep_raw_metrics=decoder_keep_raw_metrics,
        thresholds_src=thresholds_src, val_y_grade=val_y_grade, val_y_stage=val_y_stage,
        val_p_ge_grade=val_p_ge_grade, val_p_ge_stage=val_p_ge_stage,
        aux_scores=aux_scores, **sep_kwargs)
    if aux_scores is None or 'p_joint6' not in aux_scores:
        return metrics, rows, report_lines
    p_joint6 = np.asarray(aux_scores['p_joint6'])
    pG_fused = np.asarray(aux_scores['pG_fused'])
    pS_fused = np.asarray(aux_scores['pS_fused'])
    q1 = np.asarray(aux_scores['q1']).reshape(-1)
    q2 = np.asarray(aux_scores['q2']).reshape(-1)
    alpha = np.asarray(aux_scores['alpha_gate']).reshape(-1)
    grades_true = np.array([ordinal_targets_to_grade(row) for row in np.asarray(y_grade)])
    stages_true = np.array([ordinal_targets_to_grade(row) for row in np.asarray(y_stage)])
    grade_raw = pG_fused * 0.0 + corn_probs_to_class_probs_torch(np.asarray(p_ge_grade))
    stage_raw = pS_fused * 0.0 + corn_probs_to_class_probs_torch(np.asarray(p_ge_stage))
    grade_pred_raw = np.argmax(grade_raw, axis=1)
    stage_pred_raw = np.argmax(stage_raw, axis=1)
    joint_pred_fused = np.argmax(p_joint6, axis=1)
    grade_pred_fused = np.array([JOINT_LABELS[idx][0] for idx in joint_pred_fused])
    stage_pred_fused = np.array([JOINT_LABELS[idx][1] for idx in joint_pred_fused])
    joint_graph_w_00_11 = float(kwargs['joint_graph_w_00_11'])
    joint_graph_w_11_21 = float(kwargs['joint_graph_w_11_21'])
    joint_graph_w_11_12 = float(kwargs['joint_graph_w_11_12'])
    joint_graph_w_21_22 = float(kwargs['joint_graph_w_21_22'])
    joint_graph_w_12_22 = float(kwargs['joint_graph_w_12_22'])
    joint_graph_w_22_32 = float(kwargs['joint_graph_w_22_32'])
    D = build_v2_joint_graph_distance_matrix_numpy(
        joint_graph_w_00_11=joint_graph_w_00_11,
        joint_graph_w_11_21=joint_graph_w_11_21,
        joint_graph_w_11_12=joint_graph_w_11_12,
        joint_graph_w_21_22=joint_graph_w_21_22,
        joint_graph_w_12_22=joint_graph_w_12_22,
        joint_graph_w_22_32=joint_graph_w_22_32,
    )
    joint_gt = np.array([JOINT_LABEL_TO_INDEX[(int(g), int(s))] for g, s in zip(grades_true, stages_true)])
    fused_pairs = [(int(g), int(s)) for g, s in zip(grade_pred_fused, stage_pred_fused)]
    fused_invalid = np.array([pair not in JOINT_LABEL_TO_INDEX for pair in fused_pairs], dtype=np.float32)
    expected_graph_cost = float(np.mean(np.sum(p_joint6 * D[joint_gt], axis=1)))
    metrics.update({
        'Confmat_grade_raw': confusion_matrix(grades_true, grade_pred_raw, labels=[0,1,2,3]).tolist(),
        'Confmat_grade_fused': confusion_matrix(grades_true, grade_pred_fused, labels=[0,1,2,3]).tolist(),
        'Confmat_stage_raw': confusion_matrix(stages_true, stage_pred_raw, labels=[0,1,2]).tolist(),
        'Confmat_stage_fused': confusion_matrix(stages_true, stage_pred_fused, labels=[0,1,2]).tolist(),
        'Confmat_joint_fused_6class': confusion_matrix(joint_gt, joint_pred_fused, labels=list(range(6))).tolist(),
        'fused_invalid_rate': float(fused_invalid.mean()) if fused_invalid.size > 0 else np.nan,
        'mean_alpha_gate': float(alpha.mean()),
        'mean_q1': float(q1.mean()),
        'mean_q2': float(q2.mean()),
        'mean_expected_joint_graph_cost': expected_graph_cost,
        'joint_graph_tau': float(kwargs['joint_graph_tau']),
        'joint_beta_stage': float(kwargs['joint_beta_stage']),
        'joint_gamma_cond': float(kwargs['joint_gamma_cond']),
        'lambda_stage_marg': float(kwargs['lambda_stage_marg']),
        'lambda_cond_stage': float(kwargs['lambda_cond_stage']),
        'lambda_soft_joint': float(kwargs['lambda_soft_joint']),
        'stage_fused_aux_weight': float(kwargs['stage_fused_aux_weight']),
        'cond_pos_weight_g1': float(kwargs['cond_pos_weight_g1']),
        'cond_pos_weight_g2': float(kwargs['cond_pos_weight_g2']),
        'alpha_gate_min': float(kwargs['alpha_gate_min']),
        'alpha_gate_max': float(kwargs['alpha_gate_max']),
        'v2_soft_joint_start_epoch': int(kwargs['v2_soft_joint_start_epoch']),
        'v2_soft_joint_warmup_epochs': int(kwargs['v2_soft_joint_warmup_epochs']),
        'use_stopgrad_grade_for_cond': bool(kwargs['use_stopgrad_grade_for_cond']),
        'teacher_force_grade_epochs': f"{int(kwargs['teacher_force_grade_epochs'])} / not_used_in_v2_patch1",
        'v2_disable_legacy_joint': True,
        'joint_graph_w_00_11': joint_graph_w_00_11,
        'joint_graph_w_11_21': joint_graph_w_11_21,
        'joint_graph_w_11_12': joint_graph_w_11_12,
        'joint_graph_w_21_22': joint_graph_w_21_22,
        'joint_graph_w_12_22': joint_graph_w_12_22,
        'joint_graph_w_22_32': joint_graph_w_22_32,
        'graph_edge_weights_summary': {
            '00_11': joint_graph_w_00_11, '11_21': joint_graph_w_11_21,
            '11_12': joint_graph_w_11_12, '21_22': joint_graph_w_21_22,
            '12_22': joint_graph_w_12_22, '22_32': joint_graph_w_22_32,
        },
        'joint_graph_distance_matrix': D.tolist(),
    })
    mask_g1 = grades_true == 1
    mask_g2 = grades_true == 2
    metrics['AUC_cond_11_vs12'] = safe_roc_auc((stages_true[mask_g1] == 1).astype(int), q1[mask_g1]) if mask_g1.sum() > 1 else np.nan
    metrics['AUC_cond_21_vs22'] = safe_roc_auc((stages_true[mask_g2] == 1).astype(int), q2[mask_g2]) if mask_g2.sum() > 1 else np.nan
    _plot_confusion_matrix(confusion_matrix(grades_true, grade_pred_fused, labels=[0,1,2,3]), ["0","1","2","3"], 'Grade Fused Confmat', os.path.join(output_dir, 'Confmat_grade_fused.png'))
    _plot_confusion_matrix(confusion_matrix(stages_true, stage_pred_fused, labels=[0,1,2]), ["0","1","2"], 'Stage Fused Confmat', os.path.join(output_dir, 'Confmat_stage_fused.png'))
    _plot_confusion_matrix(confusion_matrix(joint_gt, joint_pred_fused, labels=list(range(6))), ["00","11","12","21","22","32"], 'Joint Fused Confmat', os.path.join(output_dir, 'Confmat_grade_stage_fused.png'))
    for i, row in enumerate(rows):
        row.update({
            'grade_pred_raw_v2': int(grade_pred_raw[i]), 'stage_pred_raw_v2': int(stage_pred_raw[i]),
            'joint_pred_fused': int(joint_pred_fused[i]), 'grade_pred_fused': int(grade_pred_fused[i]), 'stage_pred_fused': int(stage_pred_fused[i]),
            'q1': float(q1[i]), 'q2': float(q2[i]), 'alpha_gate': float(alpha[i]),
            'p_joint_00': float(p_joint6[i,0]), 'p_joint_11': float(p_joint6[i,1]), 'p_joint_12': float(p_joint6[i,2]),
            'p_joint_21': float(p_joint6[i,3]), 'p_joint_22': float(p_joint6[i,4]), 'p_joint_32': float(p_joint6[i,5]),
        })
    report_lines.extend(['', '[v2 Fused Summary]', json.dumps({
        'fused_invalid_rate': metrics['fused_invalid_rate'], 'mean_alpha_gate': metrics['mean_alpha_gate'],
        'mean_q1': metrics['mean_q1'], 'mean_q2': metrics['mean_q2'], 'AUC_cond_11_vs12': metrics['AUC_cond_11_vs12'],
        'AUC_cond_21_vs22': metrics['AUC_cond_21_vs22'], 'v2_disable_legacy_joint': True,
        'teacher_force_grade_epochs': metrics['teacher_force_grade_epochs'],
        'graph_edge_weights_summary': metrics['graph_edge_weights_summary'],
    }, ensure_ascii=False)])
    return metrics, rows, report_lines
