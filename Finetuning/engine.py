
import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import copy
import json
import csv
import pandas as pd

from models import build_classification_model, save_checkpoint
from utils import *
from sklearn.metrics import accuracy_score

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
from trainer import train_one_epoch, evaluate, test_classification, test_model

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import torch.nn.functional as F

sys.setrecursionlimit(40000)

def safe_collate(batch):
    """过滤 (None, None) 的样本；若整批无有效样本，返回 None。"""
    batch = [b for b in batch if b is not None and b[0] is not None and b[1] is not None]
    if len(batch) == 0:
        return None
    # 支持 (img, label, path) 结构，路径保持为 list
    if len(batch[0]) == 3:
        imgs, labels, paths = zip(*batch)
        return default_collate(imgs), default_collate(labels), list(paths)
    return default_collate(batch)


def _collect_outputs(model, data_loader, device, args):
    """无梯度地收集 logits->sigmoid 概率与标签"""
    model.eval()
    y_all = torch.FloatTensor().to(device)
    p_all = torch.FloatTensor().to(device)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if batch is None:
                continue
            samples, targets = batch
            samples = samples.float().to(device)
            targets = targets.float().to(device)
            y_all = torch.cat((y_all, targets), 0)
            out = model(samples)
            out = torch.sigmoid(out)
            p_all = torch.cat((p_all, out), 0)
    return y_all.cpu().numpy(), p_all.cpu().numpy()


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class WeightedOrdinalCrossEntropy(torch.nn.Module):
    """基于 BCEWithLogitsLoss 的 ordinal 三通道损失"""

    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)


class MultiHeadOrdinalLoss(torch.nn.Module):
    def __init__(self, pos_weight_grade=None, pos_weight_stage=None, w_grade=1.0, w_stage=1.0,
                 use_joint_train=False, lambda_incomp=0.0, lambda_joint=0.0, joint_gate="htn_only",
                 joint_detach="both", joint_ce_weight=None, joint_warmup_epochs=5, incomp_mode="mask_sum",
                 joint_loss_use_prior=False, joint_prior=None, joint_prior_alpha=0.2,
                 ordinal_mode="default"):
        super().__init__()
        self.loss_grade = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_grade)
        self.loss_stage = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_stage)
        self.loss_joint = torch.nn.NLLLoss(weight=joint_ce_weight)
        self.w_grade = w_grade
        self.w_stage = w_stage
        self.w_anyhtn = 1.0
        self.pos_weight_anyhtn = None
        self.coarse_auc_loss_mode = "none"
        self.loss_w_anyhtn_auc = 0.0
        self.auc_margin = 1.0
        self.auc_pair_subsample = 256
        self.auc_loss_detach_probs = False
        self.fine_soft_label_mode = "none"
        self.grade_soft_center = 0.85
        self.stage_label_smoothing = 0.05
        self.loss_w_grade_soft = 0.2
        self.loss_w_stage_soft = 0.1
        self.loss_w_stage_smooth = 1.0
        self.v1_soft_label_mode = "none"
        self.grade_soft_scheme = "asym_v1"
        self.stage_soft_scheme = "asym_v1"
        self.use_joint_train = use_joint_train
        self.lambda_incomp = lambda_incomp
        self.lambda_joint = lambda_joint
        self.joint_gate = joint_gate
        self.joint_detach = joint_detach
        self.joint_warmup_epochs = joint_warmup_epochs
        self.incomp_mode = incomp_mode
        self.joint_loss_use_prior = joint_loss_use_prior
        self.joint_prior = joint_prior
        self.joint_prior_alpha = joint_prior_alpha
        self.ordinal_mode = ordinal_mode
        self.current_epoch = 0
        self.last_components = None
        self.lpv3_active = False
        self.lpv3_enable_cond_after_epoch = 3
        self.lpv3_enable_soft_joint_after_epoch = 10

    def _compute_pairwise_auc_loss(self, scores, labels):
        mode = str(getattr(self, "coarse_auc_loss_mode", "none") or "none").lower()
        weight = float(getattr(self, "loss_w_anyhtn_auc", 0.0) or 0.0)
        if mode == "none" or weight <= 0:
            return scores.new_tensor(0.0)

        labels = labels.view(-1)
        scores = scores.view(-1)
        if getattr(self, "auc_loss_detach_probs", False):
            scores = scores.detach()

        pos_idx = torch.where(labels > 0.5)[0]
        neg_idx = torch.where(labels <= 0.5)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            return scores.new_tensor(0.0)

        max_n = int(getattr(self, "auc_pair_subsample", 256) or 256)
        if max_n > 0 and pos_idx.numel() > max_n:
            pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[:max_n]]
        if max_n > 0 and neg_idx.numel() > max_n:
            neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[:max_n]]

        pos_scores = scores[pos_idx].view(-1, 1)
        neg_scores = scores[neg_idx].view(1, -1)
        delta = pos_scores - neg_scores
        if mode == "pairwise_hinge":
            margin = float(getattr(self, "auc_margin", 1.0) or 1.0)
            return F.relu(margin - delta).mean()
        if mode == "pairwise_logistic":
            return F.softplus(-delta).mean()
        return scores.new_tensor(0.0)

    def _build_positive_grade_probs_from_corn(self, logits_pos):
        q = torch.sigmoid(logits_pos)
        a1 = q[:, 0:1]
        a2 = q[:, 1:2]
        p1 = torch.clamp(1.0 - a1, 0.0, 1.0)
        p2 = torch.clamp(a1 * (1.0 - a2), 0.0, 1.0)
        p3 = torch.clamp(a1 * a2, 0.0, 1.0)
        p = torch.cat([p1, p2, p3], dim=1)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return p

    def _build_grade_soft_targets(self, raw_grade_pos, center):
        center = float(center)
        rest = max(0.0, 1.0 - center)
        y = torch.zeros((raw_grade_pos.shape[0], 3), dtype=torch.float32, device=raw_grade_pos.device)
        cls = (raw_grade_pos.long() - 1).clamp(min=0, max=2)
        if y.shape[0] == 0:
            return y
        # grade=1
        m0 = cls == 0
        y[m0, 0] = center
        y[m0, 1] = rest
        # grade=2
        m1 = cls == 1
        y[m1, 0] = rest / 2.0
        y[m1, 1] = center
        y[m1, 2] = rest / 2.0
        # grade=3
        m2 = cls == 2
        y[m2, 1] = rest
        y[m2, 2] = center
        return y

    def _compute_grade_soft_loss(self, p_grade_pos, y_soft):
        p = p_grade_pos.clamp_min(1e-8)
        return (-(y_soft * torch.log(p)).sum(dim=1)).mean()

    def _smooth_stage_targets(self, stage_pos_bin, eps):
        eps = float(eps)
        y = stage_pos_bin.float().view(-1, 1)
        return y * (1.0 - eps) + (1.0 - y) * eps

    def _build_full_grade_probs_from_ge(self, ge_g):
        pG0 = 1.0 - ge_g[:, 0]
        pG1 = torch.clamp(ge_g[:, 0] - ge_g[:, 1], 0.0, 1.0)
        pG2 = torch.clamp(ge_g[:, 1] - ge_g[:, 2], 0.0, 1.0)
        pG3 = torch.clamp(ge_g[:, 2], 0.0, 1.0)
        pG = torch.stack([pG0, pG1, pG2, pG3], dim=1)
        return pG / pG.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _build_full_stage_probs_from_ge(self, ge_s):
        pS0 = 1.0 - ge_s[:, 0]
        pS1 = torch.clamp(ge_s[:, 0] - ge_s[:, 1], 0.0, 1.0)
        pS2 = torch.clamp(ge_s[:, 1], 0.0, 1.0)
        pS = torch.stack([pS0, pS1, pS2], dim=1)
        return pS / pS.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _build_v1_grade_soft_targets(self, raw_grade, scheme="asym_v1"):
        y = torch.zeros((raw_grade.shape[0], 4), dtype=torch.float32, device=raw_grade.device)
        cls = raw_grade.long().clamp(min=0, max=3)
        # grade0: [0.95, 0.05, 0.00, 0.00]
        m0 = cls == 0
        y[m0, 0] = 0.95
        y[m0, 1] = 0.05
        # grade1: [0.10, 0.80, 0.10, 0.00]
        m1 = cls == 1
        y[m1, 0] = 0.10
        y[m1, 1] = 0.80
        y[m1, 2] = 0.10
        # grade2: [0.00, 0.10, 0.80, 0.10]
        m2 = cls == 2
        y[m2, 1] = 0.10
        y[m2, 2] = 0.80
        y[m2, 3] = 0.10
        # grade3: [0.00, 0.00, 0.05, 0.95]
        m3 = cls == 3
        y[m3, 2] = 0.05
        y[m3, 3] = 0.95
        return y

    def _build_v1_stage_soft_targets(self, raw_stage, scheme="asym_v1"):
        y = torch.zeros((raw_stage.shape[0], 3), dtype=torch.float32, device=raw_stage.device)
        cls = raw_stage.long().clamp(min=0, max=2)
        # stage0: [0.95, 0.05, 0.00]
        m0 = cls == 0
        y[m0, 0] = 0.95
        y[m0, 1] = 0.05
        # stage1: [0.10, 0.80, 0.10]
        m1 = cls == 1
        y[m1, 0] = 0.10
        y[m1, 1] = 0.80
        y[m1, 2] = 0.10
        # stage2: [0.00, 0.05, 0.95]
        m2 = cls == 2
        y[m2, 1] = 0.05
        y[m2, 2] = 0.95
        return y

    def _compute_soft_ce(self, p, y_soft):
        p = p.clamp_min(1e-8)
        return (-(y_soft * torch.log(p)).sum(dim=1)).mean()


    def _v2_soft_joint_factor(self):
        if getattr(self, "lpv3_active", False):
            start_epoch = max(int(getattr(self, "v2_soft_joint_start_epoch", 5) or 5), int(getattr(self, "lpv3_enable_soft_joint_after_epoch", 10) or 10))
        else:
            start_epoch = int(getattr(self, "v2_soft_joint_start_epoch", 5) or 5)
        warmup_epochs = int(getattr(self, "v2_soft_joint_warmup_epochs", 5) or 5)
        if self.current_epoch < start_epoch:
            return 0.0
        if warmup_epochs <= 0:
            return 1.0
        if self.current_epoch >= start_epoch + warmup_epochs:
            return 1.0
        return float(self.current_epoch - start_epoch + 1) / float(warmup_epochs)

    def _graph_distance_matrix(self, device):
        inf = 1e9
        D = torch.full((6, 6), inf, dtype=torch.float32, device=device)
        for i in range(6):
            D[i, i] = 0.0
        edges = [
            (0, 1, float(getattr(self, 'joint_graph_w_00_11', 1.0))),
            (1, 3, float(getattr(self, 'joint_graph_w_11_21', 0.6))),
            (1, 2, float(getattr(self, 'joint_graph_w_11_12', 1.2))),
            (3, 4, float(getattr(self, 'joint_graph_w_21_22', 0.8))),
            (2, 4, float(getattr(self, 'joint_graph_w_12_22', 0.7))),
            (4, 5, float(getattr(self, 'joint_graph_w_22_32', 1.5))),
        ]
        for i, j, w in edges:
            D[i, j] = min(float(D[i, j]), w)
            D[j, i] = min(float(D[j, i]), w)
        for k in range(6):
            D = torch.minimum(D, D[:, k:k+1] + D[k:k+1, :])
        return D

    def _v2_joint_from_outputs(self, outputs, eps=1e-8):
        ge_g = corn_marginal_ge_probs(torch.sigmoid(outputs['grade_logits']))
        ge_s = corn_marginal_ge_probs(torch.sigmoid(outputs['stage_ind_logits']))
        pG = self._build_full_grade_probs_from_ge(ge_g)
        pS = self._build_full_stage_probs_from_ge(ge_s)
        H = -(pG.clamp_min(eps) * torch.log(pG.clamp_min(eps))).sum(dim=1) / np.log(4.0)
        alpha = float(getattr(self, 'alpha_gate_min', 0.15)) + (float(getattr(self, 'alpha_gate_max', 0.65)) - float(getattr(self, 'alpha_gate_min', 0.15))) * H
        alpha = alpha.clamp(min=0.0, max=1.0)
        q1 = torch.sigmoid(outputs['q1_logit'].view(-1))
        q2 = torch.sigmoid(outputs['q2_logit'].view(-1))
        beta = float(getattr(self, 'joint_beta_stage', 0.5))
        gamma = float(getattr(self, 'joint_gamma_cond', 0.5))
        log = torch.log
        pG_c = pG.clamp(eps, 1.0)
        pS_c = pS.clamp(eps, 1.0)
        q1c = q1.clamp(eps, 1.0 - eps)
        q2c = q2.clamp(eps, 1.0 - eps)
        a = alpha
        joint_logits = torch.stack([
            log(pG_c[:, 0]) + a * beta * log(pS_c[:, 0]),
            log(pG_c[:, 1]) + a * beta * log(pS_c[:, 1]) + a * gamma * log(q1c),
            log(pG_c[:, 1]) + a * beta * log(pS_c[:, 2]) + a * gamma * log(1.0 - q1c),
            log(pG_c[:, 2]) + a * beta * log(pS_c[:, 1]) + a * gamma * log(q2c),
            log(pG_c[:, 2]) + a * beta * log(pS_c[:, 2]) + a * gamma * log(1.0 - q2c),
            log(pG_c[:, 3]) + a * beta * log(pS_c[:, 2]),
        ], dim=1)
        P_joint6 = torch.softmax(joint_logits, dim=1)
        P_joint6 = P_joint6 / P_joint6.sum(dim=1, keepdim=True).clamp_min(eps)
        pG_fused = torch.stack([P_joint6[:, 0], P_joint6[:, 1] + P_joint6[:, 2], P_joint6[:, 3] + P_joint6[:, 4], P_joint6[:, 5]], dim=1)
        pS_fused = torch.stack([P_joint6[:, 0], P_joint6[:, 1] + P_joint6[:, 3], P_joint6[:, 2] + P_joint6[:, 4] + P_joint6[:, 5]], dim=1)
        return {
            'grade_ge': ge_g,
            'stage_ge': ge_s,
            'pG_raw4': pG,
            'pS_ind3': pS,
            'q1': q1,
            'q2': q2,
            'alpha': alpha,
            'joint_logits': joint_logits,
            'P_joint6': P_joint6,
            'pG_fused4': pG_fused,
            'pS_fused3': pS_fused,
        }

    def _v2lite_joint_from_outputs(self, outputs, eps=1e-8):
        ge_g = corn_marginal_ge_probs(torch.sigmoid(outputs['grade_logits']))
        ge_s = corn_marginal_ge_probs(torch.sigmoid(outputs['stage_ind_logits']))
        pG = self._build_full_grade_probs_from_ge(ge_g)
        pS = self._build_full_stage_probs_from_ge(ge_s)
        q1 = torch.sigmoid(outputs['q1_logit'].view(-1))
        q2 = torch.sigmoid(outputs['q2_logit'].view(-1))
        beta = float(getattr(self, 'joint_beta_stage', 0.5))
        gamma = float(getattr(self, 'joint_gamma_cond', 0.5))
        pg = pG.clamp(eps, 1.0)
        ps = pS.clamp(eps, 1.0)
        q1c = q1.clamp(eps, 1.0 - eps)
        q2c = q2.clamp(eps, 1.0 - eps)
        joint_logits = torch.stack([
            torch.log(pg[:, 0]) + beta * torch.log(ps[:, 0]),
            torch.log(pg[:, 1]) + beta * torch.log(ps[:, 1]) + gamma * torch.log(q1c),
            torch.log(pg[:, 1]) + beta * torch.log(ps[:, 2]) + gamma * torch.log(1.0 - q1c),
            torch.log(pg[:, 2]) + beta * torch.log(ps[:, 1]) + gamma * torch.log(q2c),
            torch.log(pg[:, 2]) + beta * torch.log(ps[:, 2]) + gamma * torch.log(1.0 - q2c),
            torch.log(pg[:, 3]) + beta * torch.log(ps[:, 2]),
        ], dim=1)
        P_joint6 = torch.softmax(joint_logits, dim=1)
        pG_fused = torch.stack([P_joint6[:, 0], P_joint6[:, 1] + P_joint6[:, 2], P_joint6[:, 3] + P_joint6[:, 4], P_joint6[:, 5]], dim=1)
        pS_fused = torch.stack([P_joint6[:, 0], P_joint6[:, 1] + P_joint6[:, 3], P_joint6[:, 2] + P_joint6[:, 4] + P_joint6[:, 5]], dim=1)
        return {'pG_raw4': pG, 'pS_ind3': pS, 'q1': q1, 'q2': q2, 'joint_logits': joint_logits, 'P_joint6': P_joint6, 'pG_fused4': pG_fused, 'pS_fused3': pS_fused}

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _warmup_factor(self):
        if self.joint_warmup_epochs is None or self.joint_warmup_epochs <= 0:
            return 1.0
        return min(1.0, (self.current_epoch + 1) / float(self.joint_warmup_epochs))

    def _build_joint_probs(self, pG, pS, eps=1e-8, use_prior=False, prior=None, alpha=0.2):
        if use_prior and prior is not None:
            prior = prior.to(pG.device)
            prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(eps)
        scores = []
        for g, s in JOINT_LABELS:
            score = pG[:, g] * pS[:, s]
            if use_prior and prior is not None:
                score = score * (prior[g, s] ** alpha)
            scores.append(score)
        scores = torch.stack(scores, dim=1)
        denom = scores.sum(dim=1, keepdim=True).clamp_min(eps)
        return scores / denom

    def _corn_task_loss(self, logits, targets, pos_weight=None):
        losses = []
        task_count = 0
        num_tasks = logits.shape[1]
        for t in range(num_tasks):
            if t == 0:
                mask = torch.ones_like(targets[:, 0], dtype=torch.bool)
            else:
                mask = targets[:, t - 1] >= 0.5
            if mask.sum() == 0:
                continue
            logits_t = logits[mask, t]
            target_t = targets[mask, t]
            pos_w = pos_weight[t] if pos_weight is not None else None
            loss_t = F.binary_cross_entropy_with_logits(logits_t, target_t, pos_weight=pos_w)
            losses.append(loss_t)
            task_count += 1
        if task_count == 0:
            return torch.tensor(0.0, device=logits.device)
        return sum(losses) / task_count

    def forward(self, outputs, targets):
        if isinstance(outputs, dict) and all(k in outputs for k in ["anyhtn", "grade_pos", "stage_pos"]):
            z_h = outputs["anyhtn"]
            z_g = outputs["grade_pos"]
            z_s = outputs["stage_pos"]
            y_grade = targets["y_grade"]
            raw_grade = targets.get("raw_grade")
            raw_stage = targets.get("raw_stage")
            if raw_grade is None:
                raw_grade = torch.sum(y_grade > 0.5, dim=1).long()
            if raw_stage is None:
                raw_stage = torch.zeros_like(raw_grade)

            y_h = (raw_grade > 0).float().view(-1, 1)
            if self.pos_weight_anyhtn is not None:
                loss_h = F.binary_cross_entropy_with_logits(z_h, y_h, pos_weight=self.pos_weight_anyhtn)
            else:
                loss_h = F.binary_cross_entropy_with_logits(z_h, y_h)
            loss_h_auc = self._compute_pairwise_auc_loss(z_h, y_h)
            loss_h_total = loss_h + float(getattr(self, "loss_w_anyhtn_auc", 0.0) or 0.0) * loss_h_auc

            mask_pos = y_h.view(-1) > 0.5
            mode_soft = str(getattr(self, "fine_soft_label_mode", "none") or "none").lower()
            enable_grade_soft = mode_soft in {"grade_only", "grade_and_stage"}
            enable_stage_smooth = mode_soft == "grade_and_stage"

            if mask_pos.any():
                g_pos = (raw_grade[mask_pos] - 1).long().clamp(min=0, max=2)
                lg = z_g[mask_pos]
                t1 = (g_pos >= 1).float().view(-1, 1)
                t2 = (g_pos >= 2).float().view(-1, 1)
                loss_g_corn = F.binary_cross_entropy_with_logits(lg[:, :1], t1)
                loss_g_corn = loss_g_corn + F.binary_cross_entropy_with_logits(lg[:, 1:2], t2)

                loss_g_soft = lg.sum() * 0.0
                if enable_grade_soft:
                    p_grade_pos = self._build_positive_grade_probs_from_corn(lg)
                    y_soft = self._build_grade_soft_targets(raw_grade[mask_pos], getattr(self, "grade_soft_center", 0.85))
                    loss_g_soft = self._compute_grade_soft_loss(p_grade_pos, y_soft)
                loss_g = loss_g_corn + float(getattr(self, "loss_w_grade_soft", 0.2) or 0.0) * loss_g_soft

                s_pos = (raw_stage[mask_pos] - 1).long().clamp(min=0, max=1)
                ls = z_s[mask_pos]
                stage_target = s_pos.float().view(-1, 1)
                if enable_stage_smooth:
                    stage_target = self._smooth_stage_targets(stage_target, getattr(self, "stage_label_smoothing", 0.05))
                loss_s = F.binary_cross_entropy_with_logits(ls.view(-1, 1), stage_target)
                loss_s = float(getattr(self, "loss_w_stage_smooth", 1.0) or 1.0) * loss_s
            else:
                loss_g_corn = z_g.sum() * 0.0
                loss_g_soft = z_g.sum() * 0.0
                loss_g = z_g.sum() * 0.0
                loss_s = z_s.sum() * 0.0

            loss = self.w_anyhtn * loss_h_total + self.w_grade * loss_g + self.w_stage * loss_s
            self.last_components = {
                "loss_anyhtn": float(loss_h.detach().cpu()),
                "loss_anyhtn_auc": float(loss_h_auc.detach().cpu()),
                "loss_anyhtn_total": float(loss_h_total.detach().cpu()),
                "loss_grade": float(loss_g.detach().cpu()),
                "loss_grade_corn": float(loss_g_corn.detach().cpu()),
                "loss_grade_soft": float(loss_g_soft.detach().cpu()),
                "loss_grade_total": float(loss_g.detach().cpu()),
                "stage_smooth_enabled": float(1.0 if enable_stage_smooth else 0.0),
                "loss_stage": float(loss_s.detach().cpu()),
                "loss_total": float(loss.detach().cpu()),
            }
            return loss

        if isinstance(outputs, dict) and all(k in outputs for k in ["grade_logits", "stage_ind_logits", "q1_logit", "q2_logit"]):
            y_grade = targets["y_grade"]
            y_stage = targets["y_stage"]
            raw_grade = targets.get("raw_grade")
            raw_stage = targets.get("raw_stage")
            if raw_grade is None:
                raw_grade = torch.sum(y_grade > 0.5, dim=1).long()
            if raw_stage is None:
                raw_stage = torch.sum(y_stage > 0.5, dim=1).long()
            is_v2lite = str(getattr(self, "data_set", "")).lower() == "advchex_hyp_grade_stage_embtab_v2lite"
            fused = self._v2lite_joint_from_outputs(outputs) if is_v2lite else self._v2_joint_from_outputs(outputs)
            loss_grade_base = self._corn_task_loss(outputs['grade_logits'], y_grade, pos_weight=self.loss_grade.pos_weight)
            loss_grade_soft = loss_grade_base.new_tensor(0.0)
            if str(getattr(self, 'v1_soft_label_mode', 'none') or 'none').lower() == 'full':
                y_grade_soft = self._build_v1_grade_soft_targets(raw_grade, getattr(self, 'grade_soft_scheme', 'asym_v1'))
                loss_grade_soft = self._compute_soft_ce(fused['pG_raw4'], y_grade_soft)
            loss_grade = loss_grade_base + float(getattr(self, 'loss_w_grade_soft', 0.2) or 0.0) * loss_grade_soft
            loss_stage_marg_ind = self._corn_task_loss(outputs['stage_ind_logits'], y_stage, pos_weight=self.loss_stage.pos_weight)
            loss_stage_marg_fused = F.nll_loss(torch.log(fused['pS_fused3'].clamp_min(1e-8)), raw_stage.long())

            zero = loss_grade_base.new_tensor(0.0)
            cond_enabled = (not getattr(self, 'lpv3_active', False)) or (self.current_epoch >= int(getattr(self, 'lpv3_enable_cond_after_epoch', 3) or 3))
            mask_g1 = raw_grade == 1
            mask_g2 = raw_grade == 2
            if cond_enabled and mask_g1.any():
                z1 = (raw_stage[mask_g1] == 1).float()
                pos_w1 = torch.tensor(float(getattr(self, 'cond_pos_weight_g1', 3.0)), device=z1.device)
                loss_cond_11_12 = F.binary_cross_entropy_with_logits(outputs['q1_logit'][mask_g1].view(-1), z1, pos_weight=pos_w1)
            else:
                loss_cond_11_12 = zero
            if cond_enabled and mask_g2.any():
                z2 = (raw_stage[mask_g2] == 1).float()
                pos_w2 = torch.tensor(float(getattr(self, 'cond_pos_weight_g2', 5.0)), device=z2.device)
                loss_cond_21_22 = F.binary_cross_entropy_with_logits(outputs['q2_logit'][mask_g2].view(-1), z2, pos_weight=pos_w2)
            else:
                loss_cond_21_22 = zero

            D = self._graph_distance_matrix(fused['P_joint6'].device)
            tau = float(getattr(self, 'joint_graph_tau', 0.7))
            if is_v2lite:
                loss_soft_joint = F.nll_loss(torch.log(fused['P_joint6'].clamp_min(1e-8)), targets['joint_id'].long())
                lambda_soft_joint_eff = float(getattr(self, 'lambda_joint_soft', 0.05) or 0.0)
                expected_cost = (fused['P_joint6'] * D[targets['joint_id'].long()]).sum(dim=1).mean()
                loss = loss_grade + float(getattr(self, 'lambda_stage_marg', 1.0)) * loss_stage_marg_ind + float(getattr(self, 'lambda_cond', 0.5)) * (loss_cond_11_12 + loss_cond_21_22) + lambda_soft_joint_eff * loss_soft_joint
            else:
                soft_targets = torch.softmax(-D[targets['joint_id'].long()] / max(tau, 1e-6), dim=1)
                loss_soft_joint = (-(soft_targets * torch.log(fused['P_joint6'].clamp_min(1e-8))).sum(dim=1)).mean()
                expected_cost = (fused['P_joint6'] * D[targets['joint_id'].long()]).sum(dim=1).mean()
                lambda_soft_joint_eff = float(getattr(self, 'lambda_soft_joint', 0.15) or 0.0) * self._v2_soft_joint_factor()
                if getattr(self, 'lpv3_active', False) and self.current_epoch < int(getattr(self, 'lpv3_enable_soft_joint_after_epoch', 10) or 10):
                    lambda_soft_joint_eff = 0.0
                loss = loss_grade + float(getattr(self, 'lambda_stage_marg', 0.8)) * (loss_stage_marg_ind + float(getattr(self, 'stage_fused_aux_weight', 0.3)) * loss_stage_marg_fused) + float(getattr(self, 'lambda_cond_stage', 0.6)) * (loss_cond_11_12 + loss_cond_21_22) + lambda_soft_joint_eff * loss_soft_joint
            feature_before = outputs.get('features_before_neck')
            feature_after = outputs.get('shared_features')
            mean_feature_norm_before = float(feature_before.norm(dim=1).mean().detach().cpu()) if isinstance(feature_before, torch.Tensor) else 0.0
            mean_feature_norm_after = float(feature_after.norm(dim=1).mean().detach().cpu()) if isinstance(feature_after, torch.Tensor) else 0.0
            joint_ids = targets['joint_id'].long()
            present = torch.bincount(joint_ids, minlength=len(JOINT_LABELS)) > 0
            self.last_components = {
                'loss_grade_main': float(loss_grade_base.detach().cpu()),
                'loss_grade_soft': float(loss_grade_soft.detach().cpu()),
                'loss_stage_marg_ind': float(loss_stage_marg_ind.detach().cpu()),
                'loss_stage_marg_fused': float(loss_stage_marg_fused.detach().cpu()),
                'loss_cond_11_12': float(loss_cond_11_12.detach().cpu()),
                'loss_cond_21_22': float(loss_cond_21_22.detach().cpu()),
                'loss_soft_joint': float(loss_soft_joint.detach().cpu()),
                'mean_alpha_gate': float(fused['alpha'].mean().detach().cpu()) if 'alpha' in fused else 1.0,
                'mean_q1': float(fused['q1'].mean().detach().cpu()),
                'mean_q2': float(fused['q2'].mean().detach().cpu()),
                'mean_expected_joint_graph_cost': float(expected_cost.detach().cpu()),
                'lambda_soft_joint_eff': float(lambda_soft_joint_eff),
                'mean_gate_g': float(outputs['gate_g'].mean().detach().cpu()) if 'gate_g' in outputs else 0.0,
                'mean_gate_s': float(outputs['gate_s'].mean().detach().cpu()) if 'gate_s' in outputs else 0.0,
                'mean_feature_norm_before_neck': mean_feature_norm_before,
                'mean_feature_norm_after_neck': mean_feature_norm_after,
                'mean_neck_norm': mean_feature_norm_after,
                'batch_joint_present_classes': float(present.sum().detach().cpu()),
                'batch_has_11_ratio': float(present[JOINT_LABEL_TO_INDEX[(1, 1)]].float().detach().cpu()),
                'batch_has_21_ratio': float(present[JOINT_LABEL_TO_INDEX[(2, 1)]].float().detach().cpu()),
                'batch_has_32_ratio': float(present[JOINT_LABEL_TO_INDEX[(3, 2)]].float().detach().cpu()),
                'cond_enabled': float(cond_enabled),
                'loss_total': float(loss.detach().cpu()),
            }
            return loss

        logits_grade, logits_stage = outputs
        y_grade = targets["y_grade"]
        y_stage = targets["y_stage"]
        if str(self.ordinal_mode).lower() == "corn":
            loss_grade_base = self._corn_task_loss(logits_grade, y_grade, pos_weight=self.loss_grade.pos_weight)
            loss_stage_base = self._corn_task_loss(logits_stage, y_stage, pos_weight=self.loss_stage.pos_weight)
            ge_g = corn_marginal_ge_probs(torch.sigmoid(logits_grade))
            ge_s = corn_marginal_ge_probs(torch.sigmoid(logits_stage))
        else:
            loss_grade_base = self.loss_grade(logits_grade, y_grade)
            loss_stage_base = self.loss_stage(logits_stage, y_stage)
            ge_g = torch.sigmoid(logits_grade)
            ge_s = torch.sigmoid(logits_stage)

        loss_grade_soft = loss_grade_base.new_tensor(0.0)
        loss_stage_soft = loss_stage_base.new_tensor(0.0)
        if str(getattr(self, "v1_soft_label_mode", "none") or "none").lower() == "full":
            raw_grade = targets.get("raw_grade")
            raw_stage = targets.get("raw_stage")
            if raw_grade is None:
                raw_grade = torch.sum(y_grade > 0.5, dim=1).long()
            if raw_stage is None:
                raw_stage = torch.sum(y_stage > 0.5, dim=1).long()
            pG_full = self._build_full_grade_probs_from_ge(ge_g)
            pS_full = self._build_full_stage_probs_from_ge(ge_s)
            y_grade_soft = self._build_v1_grade_soft_targets(raw_grade, getattr(self, "grade_soft_scheme", "asym_v1"))
            y_stage_soft = self._build_v1_stage_soft_targets(raw_stage, getattr(self, "stage_soft_scheme", "asym_v1"))
            loss_grade_soft = self._compute_soft_ce(pG_full, y_grade_soft)
            loss_stage_soft = self._compute_soft_ce(pS_full, y_stage_soft)

        loss_grade = loss_grade_base + float(getattr(self, "loss_w_grade_soft", 0.2) or 0.0) * loss_grade_soft
        loss_stage = loss_stage_base + float(getattr(self, "loss_w_stage_soft", 0.1) or 0.0) * loss_stage_soft
        loss = self.w_grade * loss_grade + self.w_stage * loss_stage
        loss_incomp = torch.tensor(0.0, device=loss.device)
        loss_joint = torch.tensor(0.0, device=loss.device)

        mean_p_joint_true = torch.tensor(0.0, device=loss.device)
        mean_neglog_p_joint_true = torch.tensor(0.0, device=loss.device)
        if self.use_joint_train and (self.lambda_incomp > 0 or self.lambda_joint > 0):
            # ge_g/ge_s already computed above for both CORN/CORAL branches
            pG = self._build_full_grade_probs_from_ge(ge_g)
            pS = self._build_full_stage_probs_from_ge(ge_s)
            pG = pG.detach() if self.joint_detach == "grade" else pG
            pS = pS.detach() if self.joint_detach == "stage" else pS

            raw_grade = targets.get("raw_grade")
            gate_mask = None
            if self.joint_gate == "htn_only" and raw_grade is not None:
                gate_mask = raw_grade > 0
            if gate_mask is not None and gate_mask.sum() == 0:
                loss_incomp = torch.tensor(0.0, device=loss.device)
                loss_joint = torch.tensor(0.0, device=loss.device)
            else:
                if gate_mask is not None:
                    pG = pG[gate_mask]
                    pS = pS[gate_mask]
                outer = pG[:, :, None] * pS[:, None, :]
                compat_mask = torch.zeros((4, 3), device=outer.device)
                for g, s in JOINT_LABELS:
                    compat_mask[g, s] = 1.0
                compat_mass = (outer * compat_mask).sum(dim=(1, 2))
                incompat_mass = (outer * (1 - compat_mask)).sum(dim=(1, 2))
                if self.incomp_mode == "log_barrier":
                    loss_incomp = (-torch.log(compat_mass.clamp_min(1e-6))).mean()
                else:
                    loss_incomp = incompat_mass.mean()

                joint_ids = targets.get("joint_id")
                if joint_ids is not None:
                    if gate_mask is not None:
                        joint_ids = joint_ids[gate_mask]
                    if self.joint_detach == "both":
                        pG_a = pG.detach()
                        pS_a = pS
                        pG_b = pG
                        pS_b = pS.detach()
                        P_joint_a = self._build_joint_probs(
                            pG_a, pS_a, use_prior=self.joint_loss_use_prior,
                            prior=self.joint_prior, alpha=self.joint_prior_alpha
                        )
                        P_joint_b = self._build_joint_probs(
                            pG_b, pS_b, use_prior=self.joint_loss_use_prior,
                            prior=self.joint_prior, alpha=self.joint_prior_alpha
                        )
                        log_a = torch.log(P_joint_a.clamp_min(1e-8))
                        log_b = torch.log(P_joint_b.clamp_min(1e-8))
                        loss_joint = 0.5 * self.loss_joint(log_a, joint_ids) + 0.5 * self.loss_joint(log_b, joint_ids)
                        P_joint_mean = 0.5 * (P_joint_a + P_joint_b)
                        p_true = P_joint_mean.gather(1, joint_ids.unsqueeze(1)).squeeze(1)
                    else:
                        P_joint = self._build_joint_probs(
                            pG, pS, use_prior=self.joint_loss_use_prior,
                            prior=self.joint_prior, alpha=self.joint_prior_alpha
                        )
                        log_p = torch.log(P_joint.clamp_min(1e-8))
                        loss_joint = self.loss_joint(log_p, joint_ids)
                        p_true = P_joint.gather(1, joint_ids.unsqueeze(1)).squeeze(1)
                    mean_p_joint_true = p_true.mean()
                    mean_neglog_p_joint_true = (-torch.log(p_true.clamp_min(1e-8))).mean()

            warmup = self._warmup_factor()
            loss = loss + warmup * (self.lambda_incomp * loss_incomp + self.lambda_joint * loss_joint)

        self.last_components = {
            "loss_grade": float(loss_grade.detach().cpu().item()),
            "loss_stage": float(loss_stage.detach().cpu().item()),
            "loss_grade_base": float(loss_grade_base.detach().cpu().item()),
            "loss_stage_base": float(loss_stage_base.detach().cpu().item()),
            "loss_grade_soft": float(loss_grade_soft.detach().cpu().item()),
            "loss_stage_soft": float(loss_stage_soft.detach().cpu().item()),
            "loss_incomp": float(loss_incomp.detach().cpu().item()),
            "loss_joint": float(loss_joint.detach().cpu().item()),
            "loss_total": float(loss.detach().cpu().item()),
            "mean_p_joint_true": float(mean_p_joint_true.detach().cpu().item()),
            "mean_neglog_p_joint_true": float(mean_neglog_p_joint_true.detach().cpu().item()),
        }
        return loss


def _parse_pos_weight(args, dataset_train, device):
    if args.pos_weight:
        parts = [float(x) for x in args.pos_weight.split(',')]
        return torch.tensor(parts, dtype=torch.float32, device=device)
    # 自动统计
    pos = np.zeros(args.num_class, dtype=np.float32)
    N = len(dataset_train)
    for lab in dataset_train.img_label:
        pos += np.array(lab, dtype=np.float32)
    neg = np.maximum(N - pos, 1.0)
    pos_safe = np.maximum(pos, 1.0)
    pw = torch.tensor(neg / pos_safe, dtype=torch.float32, device=device)
    return pw


def _parse_pos_weight_multi(args, dataset_train, device):
    def _from_list(values, length):
        if values is None:
            return None
        parts = [float(x) for x in values.split(',')]
        if len(parts) != length:
            raise ValueError(f"pos_weight 长度应为 {length}，但得到 {len(parts)}")
        return torch.tensor(parts, dtype=torch.float32, device=device)

    pos_weight_grade = _from_list(getattr(args, "pos_weight_grade", None), 3)
    pos_weight_stage = _from_list(getattr(args, "pos_weight_stage", None), 2)

    if pos_weight_grade is None:
        pos = np.zeros(3, dtype=np.float32)
        N = len(dataset_train.grade_labels)
        for lab in dataset_train.grade_labels:
            pos += np.array(lab, dtype=np.float32)
        neg = np.maximum(N - pos, 1.0)
        pos_safe = np.maximum(pos, 1.0)
        pos_weight_grade = torch.tensor(neg / pos_safe, dtype=torch.float32, device=device)

    if pos_weight_stage is None:
        pos = np.zeros(2, dtype=np.float32)
        N = len(dataset_train.stage_labels)
        for lab in dataset_train.stage_labels:
            pos += np.array(lab, dtype=np.float32)
        neg = np.maximum(N - pos, 1.0)
        pos_safe = np.maximum(pos, 1.0)
        pos_weight_stage = torch.tensor(neg / pos_safe, dtype=torch.float32, device=device)

    return pos_weight_grade, pos_weight_stage


def _compute_joint_class_weights(dataset_train, mode="inv_sqrt", device=None):
    if mode == "none":
        return None
    counts = np.zeros(len(JOINT_LABELS), dtype=np.float32)
    for jid in getattr(dataset_train, "joint_ids", []):
        counts[int(jid)] += 1.0
    counts = np.maximum(counts, 1.0)
    if mode == "inv":
        weights = 1.0 / counts
    elif mode == "inv_sqrt":
        weights = 1.0 / np.sqrt(counts)
    else:
        weights = np.ones_like(counts)
    weights = weights / np.mean(weights)
    tensor = torch.tensor(weights, dtype=torch.float32, device=device) if device is not None else torch.tensor(weights)
    return tensor


def _collect_outputs_multi(model, data_loader, device, ordinal_mode="default"):
    model.eval()
    y_grade_all = torch.FloatTensor().to(device)
    y_stage_all = torch.FloatTensor().to(device)
    p_grade_all = torch.FloatTensor().to(device)
    p_stage_all = torch.FloatTensor().to(device)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if batch is None:
                continue
            samples, targets = batch
            if isinstance(samples, dict):
                samples = {k: v.float().to(device) if torch.is_tensor(v) else v for k, v in samples.items()}
            else:
                samples = samples.float().to(device)
            y_grade = targets["y_grade"].float().to(device)
            y_stage = targets["y_stage"].float().to(device)
            y_grade_all = torch.cat((y_grade_all, y_grade), 0)
            y_stage_all = torch.cat((y_stage_all, y_stage), 0)
            out = model(samples)
            if isinstance(out, dict) and all(k in out for k in ["anyhtn", "grade_pos", "stage_pos"]):
                pH = torch.sigmoid(out["anyhtn"]).view(-1, 1)
                a = corn_marginal_ge_probs(torch.sigmoid(out["grade_pos"]))
                b = torch.sigmoid(out["stage_pos"]).view(-1, 1)
                p_grade = torch.cat([pH, pH * a[:, :1], pH * a[:, 1:2]], dim=1)
                p_stage = torch.cat([pH, pH * b], dim=1)
            elif isinstance(out, dict) and all(k in out for k in ["grade_logits", "stage_ind_logits", "q1_logit", "q2_logit"]):
                logits_grade = out["grade_logits"]
                logits_stage = out["stage_ind_logits"]
                if str(ordinal_mode).lower() == "corn":
                    p_grade = corn_marginal_ge_probs(torch.sigmoid(logits_grade))
                    p_stage = corn_marginal_ge_probs(torch.sigmoid(logits_stage))
                else:
                    p_grade = torch.sigmoid(logits_grade)
                    p_stage = torch.sigmoid(logits_stage)
            elif isinstance(out, tuple) and len(out) == 2:
                logits_grade, logits_stage = out
                if str(ordinal_mode).lower() == "corn":
                    q_grade = torch.sigmoid(logits_grade)
                    q_stage = torch.sigmoid(logits_stage)
                    p_grade = corn_marginal_ge_probs(q_grade)
                    p_stage = corn_marginal_ge_probs(q_stage)
                else:
                    p_grade = torch.sigmoid(logits_grade)
                    p_stage = torch.sigmoid(logits_stage)
            else:
                raise TypeError(f"_collect_outputs_multi 不支持的模型输出类型: {type(out)}")
            p_grade_all = torch.cat((p_grade_all, p_grade), 0)
            p_stage_all = torch.cat((p_stage_all, p_stage), 0)
    return (
        y_grade_all.cpu().numpy(),
        y_stage_all.cpu().numpy(),
        p_grade_all.cpu().numpy(),
        p_stage_all.cpu().numpy(),
    )


def _collect_outputs_multi_v2lite(model, data_loader, device, joint_beta_stage=0.5, joint_gamma_cond=0.5):
    model.eval()
    y_grade_all, y_stage_all = [], []
    p_grade_ge_all, p_stage_ge_all = [], []
    p_joint6_all, pG_fused_all, pS_fused_all = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if batch is None:
                continue
            samples, targets = batch
            if isinstance(samples, dict):
                samples = {k: v.float().to(device) if torch.is_tensor(v) else v for k, v in samples.items()}
            else:
                samples = samples.float().to(device)
            out = model(samples)
            if not (isinstance(out, dict) and all(k in out for k in ["grade_logits", "stage_ind_logits", "q1_logit", "q2_logit"])):
                raise TypeError("v2lite 验证期期望 dict 输出且包含 grade/stage/q1/q2")
            p_grade_ge = corn_marginal_ge_probs(torch.sigmoid(out["grade_logits"]))
            p_stage_ge = corn_marginal_ge_probs(torch.sigmoid(out["stage_ind_logits"]))
            joint = compose_v2_joint_predictions(
                p_grade_ge, p_stage_ge, out["q1_logit"], out["q2_logit"],
                joint_beta_stage=float(joint_beta_stage), joint_gamma_cond=float(joint_gamma_cond), use_entropy_alpha=False,
            )
            y_grade_all.append(targets["y_grade"].cpu())
            y_stage_all.append(targets["y_stage"].cpu())
            p_grade_ge_all.append(p_grade_ge.cpu())
            p_stage_ge_all.append(p_stage_ge.cpu())
            p_joint6_all.append(joint["P_joint6"].cpu())
            pG_fused_all.append(joint["pG_fused4"].cpu())
            pS_fused_all.append(joint["pS_fused3"].cpu())
    return {
        "y_grade": torch.cat(y_grade_all, dim=0).numpy(),
        "y_stage": torch.cat(y_stage_all, dim=0).numpy(),
        "p_grade_ge": torch.cat(p_grade_ge_all, dim=0).numpy(),
        "p_stage_ge": torch.cat(p_stage_ge_all, dim=0).numpy(),
        "p_joint6": torch.cat(p_joint6_all, dim=0).numpy(),
        "pG_fused": torch.cat(pG_fused_all, dim=0).numpy(),
        "pS_fused": torch.cat(pS_fused_all, dim=0).numpy(),
    }


def _load_ordinal_thresholds(saved_model, args):
    if args.thresholds_json and os.path.exists(args.thresholds_json):
        return load_thresholds_json(args.thresholds_json)

    meta_path = saved_model.replace('.pth.tar', '_meta.json') if isinstance(saved_model, str) else None
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            th_json = meta.get('thresholds_json')
            if th_json:
                candidate = th_json if os.path.isabs(th_json) else os.path.join(os.path.dirname(meta_path), th_json)
                if os.path.exists(candidate):
                    return load_thresholds_json(candidate)
        except Exception:
            pass

    default_json = saved_model.replace('.pth.tar', '_thresholds.json') if isinstance(saved_model, str) else None
    if default_json and os.path.exists(default_json):
        return load_thresholds_json(default_json)

    if isinstance(saved_model, str) and os.path.exists(saved_model):
        try:
            ckpt_loaded = torch.load(saved_model, weights_only=False)
            if isinstance(ckpt_loaded, dict):
                return ckpt_loaded.get('ordinal_thresholds')
        except Exception:
            pass
    return None

def _is_lpv3_active(args):
  return bool(getattr(args, "data_set", "") == "advCheX_hyp_grade_stage_v2" and getattr(args, "lpv3_enable_neck", False))


def _build_lpv3_config(args):
  return {
    "lpv3_enable_neck": bool(getattr(args, "lpv3_enable_neck", False)),
    "lpv3_neck_hidden_dim": int(getattr(args, "lpv3_neck_hidden_dim", 512)),
    "lpv3_neck_out_dim": int(getattr(args, "lpv3_neck_out_dim", 128)),
    "lpv3_neck_dropout": float(getattr(args, "lpv3_neck_dropout", 0.2)),
    "lpv3_joint_aware_sampler": bool(getattr(args, "lpv3_joint_aware_sampler", False)),
    "lpv3_sampler_mode": str(getattr(args, "lpv3_sampler_mode", "joint_inv_freq")),
    "lpv3_sampler_power": float(getattr(args, "lpv3_sampler_power", 0.5)),
    "lpv3_sampler_cap": float(getattr(args, "lpv3_sampler_cap", 5.0)),
    "lpv3_sampler_floor": float(getattr(args, "lpv3_sampler_floor", 1.0)),
    "lpv3_sampler_boost_11": float(getattr(args, "lpv3_sampler_boost_11", 2.0)),
    "lpv3_sampler_boost_21": float(getattr(args, "lpv3_sampler_boost_21", 4.0)),
    "lpv3_sampler_boost_32": float(getattr(args, "lpv3_sampler_boost_32", 1.5)),
    "lpv3_sampler_boost_12": float(getattr(args, "lpv3_sampler_boost_12", 1.0)),
    "lpv3_sampler_boost_22": float(getattr(args, "lpv3_sampler_boost_22", 1.0)),
    "lpv3_stageA_epochs": int(getattr(args, "lpv3_stageA_epochs", 5)),
    "lpv3_enable_cond_after_epoch": int(getattr(args, "lpv3_enable_cond_after_epoch", 3)),
    "lpv3_enable_soft_joint_after_epoch": int(getattr(args, "lpv3_enable_soft_joint_after_epoch", 10)),
  }


def _build_joint_aware_sampling(dataset_train, args, base_weights=None):
  joint_ids = np.array(getattr(dataset_train, "joint_ids", []), dtype=np.int64)
  if joint_ids.size == 0:
    return None
  counts = np.bincount(joint_ids, minlength=len(JOINT_LABELS)).astype(np.float64)
  counts_safe = np.maximum(counts, 1.0)
  power = float(getattr(args, "lpv3_sampler_power", 0.5) or 0.5)
  floor = float(getattr(args, "lpv3_sampler_floor", 1.0) or 1.0)
  cap = float(getattr(args, "lpv3_sampler_cap", 5.0) or 5.0)
  base_joint = (1.0 / counts_safe) ** power
  boost_map = {
    JOINT_LABEL_TO_INDEX[(1, 1)]: float(getattr(args, "lpv3_sampler_boost_11", 2.0) or 1.0),
    JOINT_LABEL_TO_INDEX[(1, 2)]: float(getattr(args, "lpv3_sampler_boost_12", 1.0) or 1.0),
    JOINT_LABEL_TO_INDEX[(2, 1)]: float(getattr(args, "lpv3_sampler_boost_21", 4.0) or 1.0),
    JOINT_LABEL_TO_INDEX[(2, 2)]: float(getattr(args, "lpv3_sampler_boost_22", 1.0) or 1.0),
    JOINT_LABEL_TO_INDEX[(3, 2)]: float(getattr(args, "lpv3_sampler_boost_32", 1.5) or 1.0),
  }
  for idx, boost in boost_map.items():
    base_joint[idx] *= boost
  joint_weights = np.clip(base_joint[joint_ids], floor, cap)
  final_weights = joint_weights.copy()
  if base_weights is not None:
    final_weights = np.asarray(base_weights, dtype=np.float64) * final_weights
  replacement = True
  summary = {
    "enabled": True,
    "replacement": replacement,
    "joint_counts": {f"{g}{s}": int(counts[idx]) for idx, (g, s) in enumerate(JOINT_LABELS)},
    "mean_joint_weight": {},
    "final_weight_min": float(final_weights.min()) if final_weights.size else 0.0,
    "final_weight_max": float(final_weights.max()) if final_weights.size else 0.0,
    "final_weight_mean": float(final_weights.mean()) if final_weights.size else 0.0,
  }
  for idx, (g, s) in enumerate(JOINT_LABELS):
    mask = joint_ids == idx
    summary["mean_joint_weight"][f"{g}{s}"] = float(final_weights[mask].mean()) if mask.any() else 0.0
  return final_weights.tolist(), summary


def classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test, test_diseases=None):
  sampler_summary = None
  device = torch.device(args.device)
  cudnn.benchmark = True

  model_path = os.path.join(model_path, args.exp_name)

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  if not os.path.exists(output_path):
    os.makedirs(output_path)
  output_file = os.path.join(output_path, args.exp_name + "_results.txt")

  ordinal_datasets = {"advCheX_hyp_multi_level", "advCheX_hyp_multi_stage_v1", "advCheX_hyp_multi_stage_v2"}
  multihead_datasets = {"advCheX_hyp_multi_grade_stage_v1", "advCheX_hyp_multi_grade_stage_sep_v1", "advCheX_hyp_grade_stage_v2", "advCheX_hyp_grade_stage_embtab_base", "advCheX_hyp_grade_stage_embtab_v2lite"}
  if args.data_set in ordinal_datasets and (getattr(args, "test_time_adjust", False) or getattr(args, "output_special", False)):
    if hasattr(dataset_test, "return_path"):
      dataset_test.return_path = True
  if args.data_set == "advCheX_hyp_grade_stage_embtab_v2lite" and bool(getattr(args, "return_path", False)):
    if hasattr(dataset_test, "return_path"):
      dataset_test.return_path = True

  data_loader_test = DataLoader(dataset=dataset_test, batch_size=int(args.batch_size/2), shuffle=False,
                            num_workers=args.workers, pin_memory=True, collate_fn=safe_collate, persistent_workers=False)
  ordinal_thresholds = None
  # training phase
  if args.mode == "train":
    train_weights_path = args.train_weights
    if args.data_set in {"advCheX_hyp_multi_stage_v1", "advCheX_hyp_multi_stage_v2"} and train_weights_path is None:
      candidate = os.path.join(args.data_dir, "train_weights.csv")
      if os.path.exists(candidate):
        train_weights_path = candidate
    base_weights = None
    sampler_summary = None
    if train_weights_path is not None:
      df_w = pd.read_csv(train_weights_path)
      weight_col = "sample_weight" if "sample_weight" in df_w.columns else "weight"
      weight_map = dict(zip(df_w['Path'], df_w[weight_col]))
      rel_paths = [os.path.relpath(p, args.data_dir) for p in dataset_train.img_list]
      base_weights = [weight_map.get(rp, 1.0) for rp in rel_paths]
    use_lpv3_sampler = bool(getattr(args, 'data_set', '') == 'advCheX_hyp_grade_stage_v2' and getattr(args, 'lpv3_joint_aware_sampler', False))
    if use_lpv3_sampler:
      joint_sampler_pack = _build_joint_aware_sampling(dataset_train, args, base_weights=base_weights)
      if joint_sampler_pack is not None:
        final_weights, sampler_summary = joint_sampler_pack
        sampler = WeightedRandomSampler(final_weights, num_samples=len(final_weights), replacement=True)
        data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, sampler=sampler,
                                       num_workers=args.workers, pin_memory=True, collate_fn=safe_collate, persistent_workers=False)
      else:
        final_weights = None
        data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.workers, pin_memory=True, collate_fn=safe_collate, persistent_workers=False)
    elif base_weights is not None:
      sampler = WeightedRandomSampler(base_weights, num_samples=len(base_weights), replacement=True)
      data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, sampler=sampler,
                                     num_workers=args.workers, pin_memory=True, collate_fn=safe_collate, persistent_workers=False)
    else:
      data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers, pin_memory=True, collate_fn=safe_collate, persistent_workers=False)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
                           
    log_file = os.path.join(model_path, "models.log")

    # training phase
    print("start training....", flush=True)
    lpv3_config = _build_lpv3_config(args)
    if sampler_summary is not None:
      print(f"[LPv3][Sampler] mode={lpv3_config['lpv3_sampler_mode']} power={lpv3_config['lpv3_sampler_power']} floor={lpv3_config['lpv3_sampler_floor']} cap={lpv3_config['lpv3_sampler_cap']} replacement={sampler_summary['replacement']}", flush=True)
      print(f"[LPv3][Sampler] train_joint_counts={sampler_summary['joint_counts']}", flush=True)
      print(f"[LPv3][Sampler] mean_joint_weight={sampler_summary['mean_joint_weight']} final_weight_stats=min:{sampler_summary['final_weight_min']:.4f} mean:{sampler_summary['final_weight_mean']:.4f} max:{sampler_summary['final_weight_max']:.4f}", flush=True)
    for i in range(args.start_index, args.num_trial):
      print("run:", str(i+1), flush=True)
      start_epoch = 0
      init_loss = 1000000
      experiment = args.exp_name + "_run_" + str(i)
      best_val_loss = init_loss
      patience_counter = 0
      save_model_path = os.path.join(model_path, experiment)
      if args.data_set in ordinal_datasets:
        pos_weight = _parse_pos_weight(args, dataset_train, device)
        criterion = WeightedOrdinalCrossEntropy(pos_weight=pos_weight)
        print(f"use WeightedOrdinalCrossEntropy, pos_weight={pos_weight}", flush=True)
      elif args.data_set in multihead_datasets:
        pos_weight_grade, pos_weight_stage = _parse_pos_weight_multi(args, dataset_train, device)
        joint_ce_weight = _compute_joint_class_weights(
          dataset_train,
          mode=getattr(args, "joint_ce_weight_mode", "inv_sqrt"),
          device=device,
        )
        joint_prior = None
        if getattr(args, "joint_loss_use_prior", False):
          joint_prior = build_joint_prior_mimic(
            np.array(dataset_train.grade_list),
            np.array(dataset_train.stage_list),
            eps=getattr(args, "joint_prior_eps", 1e-3),
          )
          joint_prior = torch.tensor(joint_prior, dtype=torch.float32, device=device)
        criterion = MultiHeadOrdinalLoss(
          pos_weight_grade=pos_weight_grade,
          pos_weight_stage=pos_weight_stage,
          w_grade=getattr(args, "loss_w_grade", 1.0),
          w_stage=getattr(args, "loss_w_stage", 1.0),
          use_joint_train=(getattr(args, "use_joint_train", False) if args.data_set != "advCheX_hyp_multi_grade_stage_sep_v1" else False),
          lambda_incomp=(getattr(args, "lambda_incomp", 0.0) if args.data_set != "advCheX_hyp_multi_grade_stage_sep_v1" else 0.0),
          lambda_joint=(getattr(args, "lambda_joint", 0.0) if args.data_set != "advCheX_hyp_multi_grade_stage_sep_v1" else 0.0),
          joint_gate=getattr(args, "joint_gate", "htn_only"),
          joint_detach=getattr(args, "joint_detach", "both"),
          joint_ce_weight=joint_ce_weight,
          joint_warmup_epochs=getattr(args, "joint_warmup_epochs", 5),
          incomp_mode=getattr(args, "incomp_mode", "mask_sum"),
          joint_loss_use_prior=getattr(args, "joint_loss_use_prior", False),
          joint_prior=joint_prior,
          joint_prior_alpha=getattr(args, "joint_prior_alpha", 0.2),
          ordinal_mode=getattr(args, "ordinal_mode", "coral").upper(),
        )
        criterion.w_anyhtn = getattr(args, "loss_w_anyhtn", 1.0)
        criterion.coarse_auc_loss_mode = str(getattr(args, "coarse_auc_loss_mode", "none") or "none").lower()
        criterion.loss_w_anyhtn_auc = float(getattr(args, "loss_w_anyhtn_auc", 0.0) or 0.0)
        criterion.auc_margin = float(getattr(args, "auc_margin", 1.0) or 1.0)
        criterion.auc_pair_subsample = int(getattr(args, "auc_pair_subsample", 256) or 256)
        criterion.auc_loss_detach_probs = bool(getattr(args, "auc_loss_detach_probs", False))
        criterion.fine_soft_label_mode = str(getattr(args, "fine_soft_label_mode", "none") or "none").lower()
        criterion.grade_soft_center = float(getattr(args, "grade_soft_center", 0.85) or 0.85)
        criterion.stage_label_smoothing = float(getattr(args, "stage_label_smoothing", 0.05) or 0.05)
        criterion.loss_w_grade_soft = float(getattr(args, "loss_w_grade_soft", 0.2) or 0.2)
        criterion.loss_w_stage_soft = float(getattr(args, "loss_w_stage_soft", 0.1) or 0.1)
        criterion.loss_w_stage_smooth = float(getattr(args, "loss_w_stage_smooth", 1.0) or 1.0)
        criterion.v1_soft_label_mode = str(getattr(args, "v1_soft_label_mode", "none") or "none").lower()
        criterion.grade_soft_scheme = str(getattr(args, "grade_soft_scheme", "asym_v1") or "asym_v1").lower()
        criterion.stage_soft_scheme = str(getattr(args, "stage_soft_scheme", "asym_v1") or "asym_v1").lower()
        criterion.lambda_stage_marg = float(getattr(args, "lambda_stage_marg", 0.8) or 0.0)
        criterion.lambda_cond_stage = float(getattr(args, "lambda_cond_stage", 0.6) or 0.0)
        criterion.lambda_soft_joint = float(getattr(args, "lambda_soft_joint", 0.15) or 0.0)
        criterion.stage_fused_aux_weight = float(getattr(args, "stage_fused_aux_weight", 0.3) or 0.0)
        criterion.cond_pos_weight_g1 = float(getattr(args, "cond_pos_weight_g1", 3.0) or 1.0)
        criterion.cond_pos_weight_g2 = float(getattr(args, "cond_pos_weight_g2", 5.0) or 1.0)
        criterion.joint_graph_tau = float(getattr(args, "joint_graph_tau", 0.7) or 0.7)
        criterion.joint_beta_stage = float(getattr(args, "joint_beta_stage", 0.5) or 0.5)
        criterion.joint_gamma_cond = float(getattr(args, "joint_gamma_cond", 0.5) or 0.5)
        criterion.v2_soft_joint_start_epoch = int(getattr(args, "v2_soft_joint_start_epoch", 5) or 5)
        criterion.v2_soft_joint_warmup_epochs = int(getattr(args, "v2_soft_joint_warmup_epochs", 5) or 5)
        criterion.lambda_cond = float(getattr(args, "lambda_cond", 0.5) or 0.0)
        criterion.lambda_joint_soft = float(getattr(args, "lambda_joint_soft", 0.05) or 0.0)
        criterion.alpha_gate_min = float(getattr(args, "alpha_gate_min", 0.15) or 0.15)
        criterion.alpha_gate_max = float(getattr(args, "alpha_gate_max", 0.65) or 0.65)
        criterion.data_set = str(getattr(args, "data_set", ""))
        criterion.lpv3_active = _is_lpv3_active(args)
        criterion.lpv3_enable_cond_after_epoch = int(getattr(args, "lpv3_enable_cond_after_epoch", 3) or 3)
        criterion.lpv3_enable_soft_joint_after_epoch = int(getattr(args, "lpv3_enable_soft_joint_after_epoch", 10) or 10)
        criterion.joint_graph_w_00_11 = float(getattr(args, "joint_graph_w_00_11", 1.0) or 1.0)
        criterion.joint_graph_w_11_21 = float(getattr(args, "joint_graph_w_11_21", 0.6) or 0.6)
        criterion.joint_graph_w_11_12 = float(getattr(args, "joint_graph_w_11_12", 1.2) or 1.2)
        criterion.joint_graph_w_21_22 = float(getattr(args, "joint_graph_w_21_22", 0.8) or 0.8)
        criterion.joint_graph_w_12_22 = float(getattr(args, "joint_graph_w_12_22", 0.7) or 0.7)
        criterion.joint_graph_w_22_32 = float(getattr(args, "joint_graph_w_22_32", 1.5) or 1.5)
        if args.data_set == "advCheX_hyp_grade_stage_embtab_base":
          criterion.v1_soft_label_mode = "full"
          criterion.loss_w_stage_soft = 0.0
        if args.data_set == "advCheX_hyp_grade_stage_embtab_v2lite":
          criterion.v1_soft_label_mode = "full"
          criterion.loss_w_stage_soft = 0.0
        if getattr(args, "pos_weight_anyhtn", None):
          try:
            criterion.pos_weight_anyhtn = torch.tensor(float(args.pos_weight_anyhtn), dtype=torch.float32, device=device)
          except Exception:
            criterion.pos_weight_anyhtn = None
        print(
          f"use MultiHeadOrdinalLoss, pos_weight_grade={pos_weight_grade}, pos_weight_stage={pos_weight_stage}",
          flush=True,
        )
        if args.data_set == "advCheX_hyp_multi_grade_stage_v1":
          print(
            f"[v1 soft_label] mode={criterion.v1_soft_label_mode} grade_scheme={criterion.grade_soft_scheme} "
            f"stage_scheme={criterion.stage_soft_scheme} loss_w_grade_soft={criterion.loss_w_grade_soft} "
            f"loss_w_stage_soft={criterion.loss_w_stage_soft}",
            flush=True,
          )
        if args.data_set == "advCheX_hyp_multi_grade_stage_sep_v1":
          print(
            f"[sep_v1 coarse_auc] mode={criterion.coarse_auc_loss_mode} alpha={criterion.loss_w_anyhtn_auc} "
            f"margin={criterion.auc_margin} pair_subsample={criterion.auc_pair_subsample} "
            f"detach={criterion.auc_loss_detach_probs}",
            flush=True,
          )
          print(
            f"[sep_v1 fine_soft] mode={criterion.fine_soft_label_mode} grade_center={criterion.grade_soft_center} "
            f"stage_smoothing={criterion.stage_label_smoothing} loss_w_grade_soft={criterion.loss_w_grade_soft} "
            f"loss_w_stage_smooth={criterion.loss_w_stage_smooth}",
            flush=True,
          )
      elif args.loss_fn.lower() == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print("use FocalLoss...", flush=True)
      else:
        criterion = torch.nn.BCEWithLogitsLoss()
      if args.data_set in ["RSNAPneumonia", "COVIDx"]:
        criterion = torch.nn.CrossEntropyLoss()
        print("use CrossEntropyLoss...", flush=True)
      if args.data_set in ["advCheXX"]:
        print("[DEBUG]...use pos_weight_BCE...", flush=True)
        pos = np.zeros(args.num_class, dtype=np.float32)
        N = len(dataset_train)
        for lab in dataset_train.img_label:
            pos += np.array(lab, dtype=np.float32)
        neg = N - pos
        eps = 1.0    # ★ 平滑：把 0 当 1 处理，避免爆表
        pos_safe = np.maximum(pos, eps)
        raw = (neg / pos_safe)
        raw = np.sqrt(raw)
        raw = np.clip(raw, 1.0, 50.0)
        pos_weight = torch.tensor(neg / (pos + eps), dtype=torch.float32, device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) #pos_weight_BCE阳性越少，权重越大，缓解CAD很多，其他几乎没有的倾斜
      model = build_classification_model(args)
      if hasattr(model, "use_stopgrad_grade_for_cond"):
        model.use_stopgrad_grade_for_cond = bool(getattr(args, "use_stopgrad_grade_for_cond", True))
      print(model)
      
      # Old freeze_encoder
      # if args.freeze_encoder:
      #   print("===> freezing encoder...")
      #   ##freeze all layers but the head 
      #   for name, param in model.named_parameters():
      #     if name not in ['head.weight', 'head.bias']:
      #         param.requires_grad = False
      
      # New freeze_encoder to ensure the linear probing
      if args.freeze_encoder and not getattr(args, "use_lora", False):
        print("===> freezing encoder (linear probe)...")
        for name, p in model.named_parameters():
          if args.data_set in multihead_datasets:
            p.requires_grad = (
              name.startswith('head_grade') or name.startswith('head_stage') or
              name.startswith('head_anyhtn') or name.startswith('head_grade_pos') or name.startswith('head_stage_pos') or name.startswith('head_') or name.startswith('cond_') or name.startswith('joint_') or name.startswith('neck')
            )
          else:
            p.requires_grad = (name in ['head.weight', 'head.bias'])

      if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
      model.to(device)

      parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

      #optimizer = torch.optim.Adam(parameters, lr=args.lr)
      # optimizer = torch.optim.SGD(parameters, lr=args.lr, weight_decay=0, momentum=args.momentum, nesterov=False)
      # lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=args.patience // 2, mode='min',
      #                                  threshold=0.0001, min_lr=0, verbose=True)
      
      # New freeze_encoder to ensure the linear probing
      total = sum(p.numel() for p in model.parameters())
      trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
      names = [n for n,p in model.named_parameters() if p.requires_grad]
      neck_trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith('neck'))
      print(f"[DEBUG] params total={total}, trainable={trainable_count}, neck_trainable={neck_trainable}", flush=True)
      if args.freeze_encoder and not getattr(args, "use_lora", False):
        print(f"[DEBUG] trainable names: {names}", flush=True)
      if getattr(args, "use_lora", False) and args.data_set == "advCheX_hyp_grade_stage_v2":
        lora_trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ("lora_A" in n or "lora_B" in n))
        prefixes = ["neck.", "head_grade.", "head_stage.", "head_cond_q1.", "head_cond_q2."]
        print(f"[DEBUG][LoRA-v2] lora_trainable={lora_trainable}, neck_trainable={neck_trainable}", flush=True)
        for prefix in prefixes:
          prefix_names = [n for n in names if n.startswith(prefix)]
          print(f"[DEBUG][LoRA-v2] {prefix} trainable_count={len(prefix_names)} names={prefix_names[:8]}", flush=True)
      trainable = [p for p in model.parameters() if p.requires_grad]

      optimizer = create_optimizer(args, trainable)

      # Old freeze_encoder
      #optimizer = create_optimizer(args, model)

      lr_scheduler, _ = create_scheduler(args, optimizer)

      if args.resume:
        resume = os.path.join(model_path, experiment + '.pth.tar')
        if os.path.isfile(resume):
          print("=> loading checkpoint '{}'".format(resume), flush=True)
          checkpoint = torch.load(resume, weights_only=True)

          start_epoch = checkpoint['epoch']
          init_loss = checkpoint['lossMIN']
          model.load_state_dict(checkpoint['state_dict'])
          lr_scheduler.load_state_dict(checkpoint['scheduler'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          best_val_loss = init_loss
          print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={:.5f})"
                .format(resume, start_epoch, init_loss), flush=True)
        else:
          print("=> no checkpoint found at '{}'".format(args.resume), flush=True)

      mean_result_list, result_list = [], []
      accuracy = []
      for epoch in range(start_epoch, args.epochs):
        if args.skip_training:
          break
        if hasattr(criterion, "set_epoch"):
          criterion.set_epoch(epoch)
        train_out = train_one_epoch(data_loader_train,device, model, criterion, optimizer, epoch)
        train_loss = train_out[0] if isinstance(train_out, tuple) else train_out
        train_components = train_out[1] if isinstance(train_out, tuple) else None

        val_out = evaluate(data_loader_val, device,model, criterion)
        val_loss = val_out[0] if isinstance(val_out, tuple) else val_out
        val_components = val_out[1] if isinstance(val_out, tuple) else None
        if args.data_set in multihead_datasets and train_components is not None and val_components is not None:
          if args.data_set == "advCheX_hyp_multi_grade_stage_v2":
            print(
              "[MultiHead][Loss][v2] train: G_main={:.4f} G_soft={:.4f} S_ind={:.4f} S_fused={:.4f} "
              "C_11v12={:.4f} C_21v22={:.4f} J_soft={:.4f} alpha={:.4f} q1={:.4f} q2={:.4f} graph_cost={:.4f} total={:.4f} | "
              "val: G_main={:.4f} G_soft={:.4f} S_ind={:.4f} S_fused={:.4f} C_11v12={:.4f} C_21v22={:.4f} "
              "J_soft={:.4f} alpha={:.4f} q1={:.4f} q2={:.4f} graph_cost={:.4f} total={:.4f}".format(
                train_components.get("loss_grade_main", 0.0),
                train_components.get("loss_grade_soft", 0.0),
                train_components.get("loss_stage_marg_ind", 0.0),
                train_components.get("loss_stage_marg_fused", 0.0),
                train_components.get("loss_cond_11_12", 0.0),
                train_components.get("loss_cond_21_22", 0.0),
                train_components.get("loss_soft_joint", 0.0),
                train_components.get("mean_alpha_gate", 0.0),
                train_components.get("mean_q1", 0.0),
                train_components.get("mean_q2", 0.0),
                train_components.get("mean_expected_joint_graph_cost", 0.0),
                train_components.get("loss_total", train_loss),
                val_components.get("loss_grade_main", 0.0),
                val_components.get("loss_grade_soft", 0.0),
                val_components.get("loss_stage_marg_ind", 0.0),
                val_components.get("loss_stage_marg_fused", 0.0),
                val_components.get("loss_cond_11_12", 0.0),
                val_components.get("loss_cond_21_22", 0.0),
                val_components.get("loss_soft_joint", 0.0),
                val_components.get("mean_alpha_gate", 0.0),
                val_components.get("mean_q1", 0.0),
                val_components.get("mean_q2", 0.0),
                val_components.get("mean_expected_joint_graph_cost", 0.0),
                val_components.get("loss_total", val_loss),
              ),
              flush=True,
            )
            print(
              "[LPv3][BatchCoverage] train_present={:.2f} has11={:.2f} has21={:.2f} has32={:.2f} feat_in={:.4f} feat_out={:.4f} | val_present={:.2f} has11={:.2f} has21={:.2f} has32={:.2f} feat_in={:.4f} feat_out={:.4f}".format(
                train_components.get("batch_joint_present_classes", 0.0),
                train_components.get("batch_has_11_ratio", 0.0),
                train_components.get("batch_has_21_ratio", 0.0),
                train_components.get("batch_has_32_ratio", 0.0),
                train_components.get("mean_feature_norm_before_neck", 0.0),
                train_components.get("mean_feature_norm_after_neck", 0.0),
                val_components.get("batch_joint_present_classes", 0.0),
                val_components.get("batch_has_11_ratio", 0.0),
                val_components.get("batch_has_21_ratio", 0.0),
                val_components.get("batch_has_32_ratio", 0.0),
                val_components.get("mean_feature_norm_before_neck", 0.0),
                val_components.get("mean_feature_norm_after_neck", 0.0),
              ),
              flush=True,
            )
          elif args.data_set == "advCheX_hyp_grade_stage_embtab_v2lite":
            print(
              "[MultiHead][Loss][v2lite] train: G_main={:.4f} G_soft={:.4f} S_ind={:.4f} "
              "C_11v12={:.4f} C_21v22={:.4f} J_legal={:.4f} gate_g={:.4f} gate_s={:.4f} total={:.4f} | "
              "val: G_main={:.4f} G_soft={:.4f} S_ind={:.4f} C_11v12={:.4f} C_21v22={:.4f} J_legal={:.4f} "
              "gate_g={:.4f} gate_s={:.4f} total={:.4f}".format(
                train_components.get("loss_grade_main", 0.0),
                train_components.get("loss_grade_soft", 0.0),
                train_components.get("loss_stage_marg_ind", 0.0),
                train_components.get("loss_cond_11_12", 0.0),
                train_components.get("loss_cond_21_22", 0.0),
                train_components.get("loss_soft_joint", 0.0),
                train_components.get("mean_gate_g", 0.0),
                train_components.get("mean_gate_s", 0.0),
                train_components.get("loss_total", train_loss),
                val_components.get("loss_grade_main", 0.0),
                val_components.get("loss_grade_soft", 0.0),
                val_components.get("loss_stage_marg_ind", 0.0),
                val_components.get("loss_cond_11_12", 0.0),
                val_components.get("loss_cond_21_22", 0.0),
                val_components.get("loss_soft_joint", 0.0),
                val_components.get("mean_gate_g", 0.0),
                val_components.get("mean_gate_s", 0.0),
                val_components.get("loss_total", val_loss),
              ),
              flush=True,
            )
          elif args.data_set == "advCheX_hyp_multi_grade_stage_sep_v1" and str(getattr(args, "sep_head_mode", "flat")).lower() == "coarse_fine":
            print(
              "[MultiHead][Loss][coarse_fine] train: H_bce={:.4f} H_auc={:.4f} H_total={:.4f} "
              "G_corn={:.4f} G_soft={:.4f} G_total={:.4f} S={:.4f} S_smooth={:.0f} total={:.4f} | "
              "val: H_bce={:.4f} H_auc={:.4f} H_total={:.4f} G_corn={:.4f} G_soft={:.4f} G_total={:.4f} "
              "S={:.4f} S_smooth={:.0f} total={:.4f}".format(
                train_components.get("loss_anyhtn", 0.0),
                train_components.get("loss_anyhtn_auc", 0.0),
                train_components.get("loss_anyhtn_total", 0.0),
                train_components.get("loss_grade_corn", 0.0),
                train_components.get("loss_grade_soft", 0.0),
                train_components.get("loss_grade_total", train_components.get("loss_grade", 0.0)),
                train_components.get("loss_stage", 0.0),
                train_components.get("stage_smooth_enabled", 0.0),
                train_components.get("loss_total", train_loss),
                val_components.get("loss_anyhtn", 0.0),
                val_components.get("loss_anyhtn_auc", 0.0),
                val_components.get("loss_anyhtn_total", 0.0),
                val_components.get("loss_grade_corn", 0.0),
                val_components.get("loss_grade_soft", 0.0),
                val_components.get("loss_grade_total", val_components.get("loss_grade", 0.0)),
                val_components.get("loss_stage", 0.0),
                val_components.get("stage_smooth_enabled", 0.0),
                val_components.get("loss_total", val_loss),
              ),
              flush=True,
            )
          else:
            print(
              "[MultiHead][Loss] train: grade={:.4f} stage={:.4f} incomp={:.4f} joint={:.4f} total={:.4f} | "
              "val: grade={:.4f} stage={:.4f} incomp={:.4f} joint={:.4f} total={:.4f}".format(
                train_components.get("loss_grade", 0.0),
                train_components.get("loss_stage", 0.0),
                train_components.get("loss_incomp", 0.0),
                train_components.get("loss_joint", 0.0),
                train_components.get("loss_total", train_loss),
                val_components.get("loss_grade", 0.0),
                val_components.get("loss_stage", 0.0),
                val_components.get("loss_incomp", 0.0),
                val_components.get("loss_joint", 0.0),
                val_components.get("loss_total", val_loss),
              ),
              flush=True,
            )

        y_val_np, p_val_np, val_auc_hyp = None, None, None
        y_grade_val, y_stage_val, p_grade_val, p_stage_val = None, None, None, None
        if args.data_set == "advCheX_hyp_multi_level":
          y_val_np, p_val_np = _collect_outputs(model, data_loader_val, device, args)
          val_metrics, _, _ = evaluate_ordinal_tasks(y_val_np, p_val_np)
          val_auc_hyp = val_metrics.get("AUROC_hypertension_vs_non")
          if val_auc_hyp is not None:
            print(f"Epoch {epoch:04d}: val_auc_hypertension={val_auc_hyp:.4f}", flush=True)
          else:
            print(f"Epoch {epoch:04d}: val_auc_hypertension=N/A (single class)", flush=True)
        if args.data_set in multihead_datasets:
          if args.data_set == "advCheX_hyp_grade_stage_embtab_v2lite":
            val_pack = _collect_outputs_multi_v2lite(
              model, data_loader_val, device,
              joint_beta_stage=getattr(args, "joint_beta_stage", 0.5),
              joint_gamma_cond=getattr(args, "joint_gamma_cond", 0.5),
            )
            y_grade_val = val_pack["y_grade"]
            y_stage_val = val_pack["y_stage"]
            p_grade_val = val_pack["p_grade_ge"]
            p_stage_val = val_pack["p_stage_ge"]
            pG_val = val_pack["pG_fused"]
            pS_val = val_pack["pS_fused"]
            grade_pred = np.argmax(pG_val, axis=1)
            stage_pred = np.argmax(pS_val, axis=1)
            joint_probs = val_pack["p_joint6"]
            joint_pred = np.argmax(joint_probs, axis=1)
            joint_true = np.array([
              JOINT_LABEL_TO_INDEX[(ordinal_targets_to_grade(g), ordinal_targets_to_grade(s))]
              for g, s in zip(y_grade_val, y_stage_val)
            ], dtype=np.int64)
            val_joint = float(np.mean(joint_pred == joint_true)) if joint_true.size > 0 else np.nan
          else:
            y_grade_val, y_stage_val, p_grade_val, p_stage_val = _collect_outputs_multi(
              model, data_loader_val, device, ordinal_mode=getattr(args, "ordinal_mode", "default")
            )
            prior = build_joint_prior_mimic(
              np.array([ordinal_targets_to_grade(row) for row in y_grade_val]),
              np.array([ordinal_targets_to_grade(row) for row in y_stage_val]),
              eps=getattr(args, "joint_prior_eps", 1e-3),
            )
            val_metrics, _, _, _ = evaluate_grade_stage_joint(
              y_grade_val,
              y_stage_val,
              p_grade_val,
              p_stage_val,
              prior=prior,
              prior_alpha=getattr(args, "joint_prior_alpha", 0.2),
              softacc_gamma_over=getattr(args, "softacc_gamma_over", 0.5),
              ordinal_mode=getattr(args, "ordinal_mode", "coral").upper(),
            )
            val_joint = val_metrics.get("joint_exact_acc_pjoint")
            pG_val = ordinal_probs_to_class_probs(p_grade_val)
            pS_val = ordinal_probs_to_class_probs(p_stage_val)
            grade_pred = np.argmax(pG_val, axis=1)
            stage_pred = np.argmax(pS_val, axis=1)
            joint_probs = compute_joint_distribution(
              pG_val, pS_val, prior=prior, alpha=getattr(args, "joint_prior_alpha", 0.2)
            )
            joint_pred = np.argmax(joint_probs, axis=1)
          if val_joint is not None:
            print(f"Epoch {epoch:04d}: val_joint_exact_acc_pjoint={val_joint:.4f}", flush=True)
          print(f"[MultiHead][Mode] ordinal_mode={getattr(args, 'ordinal_mode', 'default')}", flush=True)
          grade_counts = np.bincount(grade_pred, minlength=4)
          stage_counts = np.bincount(stage_pred, minlength=3)
          joint_counts = np.bincount(joint_pred, minlength=len(JOINT_LABELS))
          grade_ratio = grade_counts / max(grade_counts.sum(), 1)
          stage_ratio = stage_counts / max(stage_counts.sum(), 1)
          joint_ratio = joint_counts / max(joint_counts.sum(), 1)
          print(
            "[MultiHead][Dist] grade_count={} ratio={} | stage_count={} ratio={} | joint_count={} ratio={}".format(
              grade_counts.tolist(),
              np.round(grade_ratio, 4).tolist(),
              stage_counts.tolist(),
              np.round(stage_ratio, 4).tolist(),
              joint_counts.tolist(),
              np.round(joint_ratio, 4).tolist(),
            ),
            flush=True,
          )
          mean_pG = np.round(pG_val.mean(axis=0), 4).tolist()
          mean_pS = np.round(pS_val.mean(axis=0), 4).tolist()
          ret_s1 = float(pS_val[:, 1].mean()) if len(pS_val) > 0 else 0.0
          top1_s1 = float((stage_pred == 1).mean()) if len(stage_pred) > 0 else 0.0
          ret_g12 = float((pG_val[:, 1] + pG_val[:, 2]).mean()) if len(pG_val) > 0 else 0.0
          top1_g12 = float(np.isin(grade_pred, [1, 2]).mean()) if len(grade_pred) > 0 else 0.0
          print(
            "[MultiHead][Retention] stage_pred_count={} grade_pred_count={} | mean_pS={} mean_pG={} | "
            "RetS1={:.4f} Top1S1={:.4f} RetG12={:.4f} Top1G12={:.4f}".format(
              stage_counts.tolist(),
              grade_counts.tolist(),
              mean_pS,
              mean_pG,
              ret_s1,
              top1_s1,
              ret_g12,
              top1_g12,
            ),
            flush=True,
          )
          grade_viol_ge2 = float(np.mean(p_grade_val[:, 1] > p_grade_val[:, 0])) if len(p_grade_val) > 0 else 0.0
          grade_viol_ge3 = float(np.mean(p_grade_val[:, 2] > p_grade_val[:, 1])) if len(p_grade_val) > 0 else 0.0
          stage_viol_ge2 = float(np.mean(p_stage_val[:, 1] > p_stage_val[:, 0])) if len(p_stage_val) > 0 else 0.0
          if str(getattr(args, "ordinal_mode", "coral")).lower() == "corn":
            grade_gap = np.stack(
              [p_grade_val[:, 0] - p_grade_val[:, 1], p_grade_val[:, 1] - p_grade_val[:, 2]],
              axis=1,
            )
            stage_gap = (p_stage_val[:, 0] - p_stage_val[:, 1])[:, None]
            grade_gap_p5 = np.percentile(grade_gap, 5, axis=0).tolist()
            grade_gap_p50 = np.percentile(grade_gap, 50, axis=0).tolist()
            stage_gap_p5 = np.percentile(stage_gap, 5, axis=0).tolist()
            stage_gap_p50 = np.percentile(stage_gap, 50, axis=0).tolist()
            print(
              "[MultiHead][OrdinalViol] (CORN) grade_ge2>ge1={:.4f} grade_ge3>ge2={:.4f} "
              "stage_ge2>ge1={:.4f} | gap_p5 grade={} stage={} | gap_p50 grade={} stage={}".format(
                grade_viol_ge2, grade_viol_ge3, stage_viol_ge2,
                np.round(grade_gap_p5, 4).tolist(),
                np.round(stage_gap_p5, 4).tolist(),
                np.round(grade_gap_p50, 4).tolist(),
                np.round(stage_gap_p50, 4).tolist(),
              ),
              flush=True,
            )
          else:
            print(
              "[MultiHead][OrdinalViol] grade_ge2>ge1={:.4f} grade_ge3>ge2={:.4f} stage_ge2>ge1={:.4f}".format(
                grade_viol_ge2, grade_viol_ge3, stage_viol_ge2
              ),
              flush=True,
            )

        lr_scheduler.step(val_loss)

        if args.test_every_epoch:
          y_test, p_test = test_model(model, data_loader_test, args)
          if args.data_set in multihead_datasets:
            print("[DEBUG] multi-head dataset skip test_every_epoch metrics", flush=True)
            continue
          if isinstance(y_test, dict):
            pass
          else:
            y_test = y_test.cpu().numpy()
            p_test = p_test.cpu().numpy()

          if args.data_set in ["RSNAPneumonia", "COVIDx"]:
            acc = accuracy_score(np.argmax(y_test,axis=1),np.argmax(p_test,axis=1))
            print(">>{}: ACCURACY = {}".format(experiment,acc))
            with open(output_file, 'a') as writer:
              writer.write(
                "{}: ACCURACY = {}\n".format(experiment, np.array2string(np.array(acc), precision=4, separator='\t')))
            accuracy.append(acc)
          
          if test_diseases is not None:
            y_test = copy.deepcopy(y_test[:,test_diseases])
            p_test = copy.deepcopy(p_test[:,test_diseases])

          mAUC, auc_scores = meanAUC(y_test, p_test)
          mMCC, mcc_scores = meanMCC(y_test, p_test)
          mAP, ap_scores = meanAP(y_test, p_test)
          mF1, f1_scores, optimal_thresholds, recall_scores = meanF1(y_test, p_test)
            
          print(">> Mean AUC = {:.4f} \nAUC = {}".format(mAUC, np.array2string(np.array(auc_scores), precision=4, separator=',')))
          print(">> Mean MCC = {:.4f} \nMCC = {}".format(mMCC, np.array2string(np.array(mcc_scores), precision=4, separator=',')))
          print(">> Mean AP = {:.4f} \nAP = {}".format(mAP, np.array2string(np.array(ap_scores), precision=4, separator=',')))
          print(">> Mean F1 = {:.4f} \nF1 = {}".format(mF1, np.array2string(np.array(f1_scores), precision=4, separator=',')))
          print(">> Optimal Thresholds = {}".format(np.array2string(np.array(optimal_thresholds), precision=4, separator=',')))
          print(">> Recall = {}".format(np.array2string(np.array(recall_scores), precision=4, separator=',')))
          mean_result_list.append(mAUC)
          result_list.append([mAUC,mMCC,mAP,mF1])
        
        if val_loss < best_val_loss:
          print(
            "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss, val_loss,
                                                                                              save_model_path))
          best_val_loss = val_loss
          patience_counter = 0
          ckpt_payload = {
            'epoch': epoch + 1,
            'lossMIN': best_val_loss,
            'state_dict': model.state_dict(),
          }
          if args.data_set in ordinal_datasets:
            if y_val_np is None or p_val_np is None:
              y_val_np, p_val_np = _collect_outputs(model, data_loader_val, device, args)
            if args.data_set == "advCheX_hyp_multi_stage_v2":
              ordinal_thresholds = compute_stage2_thresholds(y_val_np, p_val_np)
            else:
              ordinal_thresholds = compute_ordinal_thresholds(y_val_np, p_val_np)
            json_path = save_model_path + "_thresholds.json"
            save_thresholds_json(json_path, ordinal_thresholds)
            meta_path = save_model_path + "_meta.json"
            with open(meta_path, "w") as f:
              json.dump({
                "epoch": epoch + 1,
                "lossMIN": float(best_val_loss),
                "thresholds_json": os.path.basename(json_path)
              }, f, indent=2)
            print(f"[Ordinal] thresholds saved to {json_path}", flush=True)
            print(f"[Ordinal] meta saved to {meta_path}", flush=True)
          elif args.data_set in multihead_datasets:
            if y_grade_val is None or y_stage_val is None:
              y_grade_val, y_stage_val, p_grade_val, p_stage_val = _collect_outputs_multi(
                model, data_loader_val, device, ordinal_mode=getattr(args, "ordinal_mode", "default")
              )
            grade_thresholds = compute_ordinal_thresholds(y_grade_val, p_grade_val).get("youden", {})
            stage_thresholds = compute_stage2_thresholds(y_stage_val, p_stage_val)
            ordinal_thresholds = {"grade": grade_thresholds, "stage": stage_thresholds}
            json_path = save_model_path + "_thresholds.json"
            save_thresholds_json(json_path, ordinal_thresholds)
            meta_path = save_model_path + "_meta.json"
            with open(meta_path, "w") as f:
              json.dump({
                "epoch": epoch + 1,
                "lossMIN": float(best_val_loss),
                "thresholds_json": os.path.basename(json_path)
              }, f, indent=2)
            print(f"[Ordinal-MultiHead] thresholds saved to {json_path}", flush=True)
            print(f"[Ordinal-MultiHead] meta saved to {meta_path}", flush=True)
          else:
            ckpt_payload['optimizer'] = optimizer.state_dict()
            ckpt_payload['scheduler'] = lr_scheduler.state_dict()
          save_checkpoint(ckpt_payload,  filename="{}".format(save_model_path, epoch))

        else:
          print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss))
          patience_counter += 1

        if patience_counter >= args.patience:
          print("Early Stopping")
          break
          
      #save_checkpoint({
      #      'state_dict': model.state_dict(),
      #    },  filename="{}({} epoch)".format(save_model_path, epoch))
  
      # log experiment
      with open(log_file, 'a') as f:
        f.write("{} ({}epoch)\n".format(experiment,epoch-args.patience))
        f.close()

      if len(mean_result_list) > 0:
        best_rest = max(mean_result_list)
        best_epoch = mean_result_list.index(best_rest)
        print("=====> Max result:  {} at epoch {}".format(best_rest, best_epoch) )
        print("mAUC, mMCC, mAP, mF1 = {}".format(result_list[best_epoch]))
        with open(output_file, 'a') as writer:
          writer.write("=====> Max result:  {} at epoch {}\n".format(best_rest, best_epoch))
          writer.write("mAUC, mMCC, mAP, mF1 = {}\n".format(result_list[best_epoch]))
 
  if args.mode == "train" and getattr(args, "skip_test", False):
    return

  print ("start testing.....")


  log_file = os.path.join(model_path, "models.log")
  if not os.path.isfile(log_file):
    print("log_file ({}) not exists!".format(log_file))
  else:
    accuracy = []
    mean_auc, mean_mcc, mean_ap, mean_f1 = [],[],[],[]
    metric_dict = {"auc": [], "mcc": [], "ap": [], "f1": []}
    with open(log_file, 'r') as reader, open(output_file, 'a') as writer:
      experiment = reader.readline()
      print(">> Disease = {}".format(diseases))
      writer.write("Disease = {}\n".format(diseases))

      while experiment:
        experiment = experiment.split()[0]
        saved_model = os.path.join(model_path, experiment + ".pth.tar")
        pred_csv = os.path.join(model_path, experiment + ".csv")
        gt_csv = os.path.join(model_path, "gt.csv")
        path_list = None
        use_cached = os.path.exists(pred_csv) and os.path.exists(gt_csv) and args.data_set not in {
          "advCheX_hyp_multi_level",
          "advCheX_hyp_multi_grade_stage_v1",
        "advCheX_hyp_multi_grade_stage_sep_v1",
        "advCheX_hyp_grade_stage_embtab_base",
        "advCheX_hyp_grade_stage_embtab_v2lite",
        }
        if use_cached:
          y_test = read_from_csv(gt_csv)
          p_test = read_from_csv(pred_csv)
        else:
          test_out = test_classification(saved_model, data_loader_test, device, args)
          if isinstance(test_out, tuple) and len(test_out) == 3:
            y_test, p_test, path_list = test_out
          else:
            y_test, p_test = test_out
            path_list = None
          if not isinstance(y_test, dict):
            y_test = y_test.cpu().numpy()
            p_test = p_test.cpu().numpy()

        if args.data_set in ["RSNAPneumonia", "COVIDx"]:
          acc = accuracy_score(np.argmax(y_test,axis=1),np.argmax(p_test,axis=1))
          print(">>{}: ACCURACY = {}".format(experiment,acc))
          writer.write(
            "{}: ACCURACY = {}\n".format(experiment, np.array2string(np.array(acc), precision=4, separator='\t')))
          accuracy.append(acc)

        
        if args.data_set in multihead_datasets:
          thresholds_src = _load_ordinal_thresholds(saved_model, args)
          grade_thresholds = None
          stage_thresholds = None
          if isinstance(thresholds_src, dict):
            grade_thresholds = thresholds_src.get("grade")
            stage_thresholds = thresholds_src.get("stage")

          y_grade = y_test["grade"].cpu().numpy() if isinstance(y_test, dict) else y_test
          y_stage = y_test["stage"].cpu().numpy() if isinstance(y_test, dict) else y_test
          p_grade = p_test["grade"].cpu().numpy() if isinstance(p_test, dict) else p_test
          p_stage = p_test["stage"].cpu().numpy() if isinstance(p_test, dict) else p_test
          aux_scores = {}
          if isinstance(p_test, dict):
            if "anyhtn" in p_test:
              aux_scores["p_anyhtn_coarse"] = p_test["anyhtn"].cpu().numpy()
            if "grade_pos_probs" in p_test:
              aux_scores["grade_pos_probs"] = p_test["grade_pos_probs"].cpu().numpy()
            if "stage_pos_probs" in p_test:
              aux_scores["stage_pos_probs"] = p_test["stage_pos_probs"].cpu().numpy()
            if "q1" in p_test:
              aux_scores["q1"] = p_test["q1"].cpu().numpy()
              aux_scores["q2"] = p_test["q2"].cpu().numpy()
              aux_scores["p_joint6"] = p_test["p_joint6"].cpu().numpy()
              aux_scores["pG_fused"] = p_test["pG_fused"].cpu().numpy()
              aux_scores["pS_fused"] = p_test["pS_fused"].cpu().numpy()
              aux_scores["alpha_gate"] = p_test["alpha_gate"].cpu().numpy()
              if "gate_g" in p_test:
                aux_scores["gate_g"] = p_test["gate_g"].cpu().numpy()
              if "gate_s" in p_test:
                aux_scores["gate_s"] = p_test["gate_s"].cpu().numpy()

          if args.data_set in {"advCheX_hyp_multi_grade_stage_sep_v1", "advCheX_hyp_multi_grade_stage_v1", "advCheX_hyp_grade_stage_v2", "advCheX_hyp_grade_stage_embtab_base", "advCheX_hyp_grade_stage_embtab_v2lite"}:
            output_dir = os.path.dirname(output_file)
            val_y_grade = val_y_stage = val_p_grade = val_p_stage = None
            decoder_mode = str(getattr(args, "decodermode", "non")).lower()
            need_val_for_decoder = decoder_mode in {"ev", "temp_threshold", "temp_ev"}
            if decoder_mode == "threshold":
              saved_thr_grade, saved_thr_stage = extract_saved_thresholds_for_sep(thresholds_src) if getattr(args, "decoder_use_saved_thresholds", True) else (None, None)
              need_val_for_decoder = not (saved_thr_grade is not None and saved_thr_stage is not None)
            if need_val_for_decoder:
              if dataset_val is None:
                if decoder_mode == "threshold":
                  raise ValueError("Decoder mode=threshold: 未找到可用已保存阈值且 dataset_val is None，无法在验证集重搜。")
                raise ValueError(f"Decoder mode={decoder_mode} requires validation set for parameter search, but dataset_val is None.")

              data_loader_val_for_decoder = DataLoader(
                dataset=dataset_val,
                batch_size=int(args.batch_size/2),
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                collate_fn=safe_collate,
                persistent_workers=False,
              )
              val_out = test_classification(saved_model, data_loader_val_for_decoder, device, args)
              if isinstance(val_out, tuple) and len(val_out) == 3:
                y_val_pred, p_val_pred, _ = val_out
              else:
                y_val_pred, p_val_pred = val_out
              if isinstance(y_val_pred, dict):
                val_y_grade = y_val_pred["grade"].cpu().numpy()
                val_y_stage = y_val_pred["stage"].cpu().numpy()
                val_p_grade = p_val_pred["grade"].cpu().numpy()
                val_p_stage = p_val_pred["stage"].cpu().numpy()
              else:
                raise ValueError("Expected multi-head dict outputs on validation decoder pass, but got non-dict outputs.")
            eval_fn = evaluate_grade_stage_v2 if args.data_set in {"advCheX_hyp_grade_stage_v2", "advCheX_hyp_grade_stage_embtab_v2lite"} else evaluate_grade_stage_sep
            eval_kwargs = dict(
              output_dir=output_dir, path_list=path_list,
              modethese=getattr(args, "modethese", False),
              decodermode=getattr(args, "decodermode", "non"),
              decoder_objective=getattr(args, "decoder_objective", "qwk"),
              decoder_bins=getattr(args, "decoder_bins", 101),
              decoder_use_saved_thresholds=getattr(args, "decoder_use_saved_thresholds", True),
              decoder_save_debug=getattr(args, "decoder_save_debug", True),
              temperature_init=getattr(args, "temperature_init", 1.0),
              temperature_min=getattr(args, "temperature_min", 0.5),
              temperature_max=getattr(args, "temperature_max", 5.0),
              temperature_grid_size=getattr(args, "temperature_grid_size", 91),
              decoder_keep_raw_metrics=getattr(args, "decoder_keep_raw_metrics", True),
              thresholds_src=thresholds_src,
              val_y_grade=val_y_grade, val_y_stage=val_y_stage,
              val_p_ge_grade=val_p_grade, val_p_ge_stage=val_p_stage,
              sep_head_mode=getattr(args, "sep_head_mode", "flat"),
              aux_scores=aux_scores,
              loss_w_anyhtn=getattr(args, "loss_w_anyhtn", 1.0),
              pos_weight_anyhtn=getattr(args, "pos_weight_anyhtn", None),
              coarse_auc_loss_mode=getattr(args, "coarse_auc_loss_mode", "none"),
              loss_w_anyhtn_auc=getattr(args, "loss_w_anyhtn_auc", 0.0),
              auc_margin=getattr(args, "auc_margin", 1.0),
              auc_pair_subsample=getattr(args, "auc_pair_subsample", 256),
              fine_soft_label_mode=getattr(args, "fine_soft_label_mode", "none"),
              grade_soft_center=getattr(args, "grade_soft_center", 0.85),
              stage_label_smoothing=getattr(args, "stage_label_smoothing", 0.05),
              loss_w_grade_soft=getattr(args, "loss_w_grade_soft", 0.2),
              loss_w_stage_soft=getattr(args, "loss_w_stage_soft", 0.1),
              loss_w_stage_smooth=getattr(args, "loss_w_stage_smooth", 1.0),
              dataset_tag=("sep_v1" if args.data_set == "advCheX_hyp_multi_grade_stage_sep_v1" else ("v2lite" if args.data_set == "advCheX_hyp_grade_stage_embtab_v2lite" else ("v2" if args.data_set == "advCheX_hyp_grade_stage_v2" else ("embtab_base" if args.data_set == "advCheX_hyp_grade_stage_embtab_base" else "v1")))),
              v1_soft_label_mode=getattr(args, "v1_soft_label_mode", "none"),
              grade_soft_scheme=getattr(args, "grade_soft_scheme", "asym_v1"),
              stage_soft_scheme=getattr(args, "stage_soft_scheme", "asym_v1"),
              lambda_incomp=getattr(args, "lambda_incomp", 0.0),
              lambda_joint=getattr(args, "lambda_joint", 0.0),
              joint_gate=getattr(args, "joint_gate", "htn_only"),
              joint_detach=getattr(args, "joint_detach", "both"),
              incomp_mode=getattr(args, "incomp_mode", "mask_sum"),
            )
            if args.data_set in {"advCheX_hyp_grade_stage_v2", "advCheX_hyp_grade_stage_embtab_v2lite"}:
              eval_kwargs.update(
                lambda_stage_marg=getattr(args, "lambda_stage_marg", 0.8),
                lambda_cond_stage=getattr(args, "lambda_cond_stage", 0.6),
                lambda_soft_joint=getattr(args, "lambda_soft_joint", 0.15),
                lambda_cond=getattr(args, "lambda_cond", 0.5),
                lambda_joint_soft=getattr(args, "lambda_joint_soft", 0.05),
                stage_fused_aux_weight=getattr(args, "stage_fused_aux_weight", 0.3),
                cond_pos_weight_g1=getattr(args, "cond_pos_weight_g1", 3.0),
                cond_pos_weight_g2=getattr(args, "cond_pos_weight_g2", 5.0),
                joint_graph_tau=getattr(args, "joint_graph_tau", 0.7),
                joint_beta_stage=getattr(args, "joint_beta_stage", 0.5),
                joint_gamma_cond=getattr(args, "joint_gamma_cond", 0.5),
                alpha_gate_min=getattr(args, "alpha_gate_min", 0.15),
                alpha_gate_max=getattr(args, "alpha_gate_max", 0.65),
                v2_soft_joint_start_epoch=getattr(args, "v2_soft_joint_start_epoch", 5),
                v2_soft_joint_warmup_epochs=getattr(args, "v2_soft_joint_warmup_epochs", 5),
                use_stopgrad_grade_for_cond=getattr(args, "use_stopgrad_grade_for_cond", True),
                teacher_force_grade_epochs=getattr(args, "teacher_force_grade_epochs", 0),
                joint_graph_w_00_11=getattr(args, "joint_graph_w_00_11", 1.0),
                joint_graph_w_11_21=getattr(args, "joint_graph_w_11_21", 0.6),
                joint_graph_w_11_12=getattr(args, "joint_graph_w_11_12", 1.2),
                joint_graph_w_21_22=getattr(args, "joint_graph_w_21_22", 0.8),
                joint_graph_w_12_22=getattr(args, "joint_graph_w_12_22", 0.7),
                joint_graph_w_22_32=getattr(args, "joint_graph_w_22_32", 1.5),
                use_v2lite_fused_eval=getattr(args, "use_v2lite_fused_eval", True),
                use_legal_joint_composer=getattr(args, "use_legal_joint_composer", True),
              )
            metrics, pred_rows, report_lines = eval_fn(
              y_grade, y_stage, p_grade, p_stage, **eval_kwargs
            )
            with open(os.path.join(output_dir, "predictions.csv"), mode='w', newline='') as fcsv:
              if pred_rows:
                writer_csv = csv.DictWriter(fcsv, fieldnames=list(pred_rows[0].keys()))
                writer_csv.writeheader(); writer_csv.writerows(pred_rows)
            metrics['lpv3'] = _build_lpv3_config(args)
            if sampler_summary is not None:
              metrics['lpv3']['sampler_summary'] = sampler_summary
            if args.data_set == "advCheX_hyp_grade_stage_embtab_base":
              embtab_summary = {
                "img_emb_dim": int(getattr(args, "img_emb_dim", 1376)),
                "tab_dim": int(getattr(args, "tab_dim", 5)),
                "img_hidden_dim": int(getattr(args, "img_hidden_dim", 512)),
                "img_out_dim": int(getattr(args, "img_out_dim", 256)),
                "tab_hidden_dim": int(getattr(args, "tab_hidden_dim", 32)),
                "tab_out_dim": int(getattr(args, "tab_out_dim", 64)),
                "fusion_hidden_dim": int(getattr(args, "fusion_hidden_dim", 192)),
                "task_hidden_dim": int(getattr(args, "task_hidden_dim", 128)),
                "grade_tab_scale": float(getattr(args, "grade_tab_scale", 0.3)),
                "dropout_img": float(getattr(args, "dropout_img", 0.2)),
                "dropout_tab": float(getattr(args, "dropout_tab", 0.1)),
                "dropout_fusion": float(getattr(args, "dropout_fusion", 0.2)),
                "embtab_stage_soft_label": False,
                "grade_soft_scheme": str(getattr(args, "grade_soft_scheme", "asym_v1")),
                "loss_w_grade_soft": float(getattr(args, "loss_w_grade_soft", 0.2)),
              }
              metrics["embtab_base"] = embtab_summary
              report_lines.extend(["", "[embtab-base summary]"] + [f"{k}={v}" for k, v in embtab_summary.items()])
            if args.data_set == "advCheX_hyp_grade_stage_embtab_v2lite":
              v2lite_summary = {
                "mean_gate_g": float(np.mean(aux_scores["gate_g"])) if "gate_g" in aux_scores else None,
                "mean_gate_s": float(np.mean(aux_scores["gate_s"])) if "gate_s" in aux_scores else None,
                "use_stopgrad_grade_for_cond": bool(getattr(args, "use_stopgrad_grade_for_cond", True)),
                "joint_beta_stage": float(getattr(args, "joint_beta_stage", 0.5)),
                "joint_gamma_cond": float(getattr(args, "joint_gamma_cond", 0.5)),
                "cond_pos_weight_g1": float(getattr(args, "cond_pos_weight_g1", 3.0)),
                "cond_pos_weight_g2": float(getattr(args, "cond_pos_weight_g2", 5.0)),
                "embtab_v2lite_grade_fusion": "residual_gated",
                "embtab_v2lite_stage_fusion": "residual_gated",
                "embtab_v2lite_conditional_stage": True,
                "embtab_v2lite_stage_soft_label": False,
              }
              metrics["embtab_v2lite"] = v2lite_summary
              report_lines.extend(["", "[embtab-v2lite summary]"] + [f"{k}={v}" for k, v in v2lite_summary.items()])
            with open(os.path.join(output_dir, "metrics.json"), 'w') as fm:
              json.dump(metrics, fm, indent=2, ensure_ascii=False)
            with open(os.path.join(output_dir, "result.txt"), 'w', encoding='utf-8') as fr:
              lpv3_lines = ["[LPv3]"] + [f"{k}: {v}" for k, v in metrics['lpv3'].items()]
              fr.write("\n".join(report_lines + [""] + lpv3_lines) + "\n")
            writer.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            experiment = reader.readline()
            continue

          if getattr(args, "test_time_adjust", False):
            grade_thresholds = compute_ordinal_thresholds(y_grade, p_grade).get("youden", grade_thresholds)
            stage_thresholds = compute_stage2_thresholds(y_stage, p_stage)

          prior = None
          if getattr(args, "joint_prior_mode", "mimic") != "none":
            grades_prior = []
            stages_prior = []
            for dataset in [dataset_train, dataset_val]:
              if dataset is not None and hasattr(dataset, "grade_list"):
                grades_prior.extend(dataset.grade_list)
                stages_prior.extend(dataset.stage_list)
            if grades_prior:
              prior = build_joint_prior_mimic(
                np.array(grades_prior),
                np.array(stages_prior),
                eps=getattr(args, "joint_prior_eps", 1e-3),
              )
            if getattr(args, "joint_prior_mode", "mimic") == "mix":
              private_path = getattr(args, "joint_prior_private_json", None)
              if private_path and os.path.exists(private_path):
                with open(private_path, "r") as f:
                  private_prior = np.array(json.load(f), dtype=np.float32)
                beta = getattr(args, "joint_prior_beta", 0.5)
                if prior is None:
                  prior = private_prior
                else:
                  prior = (1 - beta) * prior + beta * private_prior

          metrics, pG, pS, P_joint = evaluate_grade_stage_joint(
            y_grade,
            y_stage,
            p_grade,
            p_stage,
            prior=prior,
            prior_alpha=getattr(args, "joint_prior_alpha", 0.2),
            softacc_gamma_over=getattr(args, "softacc_gamma_over", 0.5),
            ordinal_mode=getattr(args, "ordinal_mode", "coral").upper(),
          )
          if getattr(args, "modethese", False):
            grades_true = np.array([ordinal_targets_to_grade(row) for row in y_grade])
            stages_true = np.array([ordinal_targets_to_grade(row) for row in y_stage])
            output_dir = os.path.dirname(output_file)
            metrics["modethese"] = compute_modethese_outputs(
              grades_true,
              stages_true,
              pG,
              pS,
              P_joint,
              output_dir,
            )
          metrics["joint_label_order"] = JOINT_LABELS
          metrics["thresholds_grade"] = grade_thresholds
          metrics["thresholds_stage"] = stage_thresholds
          metrics["loss_grade"] = np.nan
          metrics["loss_stage"] = np.nan
          metrics["loss_incomp"] = np.nan
          metrics["loss_joint"] = np.nan
          writer.write(json.dumps(metrics, ensure_ascii=False) + "\n")
          experiment = reader.readline()
          continue

        if args.data_set in ordinal_datasets:
          thresholds_src = _load_ordinal_thresholds(saved_model, args)
          if args.data_set == "advCheX_hyp_multi_stage_v2":
            thresholds_use = thresholds_src if isinstance(thresholds_src, dict) else None
          else:
            thresholds_use = thresholds_src.get('youden') if isinstance(thresholds_src, dict) else None
          y_np = y_test if isinstance(y_test, np.ndarray) else y_test
          p_np = p_test if isinstance(p_test, np.ndarray) else p_test
          k = p_np.shape[1]
          if getattr(args, "test_time_adjust", False):
            if args.data_set == "advCheX_hyp_multi_stage_v2":
              thresholds_use = compute_stage2_thresholds(y_np, p_np)
            else:
              thresholds_use = compute_ordinal_thresholds(y_np, p_np).get("youden", thresholds_use)

          metrics, grades_true, grade_pred = evaluate_ordinal_tasks(y_np, p_np, thresholds_use)
          writer.write(json.dumps(metrics, ensure_ascii=False) + "\n")

          task_views = build_ordinal_task_views(y_np, p_np)
          # 任务阈值：测试时调整则在测试集上用 Youden 搜索；否则沿用 ge1/2/3 或 0.5 默认
          if getattr(args, "test_time_adjust", False):
            task_thresholds = compute_task_thresholds(task_views, metric="youden")
          else:
            ge_defaults = thresholds_use or {f"ge{i}": 0.5 for i in range(1, k + 1)}
            task_thresholds = {}
            for name in task_views.keys():
              if name in ["hasHTN", "hypertension_vs_non", "ge1"]:
                task_thresholds[name] = ge_defaults.get("ge1", 0.5)
              elif name in ["severe", "ge2"]:
                task_thresholds[name] = ge_defaults.get("ge2", 0.5)
              elif name in ["very_severe", "ge3"]:
                task_thresholds[name] = ge_defaults.get("ge3", 0.5)
              elif name in ["lv1_vs_non", "lv2_vs_non", "lv3_vs_non"]:
                labels, scores, _ = task_views[name]
                task_thresholds[name] = compute_threshold_by_metric(labels, scores, metric="youden")
              elif name in ["stage1_vs_non", "stage2_vs_non"]:
                if args.data_set == "advCheX_hyp_multi_stage_v2":
                  task_thresholds[name] = ge_defaults.get(name, 0.5)
                else:
                  labels, scores, _ = task_views[name]
                  task_thresholds[name] = compute_threshold_by_metric(labels, scores, metric="youden")
              else:
                task_thresholds[name] = 0.5

          threshold_metrics = evaluate_tasks_with_thresholds(task_views, task_thresholds)
          writer.write(json.dumps({"threshold_metrics": threshold_metrics}, ensure_ascii=False) + "\n")

          header = ["true_grade", "pred_grade"] + [f"p_ge{i}" for i in range(1, k + 1)]
          result_rows = [header]
          for g_t, g_p, p_vals in zip(grades_true, grade_pred, p_np.tolist()):
            result_rows.append([g_t, g_p] + p_vals)
          with open(pred_csv, mode='w', newline='') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerows(result_rows)

          if getattr(args, "output_special", False):
            examples = collect_confusion_examples(task_views, task_thresholds, path_list)
            rows = examples.get("rows", []) if isinstance(examples, dict) else []
            if rows:
              special_csv = os.path.join(model_path, experiment + "_special.csv")
              with open(special_csv, mode='w', newline='') as f:
                writer_csv = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer_csv.writeheader()
                writer_csv.writerows(rows)
          experiment = reader.readline()
          continue

        if test_diseases is not None:
          y_test = copy.deepcopy(y_test[:,test_diseases])
          p_test = copy.deepcopy(p_test[:,test_diseases])

        mAUC, auc_scores = meanAUC(y_test, p_test)
        mMCC, mcc_scores = meanMCC(y_test, p_test)
        mAP, ap_scores = meanAP(y_test, p_test)
        mF1, f1_scores, optimal_thresholds, recall_scores = meanF1(y_test, p_test)

        print(">> Mean AUC = {:.4f} \nAUC = {}".format(mAUC, np.array2string(np.array(auc_scores), precision=4, separator=',')))
        print(">> Mean MCC = {:.4f} \nMCC = {}".format(mMCC, np.array2string(np.array(mcc_scores), precision=4, separator=',')))
        print(">> Mean AP = {:.4f} \nAP = {}".format(mAP, np.array2string(np.array(ap_scores), precision=4, separator=',')))
        print(">> Mean F1 = {:.4f} \nF1 = {}".format(mF1, np.array2string(np.array(f1_scores), precision=4, separator=',')))
        print(">> Optimal Thresholds = {}".format(np.array2string(np.array(optimal_thresholds), precision=4, separator=',')))
        print(">> Recall = {}".format(np.array2string(np.array(recall_scores), precision=4, separator=',')))
        writer.write(
          "AUC = {}\nMCC = {}\nAP = {}\nF1 = {}\nOptimal Thresholds = {}\nRecall = {}\n".format(
            np.array2string(np.array(auc_scores), precision=4, separator=','),
            np.array2string(np.array(mcc_scores), precision=4, separator=','),
            np.array2string(np.array(ap_scores), precision=4, separator=','),
            np.array2string(np.array(f1_scores), precision=4, separator=','),
            np.array2string(np.array(optimal_thresholds), precision=4, separator=','),
            np.array2string(np.array(recall_scores), precision=4, separator=',')
          )
        )
        writer.write("{}: mAUC = {:.4f}, mMCC = {:.4f}, mAP = {:.4f}, mF1 = {:.4f}\n".format(experiment, mAUC, mMCC, mAP, mF1))



        data = [diseases] if test_diseases is None else [[diseases[d] for d in test_diseases]]
        data = data + p_test.tolist()
        print(len(data[0]),len(data[1]))
        # Write data to CSV file
        with open(pred_csv, mode='w', newline='') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerows(data)

        mean_auc.append(mAUC)
        mean_mcc.append(mMCC)
        mean_ap.append(mAP)
        mean_f1.append(mF1)
        metric_dict["auc"].append(auc_scores)
        metric_dict["mcc"].append(mcc_scores)
        metric_dict["ap"].append(ap_scores)
        metric_dict["f1"].append(f1_scores)
        experiment = reader.readline()
    
      
      data = [diseases] if test_diseases is None else [[diseases[d] for d in test_diseases]]
      if isinstance(y_test, dict):
        y_grade = y_test["grade"].cpu().numpy()
        y_stage = y_test["stage"].cpu().numpy()
        y_concat = np.concatenate([y_grade, y_stage], axis=1)
        data = data + y_concat.tolist()
      else:
        data = data + y_test.tolist()
      print(len(data[0]),len(data[1]))
      # Write data to CSV file
      with open(gt_csv, mode='w', newline='') as file:
          csvwriter = csv.writer(file)
          csvwriter.writerows(data)

      # 序数高血压分级不走多试次汇总逻辑，避免空列表触发后续均值/逐类统计
      if args.data_set in {
        "advCheX_hyp_multi_level",
        "advCheX_hyp_multi_stage_v1",
        "advCheX_hyp_multi_stage_v2",
        "advCheX_hyp_multi_grade_stage_v1",
        "advCheX_hyp_multi_grade_stage_sep_v1",
        "advCheX_hyp_grade_stage_v2",
        "advCheX_hyp_grade_stage_embtab_base",
        "advCheX_hyp_grade_stage_embtab_v2lite",
      }:
        return

      if len(mean_auc) == 0 or len(metric_dict.get("auc", [])) == 0:
        return

      mean_auc,mean_mcc,mean_ap,mean_f1 = np.array(mean_auc),np.array(mean_mcc),np.array(mean_ap),np.array(mean_f1)
      print(">> All trials: mAUC = {}\n mMCC = {}\n mAP = {}\n mF1 = {}\n ".format(np.array2string(mean_auc, precision=4, separator=','),
                                                                                   np.array2string(mean_mcc, precision=4, separator=','),
                                                                                   np.array2string(mean_ap, precision=4, separator=','),
                                                                                   np.array2string(mean_f1, precision=4, separator=',')))
      writer.write("All trials: mAUC  = {}\n mMCC  = {}\n mAP = {}\n mF1 = {}\n ".format(np.array2string(mean_auc, precision=4, separator='\t'),
                                                                            np.array2string(mean_mcc, precision=4, separator='\t'),
                                                                            np.array2string(mean_ap, precision=4, separator='\t'),
                                                                            np.array2string(mean_f1, precision=4, separator='\t')))
      print(">> Mean / STD over All trials: aAUC = {:.4f}({:.4f}) aMCC = {:.4f}({:.4f}) aAP = {:.4f}({:.4f}) aF1 = {:.4f}({:.4f})  \n".format(np.mean(mean_auc), np.std(mean_auc),
                                                                                                                                            np.mean(mean_mcc), np.std(mean_mcc),
                                                                                                                                            np.mean(mean_ap), np.std(mean_ap),
                                                                                                                                            np.mean(mean_f1), np.std(mean_f1)))
      writer.write("Mean / STD over All trials: aAUC = {:.4f}({:.4f}) aMCC = {:.4f}({:.4f}) aAP = {:.4f}({:.4f}) aF1 = {:.4f}({:.4f})  \n".format(np.mean(mean_auc), np.std(mean_auc),
                                                                                                                                                np.mean(mean_mcc), np.std(mean_mcc),
                                                                                                                                                np.mean(mean_ap), np.std(mean_ap),
                                                                                                                                                np.mean(mean_f1), np.std(mean_f1)))
      class_wise_scores = []
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["auc"]))
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["mcc"]))
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["ap"]))
      class_wise_scores.extend(get_classwise_mean_std(metric_dict["f1"]))
      writer.write("AUC/MCC/AP/F1(mean/std): \n{}\n".format(class_wise_scores))

      if args.data_set in ["RSNAPneumonia", "COVIDx"]:
        accuracy = np.array(accuracy)
        print(">> All trials: ACCURACY  = {}".format(np.array2string(accuracy, precision=4, separator=',')))
        writer.write("All trials: ACCURACY  = {}\n".format(np.array2string(accuracy, precision=4, separator='\t')))
      
      
