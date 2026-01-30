
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
                 joint_loss_use_prior=False, joint_prior=None, joint_prior_alpha=0.2):
        super().__init__()
        self.loss_grade = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_grade)
        self.loss_stage = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_stage)
        self.loss_joint = torch.nn.NLLLoss(weight=joint_ce_weight)
        self.w_grade = w_grade
        self.w_stage = w_stage
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
        self.current_epoch = 0
        self.last_components = None

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

    def forward(self, outputs, targets):
        logits_grade, logits_stage = outputs
        y_grade = targets["y_grade"]
        y_stage = targets["y_stage"]
        loss_grade = self.loss_grade(logits_grade, y_grade)
        loss_stage = self.loss_stage(logits_stage, y_stage)
        loss = self.w_grade * loss_grade + self.w_stage * loss_stage
        loss_incomp = torch.tensor(0.0, device=loss.device)
        loss_joint = torch.tensor(0.0, device=loss.device)

        mean_p_joint_true = torch.tensor(0.0, device=loss.device)
        mean_neglog_p_joint_true = torch.tensor(0.0, device=loss.device)
        if self.use_joint_train and (self.lambda_incomp > 0 or self.lambda_joint > 0):
            ge_g = torch.sigmoid(logits_grade)
            ge_s = torch.sigmoid(logits_stage)
            pG0 = 1 - ge_g[:, 0]
            pG1 = torch.clamp(ge_g[:, 0] - ge_g[:, 1], 0, 1)
            pG2 = torch.clamp(ge_g[:, 1] - ge_g[:, 2], 0, 1)
            pG3 = torch.clamp(ge_g[:, 2], 0, 1)
            pG = torch.stack([pG0, pG1, pG2, pG3], dim=1)
            pS0 = 1 - ge_s[:, 0]
            pS1 = torch.clamp(ge_s[:, 0] - ge_s[:, 1], 0, 1)
            pS2 = torch.clamp(ge_s[:, 1], 0, 1)
            pS = torch.stack([pS0, pS1, pS2], dim=1)
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


def _collect_outputs_multi(model, data_loader, device):
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
            samples = samples.float().to(device)
            y_grade = targets["y_grade"].float().to(device)
            y_stage = targets["y_stage"].float().to(device)
            y_grade_all = torch.cat((y_grade_all, y_grade), 0)
            y_stage_all = torch.cat((y_stage_all, y_stage), 0)
            logits_grade, logits_stage = model(samples)
            p_grade = torch.sigmoid(logits_grade)
            p_stage = torch.sigmoid(logits_stage)
            p_grade_all = torch.cat((p_grade_all, p_grade), 0)
            p_stage_all = torch.cat((p_stage_all, p_stage), 0)
    return (
        y_grade_all.cpu().numpy(),
        y_stage_all.cpu().numpy(),
        p_grade_all.cpu().numpy(),
        p_stage_all.cpu().numpy(),
    )


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

def classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test, test_diseases=None):
  device = torch.device(args.device)
  cudnn.benchmark = True

  model_path = os.path.join(model_path, args.exp_name)

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  if not os.path.exists(output_path):
    os.makedirs(output_path)
  output_file = os.path.join(output_path, args.exp_name + "_results.txt")

  ordinal_datasets = {"advCheX_hyp_multi_level", "advCheX_hyp_multi_stage_v1", "advCheX_hyp_multi_stage_v2"}
  multihead_datasets = {"advCheX_hyp_multi_grade_stage_v1"}
  if args.data_set in ordinal_datasets and (getattr(args, "test_time_adjust", False) or getattr(args, "output_special", False)):
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
    if train_weights_path is not None:
      df_w = pd.read_csv(train_weights_path)
      weight_col = "sample_weight" if "sample_weight" in df_w.columns else "weight"
      weight_map = dict(zip(df_w['Path'], df_w[weight_col]))
      rel_paths = [os.path.relpath(p, args.data_dir) for p in dataset_train.img_list]
      weights = [weight_map.get(rp, 1.0) for rp in rel_paths]
      sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
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
          use_joint_train=getattr(args, "use_joint_train", False),
          lambda_incomp=getattr(args, "lambda_incomp", 0.0),
          lambda_joint=getattr(args, "lambda_joint", 0.0),
          joint_gate=getattr(args, "joint_gate", "htn_only"),
          joint_detach=getattr(args, "joint_detach", "both"),
          joint_ce_weight=joint_ce_weight,
          joint_warmup_epochs=getattr(args, "joint_warmup_epochs", 5),
          incomp_mode=getattr(args, "incomp_mode", "mask_sum"),
          joint_loss_use_prior=getattr(args, "joint_loss_use_prior", False),
          joint_prior=joint_prior,
          joint_prior_alpha=getattr(args, "joint_prior_alpha", 0.2),
        )
        print(
          f"use MultiHeadOrdinalLoss, pos_weight_grade={pos_weight_grade}, pos_weight_stage={pos_weight_stage}",
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
            p.requires_grad = name.startswith('head_grade') or name.startswith('head_stage')
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
      if args.freeze_encoder and not getattr(args, "use_lora", False):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        names = [n for n,p in model.named_parameters() if p.requires_grad]
        print(f"[DEBUG] params total={total}, trainable={trainable}", flush=True)
        print(f"[DEBUG] trainable names: {names}", flush=True)
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
          y_grade_val, y_stage_val, p_grade_val, p_stage_val = _collect_outputs_multi(model, data_loader_val, device)
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
          )
          val_joint = val_metrics.get("joint_exact_acc_pjoint")
          if val_joint is not None:
            print(f"Epoch {epoch:04d}: val_joint_exact_acc_pjoint={val_joint:.4f}", flush=True)
          pG_val = ordinal_probs_to_class_probs(p_grade_val)
          pS_val = ordinal_probs_to_class_probs(p_stage_val)
          grade_pred = np.argmax(pG_val, axis=1)
          stage_pred = np.argmax(pS_val, axis=1)
          joint_probs = compute_joint_distribution(
            pG_val, pS_val, prior=prior, alpha=getattr(args, "joint_prior_alpha", 0.2)
          )
          joint_pred = np.argmax(joint_probs, axis=1)
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
          grade_viol_ge2 = float(np.mean(p_grade_val[:, 1] > p_grade_val[:, 0])) if len(p_grade_val) > 0 else 0.0
          grade_viol_ge3 = float(np.mean(p_grade_val[:, 2] > p_grade_val[:, 1])) if len(p_grade_val) > 0 else 0.0
          stage_viol_ge2 = float(np.mean(p_stage_val[:, 1] > p_stage_val[:, 0])) if len(p_stage_val) > 0 else 0.0
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
              y_grade_val, y_stage_val, p_grade_val, p_stage_val = _collect_outputs_multi(model, data_loader_val, device)
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
      }:
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
      
      
