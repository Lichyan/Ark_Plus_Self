
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

  ordinal_datasets = {"advCheX_hyp_multi_level", "advCheX_hyp_multi_stage_v1"}
  if args.data_set in ordinal_datasets and (getattr(args, "test_time_adjust", False) or getattr(args, "output_special", False)):
    if hasattr(dataset_test, "return_path"):
      dataset_test.return_path = True

  data_loader_test = DataLoader(dataset=dataset_test, batch_size=int(args.batch_size/2), shuffle=False,
                            num_workers=args.workers, pin_memory=True, collate_fn=safe_collate, persistent_workers=False)
  ordinal_thresholds = None
  # training phase
  if args.mode == "train":
    train_weights_path = args.train_weights
    if args.data_set == "advCheX_hyp_multi_stage_v1" and train_weights_path is None:
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
        train_one_epoch(data_loader_train,device, model, criterion, optimizer, epoch)

        val_loss = evaluate(data_loader_val, device,model, criterion)

        y_val_np, p_val_np, val_auc_hyp = None, None, None
        if args.data_set == "advCheX_hyp_multi_level":
          y_val_np, p_val_np = _collect_outputs(model, data_loader_val, device, args)
          val_metrics, _, _ = evaluate_ordinal_tasks(y_val_np, p_val_np)
          val_auc_hyp = val_metrics.get("AUROC_hypertension_vs_non")
          if val_auc_hyp is not None:
            print(f"Epoch {epoch:04d}: val_auc_hypertension={val_auc_hyp:.4f}", flush=True)
          else:
            print(f"Epoch {epoch:04d}: val_auc_hypertension=N/A (single class)", flush=True)

        lr_scheduler.step(val_loss)

        if args.test_every_epoch:
          y_test, p_test = test_model(model, data_loader_test, args)
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
        use_cached = os.path.exists(pred_csv) and os.path.exists(gt_csv) and args.data_set != "advCheX_hyp_multi_level"
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
          y_test = y_test.cpu().numpy()
          p_test = p_test.cpu().numpy()

        if args.data_set in ["RSNAPneumonia", "COVIDx"]:
          acc = accuracy_score(np.argmax(y_test,axis=1),np.argmax(p_test,axis=1))
          print(">>{}: ACCURACY = {}".format(experiment,acc))
          writer.write(
            "{}: ACCURACY = {}\n".format(experiment, np.array2string(np.array(acc), precision=4, separator='\t')))
          accuracy.append(acc)

        
        if args.data_set in ordinal_datasets:
          thresholds_src = _load_ordinal_thresholds(saved_model, args)
          thresholds_use = thresholds_src.get('youden') if isinstance(thresholds_src, dict) else None
          y_np = y_test if isinstance(y_test, np.ndarray) else y_test
          p_np = p_test if isinstance(p_test, np.ndarray) else p_test
          k = p_np.shape[1]
          if getattr(args, "test_time_adjust", False):
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
              elif name in ["lv1_vs_non", "lv2_vs_non", "lv3_vs_non", "stage1_vs_non", "stage2_vs_non"]:
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
      data = data + y_test.tolist()
      print(len(data[0]),len(data[1]))
      # Write data to CSV file
      with open(gt_csv, mode='w', newline='') as file:
          csvwriter = csv.writer(file)
          csvwriter.writerows(data)

      # 序数高血压分级不走多试次汇总逻辑，避免空列表触发后续均值/逐类统计
      if args.data_set == "advCheX_hyp_multi_level":
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
      
      
