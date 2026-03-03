from utils import MetricLogger, ProgressLogger, corn_marginal_ge_probs
from models import build_classification_model
import time
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

def train_one_epoch(data_loader_train, device,model, criterion, optimizer, epoch):
  batch_time = MetricLogger('Time', ':6.3f')
  losses = MetricLogger('Loss', ':.4e')
  progress = ProgressLogger(
    len(data_loader_train),
    [batch_time, losses],
    prefix="Epoch: [{}]".format(epoch))

  model.train()

  end = time.time()
  component_sums = {}
  component_count = 0
  for i, batch in enumerate(data_loader_train):
    if batch is None:
      continue
    samples, targets = batch
    samples = samples.float().to(device)
    if isinstance(targets, dict):
      targets = {
        k: (v.float() if k in ["y_grade", "y_stage"] else v).to(device)
        if torch.is_tensor(v) else v
        for k, v in targets.items()
      }
    else:
      targets = targets.float().to(device)

    outputs = model(samples)
    loss = criterion(outputs, targets)
    if hasattr(criterion, "last_components") and criterion.last_components is not None:
      for key, val in criterion.last_components.items():
        component_sums[key] = component_sums.get(key, 0.0) + float(val)
      component_count += 1

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.update(loss.item(), samples.size(0))
    batch_time.update(time.time() - end)
    end = time.time()

    if i % 50 == 0:
      progress.display(i)

  if component_count > 0:
    avg_components = {k: v / component_count for k, v in component_sums.items()}
    return losses.avg, avg_components
  return losses.avg


def evaluate(data_loader_val, device, model, criterion):
  model.eval()

  with torch.no_grad():
    batch_time = MetricLogger('Time', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
      len(data_loader_val),
      [batch_time, losses], prefix='Val: ')

    end = time.time()
    component_sums = {}
    component_count = 0
    for i, batch in enumerate(data_loader_val):
      if batch is None:
        continue
      samples, targets = batch
      samples = samples.float().to(device)
      if isinstance(targets, dict):
        targets = {
          k: (v.float() if k in ["y_grade", "y_stage"] else v).to(device)
          if torch.is_tensor(v) else v
          for k, v in targets.items()
        }
      else:
        targets = targets.float().to(device)

      outputs = model(samples)
      loss = criterion(outputs, targets)
      if hasattr(criterion, "last_components") and criterion.last_components is not None:
        for key, val in criterion.last_components.items():
          component_sums[key] = component_sums.get(key, 0.0) + float(val)
        component_count += 1

      losses.update(loss.item(), samples.size(0))
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 50 == 0:
        progress.display(i)

  if component_count > 0:
    avg_components = {k: v / component_count for k, v in component_sums.items()}
    return losses.avg, avg_components
  return losses.avg


def test_classification(checkpoint, data_loader_test, device, args):
  print('[DEBUG] ...heyheyhey:test_clasification', flush=True)
  model = build_classification_model(args)

  try:
    modelCheckpoint = torch.load(checkpoint, weights_only=True)
  except Exception:
    modelCheckpoint = torch.load(checkpoint, weights_only=False)
  if isinstance(modelCheckpoint, dict) and 'state_dict' in modelCheckpoint:
    state_dict = modelCheckpoint['state_dict']
  else:
    state_dict = modelCheckpoint
  for k in list(state_dict.keys()):
    if k.startswith('module.'):
      state_dict[k[len("module."):]] = state_dict[k]
      del state_dict[k]

  msg = model.load_state_dict(state_dict)
  assert len(msg.missing_keys) == 0
  print("=> loaded pre-trained model '{}'".format(checkpoint), flush=True)

  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)

  model.eval()
  
  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()
  y_grade_test = torch.FloatTensor().cuda()
  y_stage_test = torch.FloatTensor().cuda()
  p_grade_test = torch.FloatTensor().cuda()
  p_stage_test = torch.FloatTensor().cuda()
  path_list = []
  printed = False

  with torch.no_grad():
    for i, batch in enumerate(tqdm(data_loader_test)):
      if batch is None:
        continue
      if len(batch) == 3:
        samples, targets, paths = batch
      else:
        samples, targets = batch
        paths = None
      if isinstance(targets, dict):
        targets_grade = targets["y_grade"].cuda()
        targets_stage = targets["y_stage"].cuda()
        y_grade_test = torch.cat((y_grade_test, targets_grade), 0)
        y_stage_test = torch.cat((y_stage_test, targets_stage), 0)
      else:
        targets = targets.cuda()
        y_test = torch.cat((y_test, targets), 0)

      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()

      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).cuda())

      out = model(varInput)
      if not printed:
        head = None
        if hasattr(model, 'module') and hasattr(model.module, 'head'):
          head = model.module.head
        elif hasattr(model, 'head'):
          head = model.head
        if head is not None:
          print('[DEBUG] head.weight mean abs:', head.weight.abs().mean().item(), flush=True)
          print('[DEBUG] head.bias sigmoid :',
                torch.sigmoid(head.bias.detach()).cpu().numpy().round(4).tolist(), flush=True)
        print('[DEBUG] first batch input  mean/std:',
              samples.mean().item(), samples.std().item(), flush=True)
        if isinstance(out, tuple):
          print('[DEBUG] first batch output mean/std:',
                out[0].mean().item(), out[0].std().item(), flush=True)
        else:
          print('[DEBUG] first batch output mean/std:',
                out.mean().item(), out.std().item(), flush=True)
        printed = True
      if isinstance(out, tuple):
        out_grade, out_stage = out
        if str(getattr(args, "ordinal_mode", "coral")).lower() == "corn":
          out_grade = corn_marginal_ge_probs(torch.sigmoid(out_grade))
          out_stage = corn_marginal_ge_probs(torch.sigmoid(out_stage))
        else:
          out_grade = torch.sigmoid(out_grade)
          out_stage = torch.sigmoid(out_stage)
        out_grade_mean = out_grade.view(bs, n_crops, -1).mean(1)
        out_stage_mean = out_stage.view(bs, n_crops, -1).mean(1)
        p_grade_test = torch.cat((p_grade_test, out_grade_mean.data), 0)
        p_stage_test = torch.cat((p_stage_test, out_stage_mean.data), 0)
      else:
        if args.data_set in ["RSNAPneumonia", "COVIDx"]:
          out = torch.softmax(out,dim = 1)
        else:
          out = torch.sigmoid(out)
        outMean = out.view(bs, n_crops, -1).mean(1)
        p_test = torch.cat((p_test, outMean.data), 0)
      if paths is not None:
        path_list.extend(list(paths))

  if y_grade_test.numel() > 0:
    y_dict = {"grade": y_grade_test, "stage": y_stage_test}
    p_dict = {"grade": p_grade_test, "stage": p_stage_test}
    if path_list:
      return y_dict, p_dict, path_list
    return y_dict, p_dict
  if path_list:
    return y_test, p_test, path_list
  return y_test, p_test

def test_model(model, data_loader_test, args):
  print('[DEBUG] ...heyheyhey:test_model', flush=True)
  model.eval()
  
  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()
  y_grade_test = torch.FloatTensor().cuda()
  y_stage_test = torch.FloatTensor().cuda()
  p_grade_test = torch.FloatTensor().cuda()
  p_stage_test = torch.FloatTensor().cuda()
  printed = False

  with torch.no_grad():
    for i, batch in enumerate(tqdm(data_loader_test)):
      if batch is None:
        continue
      samples, targets = batch
      if isinstance(targets, dict):
        targets_grade = targets["y_grade"].cuda()
        targets_stage = targets["y_stage"].cuda()
        y_grade_test = torch.cat((y_grade_test, targets_grade), 0)
        y_stage_test = torch.cat((y_stage_test, targets_stage), 0)
      else:
        targets = targets.cuda()
        y_test = torch.cat((y_test, targets), 0)

      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()

      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).cuda())

      out = model(varInput)
      if not printed:
        head = None
        if hasattr(model, 'module') and hasattr(model.module, 'head'):
          head = model.module.head
        elif hasattr(model, 'head'):
          head = model.head
        if head is not None:
          print('[DEBUG] head.weight mean abs:', head.weight.abs().mean().item(), flush=True)
          print('[DEBUG] head.bias sigmoid :',
                torch.sigmoid(head.bias.detach()).cpu().numpy().round(4).tolist(), flush=True)
        print('[DEBUG] first batch input  mean/std:',
              samples.mean().item(), samples.std().item(), flush=True)
        if isinstance(out, tuple):
          print('[DEBUG] first batch output mean/std:',
                out[0].mean().item(), out[0].std().item(), flush=True)
        else:
          print('[DEBUG] first batch output mean/std:',
                out.mean().item(), out.std().item(), flush=True)
        printed = True
      if isinstance(out, tuple):
        out_grade, out_stage = out
        if str(getattr(args, "ordinal_mode", "coral")).lower() == "corn":
          out_grade = corn_marginal_ge_probs(torch.sigmoid(out_grade))
          out_stage = corn_marginal_ge_probs(torch.sigmoid(out_stage))
        else:
          out_grade = torch.sigmoid(out_grade)
          out_stage = torch.sigmoid(out_stage)
        out_grade_mean = out_grade.view(bs, n_crops, -1).mean(1)
        out_stage_mean = out_stage.view(bs, n_crops, -1).mean(1)
        p_grade_test = torch.cat((p_grade_test, out_grade_mean.data), 0)
        p_stage_test = torch.cat((p_stage_test, out_stage_mean.data), 0)
        outMean = out_grade_mean
      else:
        if args.data_set in ["RSNAPneumonia", "COVIDx"]:
          out = torch.softmax(out,dim = 1)
        else:
          out = torch.sigmoid(out)
        outMean = out.view(bs, n_crops, -1).mean(1)
      if i < 3:
        diffs = (outMean - outMean[0]).abs().max().item()
        print(f"[DEBUG] batch {i} max_diff_to_sample0={diffs:.6f}", flush=True) #检查 batch 内预测是否几乎相同
      if isinstance(out, tuple):
        p_test = torch.cat((p_test, outMean.data), 0)
      else:
        p_test = torch.cat((p_test, outMean.data), 0)

  if y_grade_test.numel() > 0:
    y_dict = {"grade": y_grade_test, "stage": y_stage_test}
    p_dict = {"grade": p_grade_test, "stage": p_stage_test}
    return y_dict, p_dict
  return y_test, p_test
