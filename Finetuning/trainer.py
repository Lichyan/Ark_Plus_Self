from utils import MetricLogger, ProgressLogger
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
  for i, batch in enumerate(data_loader_train):
    if batch is None:
      continue
    samples, targets = batch
    samples, targets = samples.float().to(device), targets.float().to(device)
    
    outputs = model(samples)
    #outputs = torch.sigmoid(outputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.update(loss.item(), samples.size(0))
    batch_time.update(time.time() - end)
    end = time.time()

    if i % 50 == 0:
      progress.display(i)


def evaluate(data_loader_val, device, model, criterion):
  model.eval()

  with torch.no_grad():
    batch_time = MetricLogger('Time', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
      len(data_loader_val),
      [batch_time, losses], prefix='Val: ')

    end = time.time()
    for i, batch in enumerate(data_loader_val):
      if batch is None:
        continue
      samples, targets = batch
      samples, targets = samples.float().to(device), targets.float().to(device)

      outputs = model(samples)
      #outputs = torch.sigmoid(outputs)
      loss = criterion(outputs, targets)

      losses.update(loss.item(), samples.size(0))
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 50 == 0:
        progress.display(i)

  return losses.avg


def test_classification(checkpoint, data_loader_test, device, args):
  print('[DEBUG] ...heyheyhey:test_clasification', flush=True)
  model = build_classification_model(args)

  modelCheckpoint = torch.load(checkpoint, weights_only=True)
  state_dict = modelCheckpoint['state_dict']
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
  printed = False

  with torch.no_grad():
    for i, batch in enumerate(tqdm(data_loader_test)):
      if batch is None:
        continue
      samples, targets = batch
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
        h = model.module.head if hasattr(model, 'module') else model.head
        print('[DEBUG] head.weight mean abs:', h.weight.abs().mean().item(), flush=True)
        print('[DEBUG] head.bias sigmoid :',
              torch.sigmoid(h.bias.detach()).cpu().numpy().round(4).tolist(), flush=True)
        print('[DEBUG] first batch input  mean/std:',
              samples.mean().item(), samples.std().item(), flush=True)
        print('[DEBUG] first batch output mean/std:',
              out.mean().item(), out.std().item(), flush=True)
        printed = True
      if args.data_set in ["RSNAPneumonia", "COVIDx"]:
        out = torch.softmax(out,dim = 1)
      else:
        out = torch.sigmoid(out)
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test

def test_model(model, data_loader_test, args):
  print('[DEBUG] ...heyheyhey:test_model', flush=True)
  model.eval()
  
  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()
  printed = False

  with torch.no_grad():
    for i, batch in enumerate(tqdm(data_loader_test)):
      if batch is None:
        continue
      samples, targets = batch
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
        h = model.module.head if hasattr(model, 'module') else model.head
        print('[DEBUG] head.weight mean abs:', h.weight.abs().mean().item(), flush=True)
        print('[DEBUG] head.bias sigmoid :',
              torch.sigmoid(h.bias.detach()).cpu().numpy().round(4).tolist(), flush=True)
        print('[DEBUG] first batch input  mean/std:',
              samples.mean().item(), samples.std().item(), flush=True)
        print('[DEBUG] first batch output mean/std:',
              out.mean().item(), out.std().item(), flush=True)
        printed = True
      if args.data_set in ["RSNAPneumonia", "COVIDx"]:
        out = torch.softmax(out,dim = 1)
      else:
        out = torch.sigmoid(out)
      outMean = out.view(bs, n_crops, -1).mean(1)
      if i < 3:
        diffs = (outMean - outMean[0]).abs().max().item()
        print(f"[DEBUG] batch {i} max_diff_to_sample0={diffs:.6f}", flush=True) #检查 batch 内预测是否几乎相同
      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test
