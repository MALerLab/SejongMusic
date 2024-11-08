import torch

def nll_loss(pred, target, eps=1e-8):
  if pred.ndim == 3:
    pred = pred.flatten(0, 1)
  # if target.ndim == 2:
  #   target = target.flatten(0, 1)
  # assert pred.ndim == 2
  # assert target.ndim == 1

  return -torch.log(pred[torch.arange(len(target)), target] + eps).mean()

def focal_loss(pred, target, eps=1e-8, alpha=0.25, gamma=2):
  if pred.ndim == 3:
    pred = pred.flatten(0, 1)
  
  predicted_prob = pred[torch.arange(len(target)), target]
  return (-alpha * (1 - predicted_prob) ** gamma * torch.log(predicted_prob + eps)).mean()

def nll_loss_transformer(pred, target, eps=1e-8):
  if pred.ndim == 3:
    pred = pred.flatten(0, 1)
  if target.ndim > 1:
    target = target.flatten()
  mask = target != 0
  mask = mask.to(pred.device)
  return (-torch.log(pred[torch.arange(len(target)), target] + eps) * mask).sum() / mask.sum()

def nll_loss_transformer_with_logsoftmax(pred, target, eps=1e-8):
  if pred.ndim == 3:
    pred = pred.flatten(0, 1)
  if target.ndim > 1:
    target = target.flatten()
  mask = target != 0
  mask = mask.to(pred.device)
  return (-pred[torch.arange(len(target)), target] * mask).sum() / mask.sum()


def nll_loss_bert(pred, target, mask):
  assert pred.shape[:2] == target.shape[:2] == mask.shape[:2]
  pred_flatten = pred.flatten(0, 1)
  pred_flatten = -torch.log_softmax(pred_flatten, dim=-1)
  target_flatten = target.flatten(0, 1)
  mask_flatten = mask.flatten(0, 1)
  losses = []
  for i in range(2):
    loss = pred_flatten[:,i][torch.arange(pred_flatten.shape[0]), target_flatten[:,i]]
    loss = loss * mask_flatten[:,i]
    num_valid = mask_flatten[:,i].sum()
    if num_valid == 0:
      continue
    loss = loss.sum() / mask_flatten[:,i].sum()
    losses.append(loss)
  losses = sum(losses) / 2
  return losses