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