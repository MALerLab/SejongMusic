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

def get_per_token_nll_from_logits(logits, targets, pad_token_id=0):
    """
    Calculates per-token NLL loss from raw logits.

    Args:
        logits (torch.Tensor): Raw output from the model. 
                               Shape: (batch_size, seq_len, vocab_size)
        targets (torch.Tensor): Ground truth token IDs. 
                                Shape: (batch_size, seq_len)
        pad_token_id (int): ID of the padding token to ignore in loss.

    Returns:
        torch.Tensor: Per-token NLL losses. 
                      Shape: (batch_size, seq_len)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Apply log_softmax to convert logits to log-probabilities
    log_probs = torch.log_softmax(logits, dim=-1)

    # Flatten logits and targets to gather easily
    # log_probs_flat: (batch_size * seq_len, vocab_size)
    # targets_flat: (batch_size * seq_len)
    log_probs_flat = log_probs.view(batch_size * seq_len, vocab_size)
    targets_flat = targets.view(batch_size * seq_len)

    # Gather the log-probabilities of the target tokens
    # selected_log_probs: (batch_size * seq_len)
    selected_log_probs = log_probs_flat[torch.arange(targets_flat.size(0)), targets_flat]

    # NLL loss is the negative of these selected log-probabilities
    # token_losses_flat: (batch_size * seq_len)
    token_losses_flat = -selected_log_probs

    # Create mask for padding tokens
    # mask_flat: (batch_size * seq_len)
    mask_flat = (targets_flat != pad_token_id).float()

    # Apply mask (set loss to 0 for padding tokens)
    masked_token_losses_flat = token_losses_flat * mask_flat

    # Reshape back to (batch_size, seq_len)
    per_token_losses = masked_token_losses_flat.view(batch_size, seq_len)

    return per_token_losses