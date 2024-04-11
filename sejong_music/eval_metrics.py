from collections import Counter
import numpy as np
import torch
from sejong_music.jg_code import JeongganPiece
from sejong_music.constants import PITCH


def per_jg_note_acc(pred, gt, inst, tokenizer, 
                    num_frame_per_jg=6, strict=True, pitch_only=False):
  '''
  Calculates per-jeonggan note accuracy.

  Args:
    pred(List[int]): list of token indices.
    gt(List[int]): list of token indices.
    inst(str): current instrument.
    tokenizer(JeongganTokenizer): tokenizer from current dataset.
    num_frame_per_jg(int): number of beat frames per jeonggan.
    strict(bool): If True, onsets need to be in the same beat position to be considered correct.
      If False, onsets only need to be in the same jeonggan.
    pitch_only(bool): If True, ornamentations in note positions are ignored.

  Returns:
    Accuracy(float)
  '''
  if isinstance(pred, torch.Tensor):
    pred = tokenizer.decode(pred)
  if isinstance(gt, torch.Tensor):
    gt = tokenizer.decode(gt)
  
  pred_roll = JeongganPiece.convert_tokens_to_roll(pred,
                                                   inst=inst,
                                                   num_frame_per_jg=num_frame_per_jg)
  gt_roll = JeongganPiece.convert_tokens_to_roll(gt,
                                                 inst=inst,
                                                 num_frame_per_jg=num_frame_per_jg)
  assert pred_roll.shape == gt_roll.shape, f"Length of pred and gt must be the same. {pred_roll.shape} != {gt_roll.shape}"
  
  if pitch_only:
    num_total = np.isin(gt_roll[:, 0], PITCH).sum()
  else:
    num_total = (gt_roll[:, 0] != '-').sum()
  num_correct = 0
  for i in range(0, pred_roll.shape[0], num_frame_per_jg):
    pred_chunk = pred_roll[i:i+num_frame_per_jg, 0]
    gt_chunk = gt_roll[i:i+num_frame_per_jg, 0]
    if pitch_only:
      mask = np.isin(gt_chunk, PITCH)
    else:
      mask = gt_chunk != '-'
    if strict:
      num_correct += (pred_chunk[mask] == gt_chunk[mask]).sum()
    else:
      for note in gt_chunk[mask]:
        if note in pred_chunk:
          num_correct += 1

  return num_correct / num_total


def onset_f1(pred, gt, inst, tokenizer, 
             num_frame_per_jg=6, pitch_only=False):
  '''
  Calculates per-jeonggan onset f1.
  Ornamentations in note positions are also considered as onsets.

  Args:
    pred(List[int]): list of token indices.
    gt(List[int]): list of token indices.
    inst(str): current instrument.
    tokenizer(JeongganTokenizer): tokenizer from current dataset.
    num_frame_per_jg(int): number of beat frames per jeonggan.
    pitch_only(bool): If True, ornamentations in note positions are ignored.

  Returns:
    F1, Precision, Recall(float)
  '''
  if isinstance(pred, torch.Tensor):
    pred = tokenizer.decode(pred)
  if isinstance(gt, torch.Tensor):
    gt = tokenizer.decode(gt)

  pred_roll = JeongganPiece.convert_tokens_to_roll(pred,
                                                   inst=inst,
                                                   num_frame_per_jg=num_frame_per_jg)
  gt_roll = JeongganPiece.convert_tokens_to_roll(gt,
                                                 inst=inst,
                                                 num_frame_per_jg=num_frame_per_jg)
  assert pred_roll.shape == gt_roll.shape, f"Length of pred and gt must be the same. {pred_roll.shape} != {gt_roll.shape}"

  if pitch_only:
    pred_roll = np.isin(pred_roll[:, 0], PITCH)
    gt_roll = np.isin(gt_roll[:, 0], PITCH)
  else:
    pred_roll = pred_roll[:, 0] != '-'
    gt_roll = gt_roll[:, 0] != '-'
  
  tp = (pred_roll & gt_roll).sum()
  fp = (pred_roll & ~gt_roll).sum()
  fn = (~pred_roll & gt_roll).sum()
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f1 = 2 * precision * recall / (precision + recall)

  return f1, precision, recall


def count_cooccurrences(piece:JeongganPiece, ignore_octaves=True):
  '''
  Counts co-occurring notes in a JeongganPiece.

  Args:
    piece: JeongganPiece object.
    ignore_octaves: If True, notes in different octaves are considered the same.

  Returns:
    counts(Counter): {num_notes: num_occurrences}, max_cooccurring_notes(int)
  '''
  rolls = [piece.convert_tokens_to_roll(part, inst) for part, inst in zip(piece.tokenized_parts, piece.inst_list)]
  rolls = [roll[:, 0] for roll in rolls]
  
  l = len(rolls[0])
  for roll in rolls:
    assert len(roll) == l, 'All parts must have the same length'
  
  rolls = np.stack(rolls, axis=1)
  rolls = rolls[np.any(np.isin(rolls, PITCH), axis=1)]
  rolls[~np.isin(rolls, PITCH)] = '-'

  counts = []
  for row in rolls:
    row = row[row != '-']
    if ignore_octaves:
      row = [note[-1] for note in row]
    counts.append(len(set(row)))
  
  counts = Counter(counts)
  return counts, max(counts.keys())