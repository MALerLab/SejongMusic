import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

class PackedDropout(nn.Module):
  def __init__(self, p=0.25):
    super().__init__()
    self.p = p
    self.dropout = nn.Dropout(p)
    
  def forward(self, x):
    if isinstance(x, PackedSequence):
      return PackedSequence(self.dropout(x[0]), x[1], x[2], x[3])
    return self.dropout(x)