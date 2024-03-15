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
  

class SumEmbedding(nn.Module):
  def __init__(self, vocab_sizes: dict, emb_size: int) -> None:
    super().__init__()
    self.layers = []
    self.emb_size = emb_size
    for vocab_size in vocab_sizes.values():
      self.layers.append(nn.Embedding(vocab_size, self.emb_size))
    self.layers = nn.ModuleList(self.layers)
  
  def forward(self, x):
    return torch.sum(torch.stack([module(x[..., i]) for i, module in enumerate(self.layers)], dim=-1), dim=-1)
