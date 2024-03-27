from typing import List
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from .yeominrak_processing import Tokenizer


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

class MultiEmbedding(nn.Module):
  def __init__(self, vocab_sizes: dict, vocab_param) -> None:
    super().__init__()
    self.layers = []
    embedding_sizes = self.get_embedding_size(vocab_sizes, vocab_param)
    # if isinstance(embedding_sizes, int):
    #   embedding_sizes = [embedding_sizes] * len(vocab_sizes)
    # print(embedding_sizes)
    for vocab_size, embedding_size in zip(vocab_sizes.values(), embedding_sizes):
      if embedding_size != 0:
        self.layers.append(nn.Embedding(vocab_size, embedding_size))
    self.layers = nn.ModuleList(self.layers)

  def forward(self, x):
    # num_embeddings = torch.tensor([x.num_embeddings for x in self.layers])
    # max_indices = torch.max(x, dim=0)[0].cpu()
    # assert (num_embeddings > max_indices).all(), f'num_embeddings: {num_embeddings}, max_indices: {max_indices}'
    return torch.cat([module(x[..., i]) for i, module in enumerate(self.layers)], dim=-1)

  def get_embedding_size(self, vocab_sizes, vocab_param):
    embedding_sizes = [getattr(vocab_param, vocab_key) for vocab_key in vocab_sizes.keys() if vocab_key != 'offset_fraction']
    return embedding_sizes
  
  @property
  def total_size(self):
    return sum([layer.embedding_dim for layer in self.layers])
  
  
def get_emb_total_size(config):
  if 'emb' not in config.model:
    return config
  emb_param = config.model.emb
  config.model.features = list(config.model.emb.keys())[1:-1]
  total_size = 0 
  for key in config.model.features:
    size = int(emb_param[key] * emb_param.emb_size)
    total_size += size
    emb_param[key] = size
  emb_param.total_size = total_size
  # emb_param.measure_starting_point = emb_param['pitch']+emb_param['duration']
  # emb_param.beat_starting_point = emb_param['pitch']+emb_param['duration']+emb_param['measure']
  config.model.emb = emb_param
  return config


class Converter:
  def __init__(self, tokenizer:Tokenizer):
    self.tokenizer = tokenizer
    
  def _convert_note(self, note:List[int]):
    key_types = self.tokenizer.key_types[:len(note)]
    return [self.tokenizer.vocab[key][note[i]] for i, key in enumerate(key_types)]

    # return [int(note[0]), self.vocab['pitch'][note[1]], self.vocab['duration'][note[2]], self.vocab['offset'][note[3]], self.vocab['dynamic'][note[4]], self.vocab['measure_change'][note[5]]]

  def __call__(self, output):
    # print(output) # tensor([[7, 6, 6, 3, 3]])

    return [self._convert_note(note) for note in output]
  
class RollConverter(Converter):
  def __init__(self, tokenizer:Tokenizer, sampling_rate=2):
    super().__init__(tokenizer)
    self.sampling_rate = sampling_rate

  def convert_to_note_event(self, frame_list):
    note_list = []
    prev_pitch = 0
    note_duration = 0
    part_idx = frame_list[0][0]
    for frame in frame_list:
      _, pitch = frame
      if pitch == 0:
        note_duration += 1
        continue
      else:
        if prev_pitch != 0:
          note_list.append([part_idx, prev_pitch, note_duration / self.sampling_rate])
        prev_pitch = pitch
        note_duration = 1
    if prev_pitch != 0:
      note_list.append([part_idx, prev_pitch, note_duration / self.sampling_rate])
    return note_list
  
  def __call__(self, output): #[pitch, in_onset]
    converted = [self._convert_note(note) for note in output]
    converted = self.convert_to_note_event(converted)
    return converted