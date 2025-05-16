import numpy as np
from tqdm.auto import tqdm
from sejong_music.tokenizer import SingleVocabTokenizer
import torch.nn as nn
import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from .module import SumEmbeddingSingleVocab
from .yeominrak_processing import AlignedScore


class BeatPosConverter:
  def __init__(self) -> None:
    self.era_group = [(0, 1), (2, 3, 4, 5), (6, 7)]
    self.era2group = {era:group for group, eras in enumerate(self.era_group) for era in eras}
    self.anchor_points =  [
      {0.0:0.0, 2.0: 2.0, 3.0:3.0, 5.0:6.0, 6.0:8.0, 7.0:9.0, 8.0:10.0},
      {0.0:0.0, 1.0:1.0, 2.0:2.0, 3.0:3.0, 6.0:5.0, 7.0:6.0, 8.0:7.0, 9.0:8.0, 10.0:10.0},
    ]
    self.measure_len = [8, 8, 10, 10, 10, 10, 10, 10]
  
  def __call__(self, beat_pos:float, org_era_idx:int, target_era_idx:int):
    org_group, target_group = self.era2group[org_era_idx], self.era2group[target_era_idx]
    if org_group == target_group:
      return beat_pos
    assert abs(org_group - target_group) == 1, f"Only one group difference is allowed. {org_era_idx} -> {target_era_idx}"
    if beat_pos >= self.measure_len[org_era_idx]:
      num_measures = int(beat_pos // self.measure_len[org_era_idx])
      beat_pos -= num_measures * self.measure_len[org_era_idx]
    else:
      num_measures = 0


    lower_group = min(org_group, target_group)
    
    anchor_points = self.anchor_points[lower_group]
    forward = np.array(list(anchor_points.keys()))
    backward = np.array(list(anchor_points.values()))
    if org_group > target_group:
      backward, forward = forward, backward
    conv_beat_pos = np.interp(beat_pos, forward, backward)
    return conv_beat_pos + num_measures * self.measure_len[target_era_idx]
    
                          



class EraAlignPairSet:
  pos_conv = BeatPosConverter()
  
  def __init__(self, aligned_score:AlignedScore, slice_len:int=4, hop_len:int=1):
    if isinstance(aligned_score, AlignedScore):
      self.aligned_score = aligned_score
    else:
      self.aligned_score = AlignedScore()
    self.slice_len = slice_len
    self.hop_len = hop_len
    self.data_pairs = self.collect_dataset()
    self.tokenizer = SingleVocabTokenizer(unique_tokens=self.get_unique_words())
      
  def collect_dataset(self):
    total_outputs = []
    for front_idx in range(1, len(self.aligned_score.measure_features)-1):
      back_idx = front_idx + 1
      for start_measure in range(0, len(self.aligned_score.measure_features[front_idx])-self.slice_len, self.hop_len):
        end_measure = start_measure + self.slice_len
        pairs = self.get_align_pair_by_part_and_measure(front_idx, back_idx, start_measure, end_measure)
        notes = [self._convert_note_feature_to_str_list(x[0]) for x in pairs]
        labels = [x[1] for x in pairs]
        total_outputs.append([*zip(notes, labels)])
    total_outputs = [x for x in total_outputs if len(x)>0]
    return total_outputs
  
  def get_unique_words(self):
    unique_words = set()
    for sequence in self.data_pairs:
      notes = [x[0] for x in sequence]
      for note in notes:
        unique_words.update(note)
    return unique_words

  def get_align_pair_by_part_and_measure(self, front_idx:int, back_idx:int, start_measure:int, end_measure:int):
    assert front_idx < back_idx, f"front_idx should be smaller than back_idx. {front_idx} >= {back_idx}"
    assert start_measure < end_measure, f"start_measure should be smaller than end_measure. {start_measure} >= {end_measure}"
    # if len(self.aligned_score.measure_features[front_idx]) < start_measure: return []
    # if len(self.aligned_score.measure_features[back_idx]) < start_measure: return []
    front_part = self.aligned_score.measure_features[front_idx][start_measure:end_measure]
    front_part = [note for measure in front_part for note in measure]
    back_part = self.aligned_score.measure_features[back_idx][start_measure:end_measure]
    back_part = [note for measure in back_part for note in measure]

    front_part_duration = sum([x[2] for x in front_part])
    back_part_duration = sum([x[2] for x in back_part])
    # print(f"Front duration: {front_part_duration}, Back duration: {back_part_duration}")
    if front_part_duration != self.pos_conv.measure_len[front_idx] * (end_measure - start_measure): return []
    if back_part_duration != self.pos_conv.measure_len[back_idx] * (end_measure - start_measure): return []

    f_accum_position = np.cumsum([0] + [note[2] for note in front_part])
    b_accum_position = np.cumsum([0] + [note[2] for note in back_part])
    pairs = []
    b_note_id = 0
    for f_note_id, f_note in enumerate(front_part):
      f_pos = self.pos_conv(f_accum_position[f_note_id], front_idx, back_idx)
      b_pos = b_accum_position[b_note_id]
      while f_pos > b_pos and b_note_id < len(back_part)-1:
        b_note_id += 1
        b_pos = b_accum_position[b_note_id]
      b_note = back_part[b_note_id]
      is_transferred = f_note[1] == b_note[1]
      pairs.append( (f_note, is_transferred))
    return pairs
  
  def __len__(self):
    return len(self.data_pairs)
  
  @staticmethod
  def convert_duration(dur:float):
    if dur < 0.5: return 'veryshort'
    elif dur < 1.0: return 'short'
    elif dur < 2.0: return 'long'
    else: return 'verylong'
  
  def _convert_note_feature_to_str_list(self, note):
    return [f"era{note[0]}", f"pitch{int(note[1])}", self.convert_duration(note[2]), note[4]]
  
  def __getitem__(self, idx):
    sequence = self.data_pairs[idx]
    notes = self.tokenizer([x[0] for x in sequence])
    labels = [x[1] for x in sequence]
    
    return torch.LongTensor(notes), torch.tensor(labels, dtype=torch.float)


class AlignPredictor(nn.Module):
  def __init__(self, vocab_size, hidden_size=32, dropout=0.2):
    super(AlignPredictor, self).__init__()
    self.emb = SumEmbeddingSingleVocab(vocab_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=2, dropout=dropout)
    self.fc = nn.Linear(hidden_size, 1)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    if isinstance(x, PackedSequence):
      emb = PackedSequence(self.dropout(self.emb(x.data)), x[1], x[2], x[3])
      hidden, _ = self.gru(emb)
      return PackedSequence(self.fc(self.dropout(hidden.data)).sigmoid(), x[1], x[2], x[3])
    emb = self.dropout(self.emb(x))
    hidden, _ = self.gru(emb)
    return self.dropout(self.fc(hidden)).sigmoid()

def pack_collate(raw_batch):
  seq, label = zip(*raw_batch)
  return pack_sequence(seq, enforce_sorted=False), pack_sequence(label, enforce_sorted=False).float()
