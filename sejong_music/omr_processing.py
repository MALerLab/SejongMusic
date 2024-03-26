import random
import pickle
import torch
import glob
from pathlib import Path
from typing import List, Set, Dict, Tuple, Union
from collections import defaultdict, Counter
from music21 import converter, stream, note as m21_note 
from pathlib import Path
import torch

POSITION = [f":{i}" for i in range(0, 16)]
PITCH = [ '하하배임','하배황', '하배태', '하배중', '하배임', '하배남', '하배무',
          '배황', '배태', '배중', '배임', '배남', '배무',
          '황', '태', '중', '임', '남', '무', 
          '청황', '청태', '청중', '청임', '청남', '청협']
PART = ['daegeum', 'piri', 'haegeum', 'gayageum', 'geomungo', 'ajaeng']

def read_txt(path):
  with open(path, 'r') as file:
    return file.read()
  
class JeongganTokenizer:
    def __init__(self, parts, feature_types=['index', 'token', 'position'], pitch_token=PITCH, position_token=POSITION):
        self.parts = parts
        self.key_types = feature_types
        vocab_list = defaultdict(list)
        special_token = sorted([tok for tok in list(set([note for inst in self.parts for measure in inst for note in measure])) if tok not in pitch_token + position_token+ ['|']])
        vocab_list['index'] = [i for i in range(len(self.parts)+1)]
        vocab_list['token'] = ['start', 'end'] + ['|']+ pitch_token + position_token + special_token
        vocab_list['position'] = ['start', 'end'] + position_token
        self.vocab = vocab_list
        self.tok2idx = {key: {k:i for i, k in enumerate(value)} for key, value in self.vocab.items() }  
        self.note2token = {}  
    
    def hash_note_feature(self, note_feature):
        assert isinstance(note_feature, list)
        out = [self.tok2idx[self.key_types[i]][element] for i, element in enumerate(note_feature)]
        return out
        
    def __call__(self, note_feature:List[Union[int, float, str]]):  
        converted_lists = self.hash_note_feature(note_feature)
        return converted_lists
      

class JeongganDataset:
  def __init__(self, data_path= Path('music_score/gencode'), 
               valid_measure_num = [i for i in range(93, 104)],
               slice_measure_num = 2,
               is_valid=False,
               use_pitch_modification=False,
               pitch_modification_ratio=0.3,
               min_meas=3,
               max_meas=6,
               feature_types=['index', 'token', 'position'],
               part_list = PART, position_token = POSITION, pitch_token = PITCH, 
               target_instrument=0):
    
    self.data_path = data_path
    self.is_valid = is_valid
    self.score = self.get_score(PART)
    self.tokenizer = JeongganTokenizer(self.score, feature_types, pitch_token, position_token)
    self.vocab = self.tokenizer.vocab
    self.validation_split_measure = valid_measure_num
    
    if is_valid:
      self.valid_measure_num = valid_measure_num
      if valid_measure_num == 'entire':
        self.valid_measure_num = list(range(len(self.score[0])))
    else:
      self.valid_measure_num = [i for i in range(len(self.score[0])) if i not in valid_measure_num] 

    self.slice_measure_number = slice_measure_num  
    self.feature_types = feature_types
    self.slice_info = self.update_slice_info(min_meas, max_meas)
    self.target_instrument = target_instrument
    self.condition_instruments = list(range(target_instrument+1, len(self.score)))

    if self.is_valid:
      self.slice_info = [list(range(i, i+self.slice_measure_number)) for i in self.valid_measure_num[:-self.slice_measure_number+1]]
      
    self.position_list = position_token


  def update_slice_info(self, min_meas=4, max_meas=6):
    split_num_list = []
    for i in range(len(self.valid_measure_num)-min_meas):
      random_measure_num = self.slice_measure_number if self.is_valid else random.randint(min_meas, max_meas)
      if i + random_measure_num > len(self.valid_measure_num):
        random_measure_num = len(self.valid_measure_num) - i
      sliced = self.valid_measure_num[i:i+random_measure_num]
      if self.is_valid:
        split_num_list.append(sliced)
        continue
      for meas_num in sliced:
        if meas_num in self.validation_split_measure:
          break
      else:
        split_num_list.append(sliced)
    return split_num_list
  
  def get_score(self, part_list):
    texts = [str(self.data_path / f'{instument}_gencode.txt') for instument in part_list]
    scores = [read_txt(text) for text in texts]
    all_score = []
    for instrument in scores:
      measures = [[s for s in measure.split(' ') if s] for measure in instrument.split('\n')]
      all_score.append(measures)
    return all_score
  
  def get_position_feature(self, note_list):
    new_note_list = []
    hash_note = note_list[0][1]
    for note in note_list:
      if note[1] == '|':
        hash_note = ':0' # check!
      if note[1] in self.position_list:
        hash_note = note[1]
      note += [hash_note]
      new_note_list.append(note)
    return new_note_list  
  
  def get_processed_feature(self, front_part_idx, back_part_idx, idx):
      assert isinstance(front_part_idx, list), "front_part_idx should be a list"
      source_start_token = [front_part_idx[0]] + ['start'] * (len(self.feature_types) - 1)
      source_end_token = [front_part_idx[-1]]+ ['end'] * (len(self.feature_types) - 1)  #맞나?
      target_start_token = [back_part_idx] + ['start'] * (len(self.feature_types) - 1)
      target_end_token = [back_part_idx] + ['end'] * (len(self.feature_types) - 1)
      measure_list = [idx] if self.is_valid else self.slice_info[idx]
      original_source_list = [[f_idx, item] for f_idx in front_part_idx for idx in measure_list for item in self.score[f_idx][idx]]
      original_target_list = [[back_part_idx, item] for idx in measure_list for item in self.score[back_part_idx][idx]]
      source_list = [source_start_token] + self.get_position_feature(original_source_list) + [source_end_token]
      targets = self.get_position_feature(original_target_list)
      target_list = [target_start_token] + targets
      shifted_target_list = targets + [target_end_token]

      source = [self.tokenizer(note_feature) for note_feature in source_list]
      target = [self.tokenizer(note_feature) for note_feature in target_list]
      shifted_target = [self.tokenizer(note_feature) for note_feature in shifted_target_list]
      
      return torch.LongTensor(source), torch.LongTensor(target), torch.LongTensor(shifted_target)    
  
  def __len__(self):
    if self.is_valid:
      return len(self.result_pairs) # TODO: result_paris 만들기
    return len(self.slice_info)-1

  def __getitem__(self, idx):
    front_part_idx = self.condition_instruments
    back_part_idx = self.target_instrument
    src, tgt, shifted_tgt = self.get_processed_feature(front_part_idx, back_part_idx, idx)          
    return src, tgt, shifted_tgt    

if __name__ == '__main__':

  score = JeongganDataset(is_valid=True)
  len(score)