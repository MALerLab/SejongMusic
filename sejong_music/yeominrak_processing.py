import random
import pickle
import torch

from pathlib import Path
from typing import List, Set, Dict, Tuple, Union

from music21 import converter, stream, note as m21_note 

import torch

from .utils import SimpleNote, Part, Gnote, make_offset_set, as_fraction, FractionEncoder, pack_collate, pad_collate, round_number
from .tokenizer import Tokenizer

class AlignedScore:
  def __init__(self, xml_path='music_score/yeominlak.musicxml', 
               valid_measure_num = [i for i in range(93, 104)],
               slice_measure_num = 2, 
               is_valid = False, 
               use_pitch_modification=False, 
               pitch_modification_ratio=0.0, 
               min_meas=3, 
               max_meas=6,
               feature_types=['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx'],
               sampling_rate=None,
               start_part_idx=0) -> None:
        
    self.score = converter.parse(xml_path)
    # self.parts = [Part(part, i+start_part_idx) for i, part in enumerate(self.score.parts)]
    if Path(xml_path).with_suffix('.pkl').exists():
        self.parts = pickle.load(open(Path(xml_path).with_suffix('.pkl'), 'rb'))
    else:
        self.parts = [Part(part, i+start_part_idx) for i, part in enumerate(self.score.parts)]
        pickle.dump(self.parts, open(Path(xml_path).with_suffix('.pkl'), 'wb'))
    self.offset_list = [part_list.measure_duration for part_list in self.parts]
    self.feature_types = feature_types
    self.tokenizer = Tokenizer(self.parts, feature_types)
    self.vocab = self.tokenizer.vocab
    self.vocab_size_dict = self.tokenizer.vocab_size_dict
    self.maximum_era_diff = 2

    self.measure_offset_vocab = []
    self.measure_features = [self.get_feature(i) for i in range(len(self.parts))]
    max_len = max([len(part) for part in self.measure_features])
    self.measure_features = [part + [[] for _ in range(max_len - len(part))] for part in self.measure_features]
    self.fix_part_by_rule()
    self.slice_measure_number = slice_measure_num
    
    self.is_valid = is_valid
    self.use_pitch_modification = use_pitch_modification
    self.pitch_modification_ratio = pitch_modification_ratio
    self.min_meas = min_meas
    self.max_meas = max_meas
    self.validation_split_measure = valid_measure_num
    
    if is_valid:
      self.valid_measure_num = valid_measure_num
      if valid_measure_num == 'entire':
         self.valid_measure_num = list(range(len(self.measure_features[0])))
    else:
      self.valid_measure_num = [i for i in range(len(self.measure_features[0])) 
                                if i not in valid_measure_num 
                                # and (i > valid_measure_num[-1] or i+max_meas <= valid_measure_num[0])
                                ] # 남은 trainset 마디 개수!
    
    self.slice_info = self.update_slice_info(min_meas, max_meas)
    valid_list = [0,1,2,3,5,6,7]
    self.result_pairs = [[a, b] for a in valid_list for b in valid_list if b <= a + self.maximum_era_diff and b>a]
    self.result_pairs = [pair+[i]  for pair in self.result_pairs for i in self.slice_info]
  
  def _get_feature_by_note(self, note:Gnote, part_idx):
      features = [part_idx]
      # if 'pitch' in self.tokenizer.key_types:
      features.append(note.pitch)
      # if 'duration' in self.tokenizer.key_types:
      features.append(note.duration)
      if 'offset' in self.tokenizer.key_types:
          features.append(note.measure_offset)
      if 'dynamic' in self.tokenizer.key_types:
          features.append(note.dynamic)
      if 'measure_idx' in self.tokenizer.key_types:
          features.append(note.measure_number)
      if 'measure_change' in self.tokenizer.key_types:
          changed = 1 if note.measure_offset == 0.0 else 0
          features.append(changed)
      return features

  def get_feature(self, part_idx):
    part = self.parts[part_idx]
    measure_set = []
    for measure_idx, measure in enumerate(part.measures):
        each_measure = []
        measure_dur = sum([note.duration for note in measure])
        for note in measure:
            each_measure.append(self._get_feature_by_note(note, part_idx))
        if (part_idx in [0] and abs(measure_dur - 4.0) > 0.1) or (part_idx in [1] and abs(measure_dur - 8.0) > 0.1) or (part_idx in [2,3,4,5,6,7] and abs(measure_dur - 10.0) > 0.1):
            each_measure = []
        measure_set.append(each_measure) 
    return measure_set  
  
  
  def fix_part_by_rule(self):
    # fix part zero duration
    part_zero = self.measure_features[0].copy()
    for measure in part_zero:
      for note in measure:
        note[2] *= 2
        note[3] *= 2
    self.measure_features[0] = part_zero
    self.offset_list[0] *= 2
  
  def update_slice_info(self, min_meas=4, max_meas=6):
    split_num_list = []
    for i in range(len(self.valid_measure_num)-min_meas):
      if self.is_valid: 
        random_measure_num = self.slice_measure_number
      else:
        random_measure_num = random.randint(min_meas, max_meas)
      if i + random_measure_num > len(self.valid_measure_num):
        random_measure_num = len(self.valid_measure_num) - i
      sliced = self.valid_measure_num[i:i+random_measure_num]
      if sliced[-1] - sliced[0] != random_measure_num - 1: # 연속된 마디가 아니면
        continue
      if self.is_valid:
        split_num_list.append(sliced)
        continue
      for meas_num in sliced:
        if meas_num in self.validation_split_measure:
          break
      else:
        split_num_list.append(sliced)
    return split_num_list

#   def update_slice_info(self):
#     numbers = self.valid_measure_num
#     split_num_list = []
#     while numbers:
#       if self.is_valid: 
#         random_measure_num = self.slice_measure_number
#       else:
#         random_measure_num = random.randint(2, 4)
#       split_num_list.append(numbers[:random_measure_num])
#       numbers = numbers[random_measure_num:]
#     return split_num_list

  def __len__(self):
    if self.is_valid:
      return len(self.result_pairs)
    return len(self.slice_info)-1
  
  def modify_pitch(self, target_list):
    ratio = self.pitch_modification_ratio
    modified_list = []
    for note in target_list[1:-1]:
      if random.random() < ratio:
        pitch = random.choice(self.vocab['pitch'][3:])
      else:
        pitch = note[1]
      # new_note = [note[0], pitch, note[2], note[3], note[4], [note[5]]]
    #   new_note = [note[0], pitch, note[2], note[3], note[4], note[5]]
      new_note = [note[0], pitch] + note[2:]
      modified_list.append(new_note)
    completed_list = [target_list[0]] + modified_list + [target_list[-1]]
    return completed_list  

  def get_processed_feature(self, front_part_idx, back_part_idx, idx):
    source_start_token = [front_part_idx, 'start', 'start', 'start', 'start', 'start']
    source_end_token = [front_part_idx, 'end', 'end', 'end', 'end', 'end']
    target_start_token = [back_part_idx, 'start', 'start', 'start', 'start', 'start']
    target_end_token = [back_part_idx, 'end', 'end', 'end', 'end', 'end']
    if self.is_valid:
        measure_list = idx
    else:    
        measure_list = self.slice_info[idx]


    original_source_list = [item for idx in measure_list for item in self.measure_features[front_part_idx][idx]]
    original_target_list = [item for idx in measure_list for item in self.measure_features[back_part_idx][idx]]
    for i, idx in enumerate(measure_list):
        if len(self.measure_features[front_part_idx][idx]) == 0:
            return torch.LongTensor([[]]), torch.LongTensor([[]]), torch.LongTensor([[]])
        if len(self.measure_features[back_part_idx][idx]) == 0:
            return torch.LongTensor([[]]), torch.LongTensor([[]]), torch.LongTensor([[]])
        
        # if 'measure_idx' in self.tokenizer.key_types:
        #   for item in self.measure_features[front_part_idx][idx]:
        #       original_source_list.append(item+[i])
        #   for item in self.measure_features[back_part_idx][idx]:
        #       original_target_list.append(item+[i])
        # else:
        #   for item in self.measure_features[front_part_idx][idx]:
        #       original_source_list.append(item)
        #   for item in self.measure_features[back_part_idx][idx]:
        #       original_target_list.append(item)

    # 여기서 measure_diff 및 measure_idx를 집어넣자!
    
    # original_source_list = [copy.copy(item) for idx in measure_list for item in self.measure_features[front_part_idx][idx]]
    # original_target_list = [copy.copy(item) for idx in measure_list for item in self.measure_features[back_part_idx][idx]]

    if 'measure_idx' in self.tokenizer.key_types:
      m_idx_pos = self.tokenizer.key2idx['measure_idx']
      source_first_measure_idx = original_source_list[0][m_idx_pos]
      target_first_measure_idx = original_target_list[0][m_idx_pos]

      original_source_list = [note[:m_idx_pos] + [note[m_idx_pos]-source_first_measure_idx] + note[m_idx_pos+1:] for note in original_source_list]
      original_target_list = [note[:m_idx_pos] + [note[m_idx_pos]-target_first_measure_idx] + note[m_idx_pos+1:] for note in original_target_list]

    
    source_list = [source_start_token] + original_source_list + [source_end_token]
    target_list = [target_start_token] + original_target_list
    if self.use_pitch_modification and not self.is_valid:
        target_list = self.modify_pitch(target_list)
    shifted_target_list = original_target_list + [target_end_token]
    # print(target_list, shifted_target_list)
    source = [self.tokenizer(note_feature) for note_feature in source_list]
    target = [self.tokenizer(note_feature) for note_feature in target_list]
    shifted_target = [self.tokenizer(note_feature) for note_feature in shifted_target_list]
    
    return torch.LongTensor(source), torch.LongTensor(target), torch.LongTensor(shifted_target)
    
  def __getitem__(self, idx):
    if self.is_valid:
      front_part_idx, back_part_idx, measure_idx = self.result_pairs[idx]
      src, tgt, shifted_tgt = self.get_processed_feature(front_part_idx, back_part_idx, measure_idx)
      
      return src, tgt, shifted_tgt
    
    sample_success = False
    num_try = 0
    while not sample_success:
      front_part_idx = random.choice(range(len(self.parts)-1))
      # back_part_idx should be bigger than front_part_idx
      back_part_idx = random.randint(front_part_idx + 1, min(len(self.parts) - 1, front_part_idx + self.maximum_era_diff))
      src, tgt, shifted_tgt = self.get_processed_feature(front_part_idx, back_part_idx, idx)          
      if len(src) > 2 and len(tgt) > 1:
        sample_success = True
      # num_try += 1
    return src, tgt, shifted_tgt    
    


class ShiftedAlignedScore(AlignedScore):
    def __init__(self, xml_path='0_edited.musicxml', 
                 valid_measure_num = [i for i in range(93, 104)],
                 slice_measure_num = 2, 
                 is_valid = False, 
                 use_pitch_modification=False, 
                 pitch_modification_ratio=0.3, 
                 min_meas=3, 
                 max_meas=6,
                 feature_types=['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx'],
                 sampling_rate=None,
                 start_part_idx=0) -> None:
        super().__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, feature_types, sampling_rate, start_part_idx)


    
    def shift_condition(self, note_list, shift_start=3):
        num_features = len(note_list[0])
        shift_idxs = list(range(shift_start, num_features))
        non_shift_idxs = [i for i in range(len(note_list[0])) if i not in shift_idxs]
        pred_features = [ [x[i] for i in non_shift_idxs] for x in note_list]
        conditions = [ [x[i] for i in shift_idxs] for x in note_list] 
        
        pred_features = [[note_list[0][0], 'start', 'start']] + pred_features
        # conditions = conditions +  [[0, 'strong', note_list[-1][-1]+1]]
        last_condition = []
        if 'offset' in self.tokenizer.key_types:
            last_condition.append(0)
        if 'daegang_offset' in self.tokenizer.key_types:
            last_condition += [0, 0, 0]
        if 'dynamic' in self.tokenizer.key_types:
            last_condition.append('strong')
        if 'measure_idx' in self.tokenizer.key_types:
            last_condition.append(note_list[-1][-1]+1)
        if 'measure_change' in self.tokenizer.key_types:
            last_condition.append(1)
        conditions = conditions +  [last_condition]
        
        combined = [ pred_features[i] + conditions[i] for i in range(len(pred_features))] + [[note_list[-1][0]] + ['end'] * (num_features-1) ]
        
        return combined
    
    def get_processed_feature(self, front_part_idx, back_part_idx, idx):
        source_start_token = [front_part_idx, 'start', 'start', 'start', 'start', 'start']
        source_end_token = [front_part_idx, 'end', 'end', 'end', 'end', 'end']
        # target_start_token = [back_part_idx, 'start', 'start', 0, 'strong']
        # target_end_token = [back_part_idx, 'end', 'end', 0, 'strong']
        if self.is_valid:
            measure_list = idx
        else:    
            measure_list = self.slice_info[idx]
            
        for i, idx in enumerate(measure_list):
            if len(self.measure_features[front_part_idx][idx]) == 0:
                return torch.LongTensor([[]]), torch.LongTensor([[]]), torch.LongTensor([[]])
            if len(self.measure_features[back_part_idx][idx]) == 0:
                return torch.LongTensor([[]]), torch.LongTensor([[]]), torch.LongTensor([[]])


        original_source_list = [item for idx in measure_list for item in self.measure_features[front_part_idx][idx]]
        original_target_list = [item for idx in measure_list for item in self.measure_features[back_part_idx][idx]]

        if 'measure_idx' in self.tokenizer.key_types:
          m_idx_pos = self.tokenizer.key2idx['measure_idx']
          source_first_measure_idx = original_source_list[0][m_idx_pos]
          target_first_measure_idx = original_target_list[0][m_idx_pos]

          original_source_list = [note[:m_idx_pos] + [note[m_idx_pos]-source_first_measure_idx] + note[m_idx_pos+1:] for note in original_source_list]
          original_target_list = [note[:m_idx_pos] + [note[m_idx_pos]-target_first_measure_idx] + note[m_idx_pos+1:] for note in original_target_list]
        
        condition_shifted_target = self.shift_condition(original_target_list)
        target_list = condition_shifted_target[: -1]
        
        if self.use_pitch_modification and not self.is_valid:
            target_list = self.modify_pitch(target_list)
        shifted_target_list = condition_shifted_target[1:]
        
        source_list = [source_start_token] + original_source_list + [source_end_token]
        
        source = [self.tokenizer(note_feature) for note_feature in source_list]
        target = [self.tokenizer(note_feature) for note_feature in target_list]
        shifted_target = [self.tokenizer(note_feature) for note_feature in shifted_target_list]
        
        return torch.LongTensor(source), torch.LongTensor(target), torch.LongTensor(shifted_target)        
        

class TestScore(AlignedScore):
    def __init__(self, xml_path='0_edited.musicxml', 
                 valid_measure_num=[i for i in range(93, 104)], 
                 slice_measure_num=2, 
                 is_valid=False, 
                 use_pitch_modification=False, 
                 pitch_modification_ratio=0, 
                 min_meas=3, 
                 max_meas=6, 
                 transpose=0, 
                 feature_types=['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx'],
                 sampling_rate=None) -> None:
        # DYNAMIC_MAPPING[0] = {0.0: 'strong', 1.5: 'strong', 3.0: 'middle'}
        super().__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, feature_types)
        self.transpose_value = transpose
        self.transpose()
        self.result_pairs = self.result_pairs[:len(self.slice_info)]

    # def fix_part_zero_duration(self):
    #     return
    def fix_part_by_rule(self):
        flattened_notes = [SimpleNote(note) for measure in self.parts[0].measures for note in measure]
                  
        measure_shifted_notes = []
        for note in flattened_notes:
          new_offset = (note.measure_offset - 1.5 )
          new_note = None
          if note.pitch == 44:
            note.pitch = 45 # 44 is error
          if note.duration > 4:
            break
          if new_offset < 0:
            new_offset += 4
            note.measure_number -= 1
            note.end_measure_number -= 1
            if note.measure_number <= 0:
              new_note = copy.copy(note)
              new_note.measure_offset = 0
              new_note.duration = 4 - note.duration
              measure_shifted_notes.append(new_note)
              new_note = None
          note.measure_offset = new_offset
          if note.measure_offset + note.duration > 4:
            # print("note duration is too long", note.measure_offset, note.duration)
            extended_duration = note.measure_offset + note.duration - 4
            note.duration -= extended_duration
            new_note = copy.copy(note)
            new_note.measure_offset = 0
            new_note.measure_number += 1
            new_note.end_measure_number += 1
            new_note.duration = extended_duration
          measure_shifted_notes.append(note)
          if new_note:
            measure_shifted_notes.append(new_note)
        new_note = copy.copy(note)
        new_note.measure_offset += note.duration
        new_note.duration = 1.5
        measure_shifted_notes.append(new_note)

        measure_shifted_notes.sort(key=lambda x: (x.measure_number, x.measure_offset))

        # make measure list
        note_by_measure = []
        temp_measure = []
        prev_measure_number = 0
        for note in measure_shifted_notes:
          if note.measure_number != prev_measure_number:
            note_by_measure.append(temp_measure)
            temp_measure = []
            prev_measure_number = note.measure_number
          temp_measure.append(note)
        note_by_measure.append(temp_measure)

        self.parts[0].measures = note_by_measure
        self.measure_features = [self.get_feature(i) for i in range(len(self.parts))]

        part_zero = self.measure_features[0].copy()
        for measure in part_zero:
          for note in measure:
            note[2] *= 2
            note[3] *= 2
        self.measure_features[0] = part_zero
        self.offset_list[0] *= 2


    def transpose(self):
      for idx in range(len(self.parts)):
        part_zero = self.measure_features[idx].copy()
        for measure in part_zero:
          for note in measure:
            note[1] += self.transpose_value
        self.measure_features[idx] = part_zero

    def get_processed_feature(self, front_part_idx, back_part_idx, idx):
      source_start_token = [front_part_idx, 'start', 'start', 'start', 'start', 'start']
      source_end_token = [front_part_idx, 'end', 'end', 'end', 'end', 'end']
      if self.is_valid:
          measure_list = idx
      else:    
          measure_list = self.slice_info[idx]
      original_source_list = [item for idx in measure_list for item in self.measure_features[front_part_idx][idx]]
      if 'measure_idx' in self.tokenizer.key_types:
        m_idx_pos = self.tokenizer.key2idx['measure_idx']
        source_first_measure_idx = original_source_list[0][m_idx_pos]

        original_source_list = [note[:m_idx_pos] + [note[m_idx_pos]-source_first_measure_idx] + note[m_idx_pos+1:] for note in original_source_list]

      source_list = [source_start_token] + original_source_list + [source_end_token]
      source = [self.tokenizer(note_feature) for note_feature in source_list]
      
      return torch.LongTensor(source)
        
    def __getitem__(self, idx):

      front_part_idx, back_part_idx, measure_idx = self.result_pairs[idx]
      src = self.get_processed_feature(front_part_idx, back_part_idx, measure_idx)
      
      return src

class TestScoreCPH(TestScore):
  def __init__(self, xml_path='0_edited.musicxml', valid_measure_num=..., slice_measure_num=2, is_valid=False, use_pitch_modification=False, pitch_modification_ratio=0, min_meas=3, max_meas=6, transpose=0, feature_types=..., sampling_rate=None) -> None:
    super().__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, transpose, feature_types, sampling_rate)


  def fix_part_by_rule(self):
    flattened_notes = [SimpleNote(note) for measure in self.parts[0].measures for note in measure]
              
    measure_shifted_notes = []
    for note in flattened_notes:
      new_offset = (note.measure_offset - 2.5 )
      new_note = None

      ### pitch modification
      if note.pitch == 51:
        note.pitch = 52 # 51 is error
      if note.pitch == 39:
        note.pitch = 40
      if note.pitch == 56:
        note.pitch = 57
      if note.pitch == 57:
        note.pitch = 45

      if new_offset < 0:
        new_offset += 4
        note.measure_number -= 1
        note.end_measure_number -= 1
      note.measure_offset = new_offset
      if note.measure_offset + note.duration > 4:
        # print("note duration is too long", note.measure_offset, note.duration)
        extended_duration = note.measure_offset + note.duration - 4
        note.duration -= extended_duration
        new_note = copy.copy(note)
        new_note.measure_offset = 0
        new_note.measure_number += 1
        new_note.end_measure_number += 1
        new_note.duration = extended_duration
      measure_shifted_notes.append(note)
      if new_note:
        measure_shifted_notes.append(new_note)
    new_note = copy.copy(note)
    new_note.measure_offset += note.duration
    new_note.duration = 2.5
    measure_shifted_notes.append(new_note)

    beginning_dummy_note = copy.copy(measure_shifted_notes[0])
    beginning_dummy_note.measure_offset = 0
    beginning_dummy_note.duration = 1.5
    measure_shifted_notes.append(beginning_dummy_note)

    measure_shifted_notes.sort(key=lambda x: (x.measure_number, x.measure_offset))

    # make measure list
    note_by_measure = []
    temp_measure = []
    prev_measure_number = 0
    for note in measure_shifted_notes:
      if note.measure_number != prev_measure_number:
        note_by_measure.append(temp_measure)
        temp_measure = []
        prev_measure_number = note.measure_number
      temp_measure.append(note)
    note_by_measure.append(temp_measure)

    self.parts[0].measures = note_by_measure
    self.measure_features = [self.get_feature(i) for i in range(len(self.parts))]

    part_zero = self.measure_features[0].copy()
    for measure in part_zero:
      for note in measure:
        note[2] *= 2
        note[3] *= 2
    self.measure_features[0] = part_zero
    self.offset_list[0] *= 2

    


class OrchestraScore(ShiftedAlignedScore):
  def __init__(self,xml_path='music_score/FullScore_edit5.musicxml', 
                valid_measure_num = [i for i in range(93, 104)],
                slice_measure_num = 2, 
                is_valid = False, 
                use_pitch_modification=False, 
                pitch_modification_ratio=0.3, 
                min_meas=3, 
                max_meas=6,
                feature_types=['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx'],
                sampling_rate=None,
                target_instrument=None,
                is_sep=True) -> None: 
    self.dynamic_template = {x/2: 'weak' for x in range(0, 90, 3)}
    self.dynamic_template[0.0] =  'strong'
    self.dynamic_template[15.0] =  'strong'
    self.dynamic_template[9.0] =  'middle'
    self.dynamic_template[21.0] =  'middle'
    self.is_sep = is_sep
    if self.is_sep:
        offset_index = feature_types.index('offset')
        feature_types[offset_index:offset_index+1] = ['daegang_offset', 'jeonggan_offset', 'beat_offset']
    super().__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, feature_types, sampling_rate, start_part_idx=1)
    self.era_dataset = ShiftedAlignedScore(xml_path='/home/danbi/userdata/DANBI/gugakwon/SejongMusic/music_score/yeominlak.musicxml', is_valid=is_valid, min_meas=min_meas, max_meas=max_meas, feature_types=feature_types, slice_measure_num=slice_measure_num)
    valid_list = list(range(len(self.parts)))
    self.result_pairs = [[7, b] for b in valid_list]
    self.result_pairs = [pair+[i]  for pair in self.result_pairs for i in self.slice_info]
    
    
  def get_feature(self, part_idx):
    part = self.parts[part_idx]
    measure_set = []
    for measure_idx, measure in enumerate(part.measures):
        each_measure = []
        for note in measure:
            each_measure.append(self._get_feature_by_note(note, part_idx))
        measure_set.append(each_measure) 
    return measure_set  
  
  def _get_feature_by_note(self, note, part_idx):
    features = [part_idx+1] # orchestra score starts from 1
    # if 'pitch' in self.tokenizer.key_types:
    features.append(note.pitch)
    # if 'duration' in self.tokenizer.key_types:
    features.append(note.duration)
    if 'offset' in self.tokenizer.key_types:
        features.append(note.measure_offset)
    if self.is_sep:
        offsets = make_offset_set(note.measure_offset)  # [daegang, jeonggan, beat]
        features += offsets
    if 'dynamic' in self.tokenizer.key_types:
        features.append(self.dynamic_template.get(note.measure_offset, 'none'))
    if 'measure_idx' in self.tokenizer.key_types:
        features.append(note.measure_number)
    if 'measure_change' in self.tokenizer.key_types:
        changed = 1 if note.measure_offset == 0.0 else 0
        features.append(changed)
    return features

  def fix_part_by_rule(self):
    return

  def get_processed_feature(self, front_part_idx, back_part_idx, idx):
      source_start_token = [front_part_idx] + ['start'] * (len(self.feature_types) - 1)
      source_end_token = [front_part_idx]+ ['end'] * (len(self.feature_types) - 1) 
      if self.is_valid:
          measure_list = idx
      else:    
          measure_list = self.slice_info[idx]
          
      for i, idx in enumerate(measure_list):
          if len(self.era_dataset.measure_features[front_part_idx][idx]) == 0:
              return torch.LongTensor([[]]), torch.LongTensor([[]]), torch.LongTensor([[]])
          if len(self.measure_features[back_part_idx][idx]) == 0:
              return torch.LongTensor([[]]), torch.LongTensor([[]]), torch.LongTensor([[]])

      original_source_list = [item for idx in measure_list for item in self.era_dataset.measure_features[front_part_idx][idx]]
      original_target_list = [item for idx in measure_list for item in self.measure_features[back_part_idx][idx]]

      if 'measure_idx' in self.tokenizer.key_types:
        m_idx_pos = self.tokenizer.key2idx['measure_idx']
        source_first_measure_idx = original_source_list[0][m_idx_pos]
        target_first_measure_idx = original_target_list[0][m_idx_pos]

        original_source_list = [note[:m_idx_pos] + [note[m_idx_pos]-source_first_measure_idx] + note[m_idx_pos+1:] for note in original_source_list]
        original_target_list = [note[:m_idx_pos] + [note[m_idx_pos]-target_first_measure_idx] + note[m_idx_pos+1:] for note in original_target_list]
      condition_shifted_target = self.shift_condition(original_target_list)

      target_list = condition_shifted_target[: -1]

      if self.use_pitch_modification and not self.is_valid:
          target_list = self.modify_pitch(target_list)
      shifted_target_list = condition_shifted_target[1:]

      source_list = [source_start_token] + original_source_list + [source_end_token]
      
      source = [self.era_dataset.tokenizer(note_feature) for note_feature in source_list]
      target = [self.tokenizer(note_feature) for note_feature in target_list]
      shifted_target = [self.tokenizer(note_feature) for note_feature in shifted_target_list]
      
      return torch.LongTensor(source), torch.LongTensor(target), torch.LongTensor(shifted_target)        
        
  def __len__(self):
     if self.is_valid:
      return len(self.result_pairs)
     return len(self.era_dataset)

  def __getitem__(self, idx):
    if self.is_valid:
      front_part_idx, back_part_idx, measure_idx = self.result_pairs[idx]
      src, tgt, shifted_tgt = self.get_processed_feature(front_part_idx, back_part_idx, measure_idx)
      
      return src, tgt, shifted_tgt
    
    sample_success = False
    while not sample_success:
      # front_part_idx = random.choice(range(len(self.parts)-1))
      front_part_idx = 7
      back_part_idx = random.randint(0, len(self.parts)-1)
      src, tgt, shifted_tgt = self.get_processed_feature(front_part_idx, back_part_idx, idx)          
      if len(src) > 2 and len(tgt) > 1:
        sample_success = True
    return src, tgt, shifted_tgt    

class OrchestraScoreGMG(OrchestraScore):
  def __init__(self,xml_path='0_edited.musicxml', 
                valid_measure_num = [i for i in range(93, 104)],
                slice_measure_num = 2, 
                is_valid = False, 
                use_pitch_modification=False, 
                pitch_modification_ratio=0.3, 
                min_meas=3, 
                max_meas=6,
                feature_types=['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx'],
                sampling_rate=None,
                target_instrument=None) -> None:
    super().__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, feature_types, sampling_rate)
    valid_list = list(range(len(self.parts)))
    self.result_pairs = [[7, b] for b in valid_list]
    self.result_pairs = [pair+[i]  for pair in self.result_pairs for i in self.slice_info]

  
  def __getitem__(self, idx):
    front_part_idx = 7
    back_part_idx = len(self.parts) - 1
    if self.is_valid:
      front_part_idx, _, measure_idx = self.result_pairs[idx]
      src, tgt, shifted_tgt = self.get_processed_feature(front_part_idx, back_part_idx, measure_idx)
      
      return src, tgt, shifted_tgt
    
    src, tgt, shifted_tgt = self.get_processed_feature(front_part_idx, back_part_idx, idx)          
    return src, tgt, shifted_tgt    

class OrchestraScoreSeq(ShiftedAlignedScore):
  def __init__(self,xml_path='0_edited.musicxml', 
                valid_measure_num = [i for i in range(93, 104)],
                slice_measure_num = 2, 
                is_valid = False, 
                use_pitch_modification=False, 
                pitch_modification_ratio=0.3, 
                min_meas=3, 
                max_meas=6,
                feature_types=['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx'],
                sampling_rate=None,
                target_instrument=0,
                is_sep=True) -> None:
    self.dynamic_template = {x/2: 'weak' for x in range(0, 60, 3)}
    self.dynamic_template[0.0] =  'strong'
    self.dynamic_template[15.0] =  'strong'
    self.dynamic_template[9.0] =  'middle'
    self.dynamic_template[21.0] =  'middle'
    self.target_instrument = target_instrument
    self.is_sep = is_sep
    # if self.is_sep:
    #     offset_index = feature_types.index('offset')
    #     feature_types[offset_index:offset_index+1] = ['daegang_offset', 'jeonggan_offset', 'beat_offset']
    super().__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, feature_types, sampling_rate, start_part_idx=1)
    self.condition_instruments = list(range(target_instrument+1, len(self.parts)))
    if self.is_valid:
      self.slice_info = [list(range(i, i+self.slice_measure_number)) for i in self.valid_measure_num[:-self.slice_measure_number+1]]

  def get_feature(self, part_idx):
    part = self.parts[part_idx]
    measure_set = []
    for measure_idx, measure in enumerate(part.measures):
        each_measure = []
        for note in measure:
            each_measure.append(self._get_feature_by_note(note, part_idx))
        measure_set.append(each_measure) 
    return measure_set  
  
  def _get_feature_by_note(self, note, part_idx):
    features = [part_idx+1] # orchestra score starts from 1
    # if 'pitch' in self.tokenizer.key_types:
    features.append(note.pitch)
    # if 'duration' in self.tokenizer.key_types:
    features.append(note.duration)
    if 'offset' in self.tokenizer.key_types:
        features.append(note.measure_offset)
    if 'daegang_offset' in self.tokenizer.key_types:
        offsets = make_offset_set(note.measure_offset)  # [daegang, jeonggan, beat]
        features += offsets    
    if 'dynamic' in self.tokenizer.key_types:
        features.append(self.dynamic_template.get(note.measure_offset, 'none'))
    if 'measure_idx' in self.tokenizer.key_types:
        features.append(note.measure_number)
    if 'measure_change' in self.tokenizer.key_types:
        changed = 1 if note.measure_offset == 0.0 else 0
        features.append(changed)
    return features

  def fix_part_by_rule(self):
    return

  def get_processed_feature(self, front_part_idx, back_part_idx, idx):
      assert isinstance(front_part_idx, list), "front_part_idx should be a list"
    #   source_start_token = [front_part_idx[0], 'start', 'start', 'start', 'start', 'start'] # ['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx']
    #   source_end_token = [front_part_idx[0], 'end', 'end', 'end', 'end', 'end']
      source_start_token = [front_part_idx[0]] + ['start'] * (len(self.feature_types) - 1)
      source_end_token = [front_part_idx[0]]+ ['end'] * (len(self.feature_types) - 1) 
      measure_list = self.slice_info[idx]
      
      original_source_list = [item for f_idx in front_part_idx for idx in measure_list for item in self.measure_features[f_idx][idx]]
      original_target_list = [item for idx in measure_list for item in self.measure_features[back_part_idx][idx]]

      if 'measure_idx' in self.tokenizer.key_types:
        m_idx_pos = self.tokenizer.key2idx['measure_idx']
        source_first_measure_idx = original_source_list[0][m_idx_pos]
        target_first_measure_idx = original_target_list[0][m_idx_pos]

        original_source_list = [note[:m_idx_pos] + [note[m_idx_pos]-source_first_measure_idx] + note[m_idx_pos+1:] for note in original_source_list]
        original_target_list = [note[:m_idx_pos] + [note[m_idx_pos]-target_first_measure_idx] + note[m_idx_pos+1:] for note in original_target_list]
      condition_shifted_target = self.shift_condition(original_target_list)

      target_list = condition_shifted_target[: -1]

      if self.use_pitch_modification and not self.is_valid:
          target_list = self.modify_pitch(target_list)
      shifted_target_list = condition_shifted_target[1:]

      source_list = [source_start_token] + original_source_list + [source_end_token]
      
      source = [self.tokenizer(note_feature) for note_feature in source_list]
      target = [self.tokenizer(note_feature) for note_feature in target_list]
      shifted_target = [self.tokenizer(note_feature) for note_feature in shifted_target_list]
      
      return torch.LongTensor(source), torch.LongTensor(target), torch.LongTensor(shifted_target)        
        
  def __len__(self):
     return len(self.slice_info)

  def __getitem__(self, idx):
    front_part_idx = self.condition_instruments
    back_part_idx = self.target_instrument
    src, tgt, shifted_tgt = self.get_processed_feature(front_part_idx, back_part_idx, idx)          
    return src, tgt, shifted_tgt    

