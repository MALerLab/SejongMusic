import copy
import math
import random
from fractions import Fraction

from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Set, Dict, Tuple, Union

import music21
from music21 import converter, stream, note as m21_note 
import torch
import numpy as np
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pack_padded_sequence, pad_sequence 

from .constants import get_dynamic
from .metric import make_dynamic_template, convert_dynamics_to_integer, convert_note_to_sampling, convert_onset_to_sustain_token

class Gnote:
    def __init__(self, note: music21.note.Note, part_idx:int) -> None:
        self.note = [note]  

        self.pitch = note.pitch.ps
        self.tie = note.tie
        self.part_idx = part_idx
        if part_idx == 0:
            self.pitch -= (9+12) # C4 -> EB2
        self.measure_number = note.measureNumber
        self.offset = note.offset
        self.measure_offset = 0.0 # dummy value, float로 함!(offset: float)
        # self.duration = sum([note.duration.quarterLength for note in self.note])
        # self.dynamic = 0.0
        
    def __add__(self, note: music21.note.Note):
        # assert self.pitch == note.pitch.ps
        self.note.append(note)
        return self

    @property
    def duration(self):
        return sum([note.duration.quarterLength for note in self.note])

    @property
    def end_measure_number(self):
        return self.note[-1].measureNumber
    
    @property
    def dynamic(self):
        return get_dynamic(self.measure_offset, self.part_idx)

    
    def __repr__(self) -> str:
        return f'Gnote: {self.pitch}, {self.duration}, {self.measure_offset}, {self.dynamic}'

class SimpleNote:
    def __init__(self, note: Gnote) -> None:
        self.pitch = note.pitch
        self.duration = note.duration
        self.offset = note.offset
        self.measure_offset = note.measure_offset
        self.measure_number = note.measure_number
        self.end_measure_number = note.end_measure_number
        self.part_idx = note.part_idx

    def __repr__(self) -> str:
        return f'SimpleNote: P:{self.pitch}, D:{self.duration}, M:{self.measure_number}, MO: {self.measure_offset}, Dyn: {self.dynamic}'

    @property
    def dynamic(self):
        return get_dynamic(self.measure_offset, self.part_idx)




def apply_tie_but_measure_split(notes: List[music21.note.Note], part_idx) -> List[music21.note.Note]:
    tied_notes = []
    previous_measure_number = notes[0].measureNumber
    # print(previous_measure_number)
    for note in notes:
        if note.tie is not None:
            if note.tie.type == 'start' or note.measureNumber != previous_measure:
                tied_notes.append(Gnote(note, part_idx))
            elif note.tie.type == 'continue':
                tied_notes[-1] +=  note
            elif note.tie.type == 'stop':
                tied_notes[-1] += note
            else:
                raise ValueError(f'Unknown tie type: {note.tie.type}')
        else:
            tied_notes.append(Gnote(note, part_idx))
        previous_measure = note.measureNumber
    return tied_notes

def apply_tie(notes: List[music21.note.Note], part_idx) -> List[music21.note.Note]:
    tied_notes = []
    for note in notes:
        if note.tie is not None:
            if note.tie.type == 'start':
                tied_notes.append(Gnote(note, part_idx))
            elif note.tie.type == 'continue':
                tied_notes[-1] +=  note
            elif note.tie.type == 'stop':
                tied_notes[-1] += note
            else:
                raise ValueError(f'Unknown tie type: {note.tie.type}')
        else:
            tied_notes.append(Gnote(note, part_idx))
        
    return tied_notes

class Part:
    def __init__(self, part: music21.stream.base.Part, part_idx) -> None:
        self.part = part
        self.time_signature = None
        
        # self.flattened_notes = [note for note in self.part.flat.notes if note.duration.isGrace == False]
        self.flattened_notes = [note for note in self.part.flat.notes]
        # self.used_notes = 
        self.num_measures = self.flattened_notes[-1].measureNumber - self.flattened_notes[0].measureNumber + 1
        self.measures = [ [] for _ in range(self.num_measures) ]
        
        # self.tie_cleaned_notes = apply_tie(self.flattened_notes, part_idx) # 붙임줄 처리!
        self.tie_cleaned_notes = apply_tie_but_measure_split(self.flattened_notes, part_idx)
        for note in self.tie_cleaned_notes:
            self.measures[note.measure_number-1].append(note)
        self.measure_duration = self.get_measure_duration()
        self.measure_offsets = [i * self.measure_duration for i in range(self.num_measures)]
        self.repair_measure_offset_and_dynamic()
        self.index = part_idx
    
    def repair_measure_offset_and_dynamic(self):
        # repair offset of self.measures using self.measure_offsets
        for i, measure in enumerate(self.measures):
            measure_offset = self.measure_offsets[i]
            for gnote in measure:
                gnote_measure = gnote.note[0].getContextByClass(stream.Measure, sortByCreationTime=True)
                recorded_music21_measure_offset = gnote_measure.offset
                if measure_offset != recorded_music21_measure_offset:
                    gnote.offset += measure_offset - recorded_music21_measure_offset
                gnote.measure_offset = round_number(gnote.offset - measure_offset)
                # gnote.dynamic = gnote.dynamic
                
    def __len__(self):
        return len(self.measures)

    def get_measure_duration(self):
        # print(self.measures[0])
        durations =  [sum([note.duration for note in measure]) for measure in self.measures]
        # print(durations)
        most_common_duration = Counter(durations).most_common(1)[0][0]
        return most_common_duration
    
    def convert_to_common_beat(self):
        common_measures = []
      
        if self.measure_duration == 4.0: # 세종 # 5+3(4.0) -> 6+4(12)
            for i, measure in enumerate(self.measures):
                for note in measure:
                    note_measure_offset = note.offset - self.measure_offsets[i]
                    if note_measure_offset < 2.5:   # 5
                        converted_offset = note_measure_offset * 12 / 5
                    else:
                        converted_offset = (note_measure_offset - 2.5) * 8 / 3 + 6    # 3
                    converted_note = copy.copy(note)
                    converted_note.offset = converted_offset + self.measure_offsets[i] * 10 / 4
                    common_measures.append(converted_note)
                    
                    
        elif self.measure_duration == 8.0: #금합자보
            for i, measure in enumerate(self.measures): # 10+6(8) -> 6+4(12)
                for note in measure:
                    note_measure_offset = note.offset - self.measure_offsets[i]
                    if note_measure_offset < 5:
                        converted_offset = note_measure_offset * 6 / 5
                    else:
                        converted_offset = (note_measure_offset - 5) * 4 / 3 + 6
                    converted_note = copy.copy(note)
                    converted_note.offset = converted_offset + self.measure_offsets[i] * 5 / 4
                    common_measures.append(converted_note)
        
        else:
            return [note for measure in self.measures for note in measure]
        return common_measures
    
    def make_pitch_contour(self, sampling_rate=12):
        common_measure = self.convert_to_common_beat()
        num_frames = len(self) * 10 * sampling_rate
        # num_frames = 161 * 10 * sampling_rate
        output = np.zeros(num_frames)
        # print(output.shape)
        for i in range(len(common_measure) - 1):
            note = common_measure[i]
            next_note = common_measure[i+1]
            
            current_offset = note.offset
            next_offset = next_note.offset
            
            current_frame_idx = round(current_offset * sampling_rate)
            end_frame_idx = round(next_offset * sampling_rate)
            
            output[current_frame_idx:end_frame_idx] = note.pitch

        output[end_frame_idx:] = common_measure[-1].pitch
        
        return output

class Tokenizer:
    def __init__(self, parts, feature_types=['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx']):
        num_parts = len(parts)
        self.key_types = feature_types
        vocab_list = defaultdict(list)
        vocab_list['index'] = [part.index for part in parts]
        vocab_list['pitch'] = ['pad', 'start', 'end'] + sorted(list(set([note.pitch for i in range(num_parts) for note in parts[i].tie_cleaned_notes])))
        vocab_list['duration'] = ['pad', 'start', 'end'] + [8.0] + sorted(list(set([note.duration for i in range(num_parts) for note in parts[i].tie_cleaned_notes])))
        if 'offset' in feature_types:
            vocab_list['offset'] = ['pad', 'start', 'end'] + sorted(list(set([note.measure_offset for i in range(num_parts) for note in parts[i].tie_cleaned_notes])))
        if 'dynamic' in feature_types:
            vocab_list['dynamic'] = ['pad', 'start', 'end'] + ['strong', 'middle', 'weak', 'none']
        if 'measure_idx' in feature_types:
            vocab_list['measure_idx'] = ['pad', 'start', 'end'] + [i for i in range(10)]
        if 'measure_change' in feature_types:
            vocab_list['measure_change'] = ['pad', 'start', 'end']  + [0, 1, 2]
        # vocab_list['offset'] = ['pad', 'start', 'end'] + sorted(list(set([note.measure_offset for i in range(num_parts) for note in parts[i].tie_cleaned_notes])))
        # vocab_list['dynamic'] = ['pad', 'start', 'end'] + ['strong', 'middle', 'weak']
        # vocab_list['measure_idx'] = ['pad', 'start', 'end'] + [i for i in range(10)]
        # vocab_list['measure_change'] = ['pad', 'start', 'end']  + [0, 1, 2]
        # vocab_list['measure_idx'] = ['pad', 'start', 'end'] + [i for i in range(6)]
        self.measure_duration = [parts[i].measure_duration for i in range(num_parts)]
        self.vocab = vocab_list
        self.vocab_size_dict = {key: len(value) for key, value in self.vocab.items()}
        self.tok2idx = {key: {k:i for i, k in enumerate(value)} for key, value in self.vocab.items() }
        if 'offset' in feature_types:
          self.tok2idx['offset_fraction'] = {Fraction(k).limit_denominator(3):v  for k, v in self.tok2idx['offset'].items() if type(k)==float}
        self.key2idx = {key: i for i, key in enumerate(self.key_types)}
        # self.key_types = ['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_change']

        self.note2token = {}
        
        # print(self.vocab_size_dict)

    def hash_note_feature(self, note_feature:Tuple[Union[int, float, str]]):
        if note_feature in self.note2token:
            return self.note2token[note_feature]
        else:
            out = [self.tok2idx[self.key_types[i]][element] for i, element in enumerate(note_feature)]
            self.note2token[note_feature] =  out
            return out
        
    def __call__(self, note_feature:List[Union[int, float, str]]):
        # converted_lists = [self.tok2idx[self.key_types[i]][element] for i, element in enumerate(note_feature)]
        converted_lists = self.hash_note_feature(tuple(note_feature))
        # key_types = ['index', 'pitch', 'duration', 'offset', 'dynamic']
        # key_types = ['index', 'pitch', 'duration', 'offset', 'dynamic']
        # converted_lists = [self.tok2idx[self.key_types[i]][element] for i, element in enumerate(note_feature)]
        
        return converted_lists
  
class AlignedScore:
  def __init__(self, xml_path='0_edited.musicxml', 
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
    self.parts = [Part(part, i+start_part_idx) for i, part in enumerate(self.score.parts)]
    self.offset_list = [part_list.measure_duration for part_list in self.parts]
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
      new_note = [note[0], pitch, note[2], note[3], note[4], note[5]]
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

    
def pack_collate(raw_batch):
  source, target, shifted_target = zip(*raw_batch)
  return [pack_sequence(source, enforce_sorted=False), pack_sequence(target, enforce_sorted=False), pack_sequence(shifted_target, enforce_sorted=False)]

def round_number(number, decimals=3):
    scale = 10.0 ** decimals
    return math.floor(number * scale) / scale

def pad_collate(raw_batch, max_len = 240):
  # source, target= zip(*raw_batch)[0], zip(*raw_batch)[1]
    source, target = zip(*raw_batch)
    padded_src = pad_sequence(source, batch_first=True, padding_value=0)
    padded_target = pad_sequence(target, batch_first=True, padding_value=0)
    if padded_src.shape[1] < max_len:
        padded_src = torch.cat([padded_src, torch.zeros(padded_src.shape[0], max_len - padded_src.shape[1], padded_src.shape[2], dtype=torch.long)], dim=1)
    if padded_target.shape[1] < max_len:
        padded_target = torch.cat([padded_target, torch.zeros(padded_target.shape[0], max_len - padded_target.shape[1], padded_target.shape[2], dtype=torch.long)], dim=1)
    return [padded_src, padded_target]



class OrchestraScore(ShiftedAlignedScore):
  def __init__(self,xml_path='0_edited.musicxml', 
                valid_measure_num = [i for i in range(93, 104)],
                slice_measure_num = 2, 
                is_valid = False, 
                use_pitch_modification=False, 
                pitch_modification_ratio=0.3, 
                min_meas=3, 
                max_meas=6,
                feature_types=['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx'],
                sampling_rate=None) -> None:
    self.dynamic_template = {x/2: 'weak' for x in range(0, 90, 3)}
    self.dynamic_template[0.0] =  'strong'
    self.dynamic_template[15.0] =  'strong'
    self.dynamic_template[9.0] =  'middle'
    self.dynamic_template[21.0] =  'middle'
    super().__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, feature_types, sampling_rate, start_part_idx=1)
    self.era_dataset = ShiftedAlignedScore(xml_path=Path(xml_path).parent/'0_edited.musicxml', is_valid=is_valid, min_meas=min_meas, max_meas=max_meas, feature_types=feature_types, slice_measure_num=slice_measure_num)
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
      source_start_token = [front_part_idx, 'start', 'start', 'start', 'start', 'start']
      source_end_token = [front_part_idx, 'end', 'end', 'end', 'end', 'end']
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

    
# ------------ Sampling-based --------------- #
class SamplingTokenizer(Tokenizer):
  def __init__(self, parts, feature_types=['index', 'pitch', 'dynamic',  'offset']):
    num_parts = len(parts)
    self.key_types = feature_types
    vocab_list = defaultdict(list)
    vocab_list['index'] = [i for i in range(num_parts)]
    vocab_list['pitch'] = ['pad', 'start', 'end', 0] + sorted(list(set([note.pitch for i in range(num_parts) for note in parts[i].tie_cleaned_notes])))
    if 'dynamic' in feature_types:
        vocab_list['dynamic'] = ['pad', 'start', 'end'] + list(range(7))
    if 'offset' in feature_types:
        vocab_list['offset'] = ['pad', 'start', 'end'] + list(range(240))

    self.measure_duration = [parts[i].measure_duration for i in range(num_parts)]
    self.vocab = vocab_list
    self.vocab_size_dict = {key: len(value) for key, value in self.vocab.items()}
    self.tok2idx = {key: {k:i for i, k in enumerate(value)} for key, value in self.vocab.items() }
    self.key2idx = {key: i for i, key in enumerate(self.key_types)}
    # self.key_types = ['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_change']

    self.note2token = {}

# class SamplingTokenizer:
#     def __init__(self, parts):
#         num_parts = len(parts)
#         vocab_list = defaultdict(list)
#         vocab_list['index'] = [i for i in range(num_parts)]
#         vocab_list['pitch'] = ['sustain'] + sorted(list(set([note.pitch for i in range(num_parts) for note in parts[i].tie_cleaned_notes])))
#         # vocab_list['is_onset'] = ['pad', 'start', 'end'] + [0, 1]
#         vocab_list['is_onset'] = [0, 1]
#         vocab_list['dynamic'] = ['pad'] + ['strong', 'middle', 'weak']
        
#         self.measure_duration = [parts[i].measure_duration for i in range(num_parts)]
#         self.vocab = vocab_list
#         self.vocab_size_dict = {key: len(value) for key, value in self.vocab.items()}
#         self.tok2idx = {key: {k:i for i, k in enumerate(value)} for key, value in self.vocab.items() }

#     def __call__(self, note_feature:list):
#         if len(note_feature) == 4:
#             key_types = ['index', 'pitch', 'is_onset', 'dynamic']
#         if len(note_feature) == 3:
#             key_types = ['index', 'pitch', 'dynamic']
#         converted_lists = [self.tok2idx[key_types[i]][element] for i, element in enumerate(note_feature)]
        
#         return converted_lists
# self, tokenizer=Tokenizer, xml_path='/home/danbi/userdata/DANBI/gugakwon/Yeominrak/0_edited.musicxml', 
# slice_measure_num = 4, sample_len = 1000, is_valid = True, use_pitch_modification=False

class ModifiedTokenizer:
    def __init__(self, parts):
        num_parts = len(parts)
        vocab_list = defaultdict(list)
        vocab_list['index'] = ['pad']+[i for i in range(num_parts)]
        vocab_list['pitch'] = ['pad'] + sorted(list(set([note.pitch for i in range(num_parts) for note in parts[i].tie_cleaned_notes])))
        # vocab_list['is_onset'] = ['pad', 'start', 'end'] + [0, 1]
        vocab_list['is_onset'] = [0, 1]
        vocab_list['dynamic'] = ['pad'] + ['strong', 'middle', 'weak']
        
        self.measure_duration = [parts[i].measure_duration for i in range(num_parts)]
        self.vocab = vocab_list
        self.vocab_size_dict = {key: len(value) for key, value in self.vocab.items()}
        self.tok2idx = {key: {k:i for i, k in enumerate(value)} for key, value in self.vocab.items() }

    def __call__(self, note_feature:list):
        key_types = ['index', 'pitch', 'is_onset', 'dynamic']
        converted_lists = [self.tok2idx[key_types[i]][element] for i, element in enumerate(note_feature)]
        
        return converted_lists


# ---------- Sampling-based ----------- #
class SamplingScore(AlignedScore):
  def __init__(self, xml_path='0_edited.musicxml', 
               valid_measure_num=[i for i in range(93, 104)], 
               slice_measure_num=2, is_valid=False, 
               use_pitch_modification=False, 
               pitch_modification_ratio=0, 
               min_meas=3, 
               max_meas=6, 
               feature_types=[],
               sampling_rate=2) -> None:
    super().__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, feature_types)
    self.sampling_rate = sampling_rate
    self.dynamic_template_list = make_dynamic_template(beat_sampling_num=self.sampling_rate)
    self.dynamic_template_list = [convert_dynamics_to_integer(d) for d in self.dynamic_template_list]
    self.tokenizer = SamplingTokenizer(self.parts)

  def get_processed_feature(self, front_part_idx, back_part_idx, idx):
    if self.is_valid:
        measure_list = idx
    else:    
        measure_list = self.slice_info[idx]

    for i, idx in enumerate(measure_list):
        if len(self.measure_features[front_part_idx][idx]) == 0:
            return torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([])
        if len(self.measure_features[back_part_idx][idx]) == 0:
            return torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([])


    original_source_list = [item for idx in measure_list for item in self.measure_features[front_part_idx][idx]]
    original_target_list = [item for idx in measure_list for item in self.measure_features[back_part_idx][idx]]


    source_roll = convert_note_to_sampling(original_source_list, self.dynamic_template_list, beat_sampling_num=self.sampling_rate)
    target_roll = convert_note_to_sampling(original_target_list, self.dynamic_template_list, beat_sampling_num=self.sampling_rate)

    source_roll = convert_onset_to_sustain_token(source_roll)
    target_roll = convert_onset_to_sustain_token(target_roll)

    enc_in = [self.tokenizer(item) for item in source_roll]
    dec_in = [self.tokenizer(item) for item in target_roll]

    enc_in = torch.tensor(enc_in, dtype=torch.long)
    dec_in = torch.tensor(dec_in, dtype=torch.long)

    target = dec_in[:, 0:2]
    dec_in = dec_in[:, [0, 2, 3]]

    return enc_in, dec_in, target

  def __getitem__(self, idx):
    if self.is_valid:
      front_part_idx, back_part_idx, measure_idx = self.result_pairs[idx]
      src, tgt, shifted_tgt = self.get_processed_feature(front_part_idx, back_part_idx, measure_idx)
      return src, tgt, shifted_tgt
    else:    
      sample_success = False
      while not sample_success:
        front_part_idx = random.choice(range(len(self.parts)-1))
        # back_part_idx should be bigger than front_part_idx
        back_part_idx = random.randint(front_part_idx + 1, min(len(self.parts) - 1, front_part_idx + 2))
        src, tgt, shifted_tgt = self.get_processed_feature(front_part_idx, back_part_idx, idx)          
        if len(src) > 0 and len(tgt) > 0:
          sample_success = True
    return src, tgt, shifted_tgt




class SamplingTestScore(TestScore, SamplingScore):
  def __init__(self, xml_path='0_edited.musicxml', valid_measure_num=[i for i in range(93, 104)], slice_measure_num=2, 
               is_valid=False, use_pitch_modification=False, pitch_modification_ratio=0, min_meas=3, max_meas=6, transpose=0, 
               feature_types=['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx'],
               sampling_rate=2) -> None:
    TestScore.__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, transpose, feature_types)
    self.sampling_rate = sampling_rate
    self.dynamic_template_list = make_dynamic_template(beat_sampling_num=self.sampling_rate)
    self.dynamic_template_list = [convert_dynamics_to_integer(d) for d in self.dynamic_template_list]
    self.tokenizer = SamplingTokenizer(self.parts)

  def get_processed_feature(self, front_part_idx, back_part_idx, idx):
    return SamplingScore.get_processed_feature(self, front_part_idx, back_part_idx, idx)

  def __getitem__(self, idx):
    return SamplingScore.__getitem__(self, idx)
       

class SamplingScoreOld(AlignedScore):
    def __init__(self, xml_path='/home/danbi/userdata/DANBI/gugakwon/Yeominrak/0_edited.musicxml', \
        slice_measure_num = 4, sample_len = 1000, is_valid=False, use_pitch_modification=False, beat_sampling_num=6) -> None:
        super().__init__(xml_path, slice_measure_num, sample_len, is_valid, use_pitch_modification)
        self.beat_sampling_num = beat_sampling_num
        self.tokenizer = SamplingTokenizer(self.parts)
        self.dynamic_templates = self.make_dynamic_template()
        self.tokenized_dynamics = []
        for tem in self.dynamic_templates:
            temp = [self.tokenizer.tok2idx['dynamic'][i] for i in tem]
            self.tokenized_dynamics.append(temp)
        
        self.converted_features = []
        for features in self.sliced_features:
            converted_features = []
            for feature in features:
                converted_feature = self.convert_to_sampling(feature)
                converted_features.append(converted_feature)
            self.converted_features.append(converted_features)

    def __len__(self):
        if self.is_valid:
            return len(self.valid_index)
        return self.sample_len
    
    def convert_to_sampling(self, data):
        new_data = []
        for measure in data:
            new_measure = []
            for note in measure:
                frame = int(self.beat_sampling_num * note[2])
                if note[0]==0:
                    frame = frame * 2 # beat strength는 절대값?
                # new_note = [[note[0], note[1], note[4], 1]] + [[note[0], note[1], note[4], 0]]*(frame-1) # [part_idx, pitch, beat strengh, is_onset]
                new_note = [[note[0], note[1], 1]] + [[note[0], note[1], 0]]*(frame-1)
                new_measure += new_note
                
            dynamic_template = self.dynamic_templates[new_measure[0][0]] #part_idx
            dynamic_template = dynamic_template * (len(new_measure)//len(dynamic_template))
            new_measure_with_dynamic = []
            # print(len(new_measure), len(dynamic_template))
            for i, frame in enumerate(new_measure):
                if len(new_measure) == len(dynamic_template):
                    new_frame = frame+[dynamic_template[i]]
                    new_measure_with_dynamic.append(new_frame)
                else:
                    ValueError('len(new_measure) != len(dynamic_template)')
            new_data.append(new_measure_with_dynamic)

        return new_data
    
    def modify_pitch(self, target_list):
        ratio = 0.3
        modified_list = []
        for note in target_list:
            if random.random() < ratio:
                pitch = random.choice(self.vocab['pitch'][3:])
            else:
                pitch = note[1]
            new_note = [note[0], pitch, note[2], note[3]]
            modified_list.append(new_note)
 
        return modified_list    

    def make_dynamic_template(self):
        whole_dynamic_template = []
        for part_idx in range(len(self.parts)):
            frame = self.beat_sampling_num
            if part_idx == 0:
                # 세종실록악보 : 2.5/1.5, 1.5/1/1.5
                dynamic_mapping = {0.0: 'strong', 2.5: 'strong', 1.5: 'middle', \
                    0.5: 'weak', 1.0: 'weak', 2.0: 'weak', 3.0: 'weak',3.5: 'weak'}
                frame = frame * 2 #세종실록인 경우는 frame이 두 배!
            
            elif part_idx == 1:
                # 금합자보: 5/3, 3/2/3
                dynamic_mapping = {0.0: 'strong', 5.0: 'strong', 3.0: 'middle',\
                    1.0: 'weak', 2.0: 'weak', 4.0: 'weak', 6.0: 'weak', 7.0: 'weak',}
            
            elif part_idx in [2, 3, 4, 5]:
                # 속악원보, 금보신증가령, 한금신보, 어은보: 6/4, 3/3/2/2 
                dynamic_mapping = {0.0: 'strong', 6.0: 'strong', 3.0: 'middle', 8.0: 'middle', \
                    1.0: 'weak', 2.0: 'weak', 4.0: 'weak', 5.0: 'weak', 7.0: 'weak', 9.0: 'weak'}
            
            elif part_idx in [6, 7]:
                # 삼죽금보, 현행: 5/5, 3/2/2/3
                dynamic_mapping = {0.0: 'strong', 5.0: 'strong', 3.0: 'middle', 7.0: 'middle',\
                    1.0: 'weak', 2.0: 'weak', 4.0: 'weak', 6.0: 'weak', 8.0: 'weak', 9.0: 'weak'}
            whole_measure_len = int(frame * self.offset_list[part_idx])
            dynamic_list = []
            for i in range(whole_measure_len):
                dynamic = dynamic_mapping.get(float(i) / frame, 'pad')
                dynamic_list.append(dynamic)
            whole_dynamic_template.append(dynamic_list)
        return whole_dynamic_template
        
        
    def _get_source_and_target(self, front_part_idx, back_part_idx, measure_number):

        original_source_list = []
        for item in self.converted_features[measure_number][front_part_idx]:
            if isinstance(item, list):
                original_source_list.extend(item)
            else:
                original_source_list.append(item)
        original_target_list = []
        for item in self.converted_features[measure_number][back_part_idx]:
            if isinstance(item, list):
                original_target_list.extend(item)
            else:
                original_target_list.append(item)

        if self.use_pitch_modification and not self.is_valid:
            target_original_target_list = self.modify_pitch(original_target_list)
    
        source = [self.tokenizer(note_feature) for note_feature in original_source_list]
        target = [self.tokenizer(note_feature) for note_feature in original_target_list]

        return source, target
    
    def get_processed_feature(self, front_part_idx, back_part_idx, measure_number):
        source, target = self._get_source_and_target(front_part_idx, back_part_idx, measure_number)
        decoder_input = [[note[0], note[3]] for note in target]
        target = [[note[1], note[2]] for note in target]
        
        return torch.LongTensor(source), torch.LongTensor(decoder_input), torch.LongTensor(target)
    
    def __getitem__(self, idx):
        if self.is_valid:
            front_part_idx, back_part_idx, measure_number = self.valid_index[idx]
        else:    
            front_part_idx = random.choice(range(len(self.parts)-1))
            back_part_idx = random.choice(range(len(self.parts)-1))
            measure_number = random.randint(0, len(self.sliced_features[0])-1 - 2) #하드코딩이긴 함...
        src, tgt, dec_input = self.get_processed_feature(front_part_idx, back_part_idx, measure_number)
        return src, tgt, dec_input
    

class SamplingScore2(SamplingScore):
    def __init__(self, xml_path='/home/danbi/userdata/DANBI/gugakwon/Yeominrak/0_edited.musicxml', \
        slice_measure_num = 2, sample_len = 1000, is_valid=False, use_pitch_modification=False, beat_sampling_num=6) -> None:
        super().__init__(xml_path, slice_measure_num, sample_len, is_valid, use_pitch_modification)

    def convert_to_sampling(self, data):
        new_data = []
        for measure in data:
            new_measure = []
            for note in measure:
                frame = int(self.beat_sampling_num * note[2])
                if note[0]==0:
                    frame = frame * 2 # beat strength는 절대값?
                # new_note = [[note[0], note[1], note[4], 1]] + [[note[0], note[1], note[4], 0]]*(frame-1) # [part_idx, pitch, beat strengh, is_onset]
                new_note = [[note[0], note[1], 1]] + [[note[0], 'sustain', 0]]*(frame-1)
                new_measure += new_note
                
            dynamic_template = self.dynamic_templates[new_measure[0][0]] #part_idx
            dynamic_template = dynamic_template * (len(new_measure)//len(dynamic_template))
            new_measure_with_dynamic = []
            # print(len(new_measure), len(dynamic_template))
            for i, frame in enumerate(new_measure):
                if len(new_measure) == len(dynamic_template):
                    new_frame = frame+[dynamic_template[i]]
                    new_measure_with_dynamic.append(new_frame)
                else:
                    ValueError('len(new_measure) != len(dynamic_template)')
            new_data.append(new_measure_with_dynamic)

        return new_data
    


    def get_processed_feature(self, front_part_idx, back_part_idx, measure_number):
        source, target = self._get_source_and_target(front_part_idx, back_part_idx, measure_number)
        # src = [part_idx, pitch, dynamics]
        decoder_input = [[note[0], note[3]] for note in target] # [part_idx, dynamics]
        target = [note[1] for note in target] # pitch만!
        
        return torch.LongTensor(source), torch.LongTensor(decoder_input), torch.LongTensor(target)



# ------------------ CNN-based ------------------ #

class CNNScore(SamplingScore):
    def __init__(self, xml_path='0_edited.musicxml', \
        slice_measure_num = 2, sample_len = 1000, is_valid=False, use_pitch_modification=False, beat_sampling_num=6) -> None:
        super().__init__(xml_path, slice_measure_num, sample_len, is_valid, use_pitch_modification)
        self.tokenizer = ModifiedTokenizer(self.parts)
    
    def get_processed_feature(self, front_part_idx, back_part_idx, measure_number):
        source, target = self._get_source_and_target(front_part_idx, back_part_idx, measure_number)
        # print(f"target={target}")
        #source/target's shape = [part_idx, pitch, is_onset, dynamic]
        fixed_src = [[note[0], note[1]] for note in source]
        fixed_tgt = [[note[0], note[1]] for note in target]
        
        return torch.LongTensor(fixed_src), torch.LongTensor(fixed_tgt)
    
    def __getitem__(self, idx):
        if self.is_valid:
            front_part_idx, back_part_idx, measure_number = self.valid_index[idx]
        else:    
            front_part_idx = random.choice(range(len(self.parts)-1))
            back_part_idx = random.choice(range(len(self.parts)-1))
            measure_number = random.randint(0, len(self.sliced_features[0])-1 - 2) #하드코딩이긴 함...
        src, tgt = self.get_processed_feature(front_part_idx, back_part_idx, measure_number)
        return src, tgt
    
class BasicWork:
    def __init__(self, xml_path='0_edited.musicxml') -> None:
        self.xml_path = xml_path
        self.score = converter.parse(xml_path)
        self.parts = [Part(part, i) for i, part in enumerate(self.score.parts)]
        self.tokenizer = Tokenizer(self.parts)
        self.vocab = self.tokenizer.vocab
        self.vocab_size_dict = self.tokenizer.vocab_size_dict
        self.offset_list = [part_list.measure_duration for part_list in self.parts]
        self.measure_offset_vocab = []
        self.measure_features = [self.get_feature(i) for i in range(len(self.parts))]
        max_len = max([len(part) for part in self.measure_features])
        self.measure_features = [part + [[] for _ in range(max_len - len(part))] for part in self.measure_features]
        self.part_templates = self._get_part_templates()
        
    def get_feature(self, part_idx):
      part = self.parts[part_idx]
      measure_set = []
      for measure_idx, measure in enumerate(part.measures):
          each_measure = []
          for note in measure:
              each_measure.append([part_idx, note.pitch, note.duration, note.measure_offset, note.dynamic])
          if part_idx in [2,3,4,5,6,7] and sum([note[2] for note in each_measure]) < 10.0:
              each_measure = []
          measure_set.append(each_measure) 
      return measure_set      

    def _get_part_templates(self):
        whole_part_templates = []
        for part in self.measure_features:
            part_templates = []
            for measure in part:
                temp = [note[2] for note in measure]
                part_templates.append(temp)
            # part_templates = set(tuple(temp) for temp in part_templates)
            whole_part_templates.append(part_templates)
        return whole_part_templates




# ===================================================================== #

# class AlignedScore:
#     def __init__(self, xml_path='0_edited.musicxml', slice_measure_num = 4, sample_len = 1000, is_valid = False, use_pitch_modification=False) -> None:
        
#         self.score = converter.parse(xml_path)
#         self.parts = [Part(part, i) for i, part in enumerate(self.score.parts)]
#         self.offset_list = [part_list.measure_duration for part_list in self.parts]
        
#         self.measure_offset_vocab = []
#         self.measure_features = [self.get_feature(i) for i in range(len(self.parts))]
        
#         self.tokenizer = Tokenizer(self.parts)
#         self.vocab = self.tokenizer.vocab
#         self.vocab_size_dict = self.tokenizer.vocab_size_dict
        
#         self.is_valid = is_valid
#         self.use_pitch_modification = use_pitch_modification
#         self.sample_len = sample_len
#         self.valid_measure_numbers = [i for i, meas in enumerate(self.parts[0].measures) if len(meas)>0]
#         self.slice_measure_number = slice_measure_num
        
#         self.sliced_features = self.slice_by_valid_length()
        
#         if self.is_valid:
#             valid_list = [0,1,2,3,5,6,7] # 4 제외
#             self.valid_part_idx_list = [[a, b] for a in valid_list for b in valid_list if a<b and b-a<3]
#             # self.valid_index = [idx_list+[int(i)] for idx_list in self.valid_part_idx_list for i in range(len(self.sliced_features))]
#             self.valid_index = [idx_list+[int(i)] for idx_list in self.valid_part_idx_list for i in range(len(self.sliced_features))]

#     def __len__(self):
#         if self.is_valid:
#             return len(self.valid_index)
#         return self.sample_len
    
#     def get_notes_from_measures(self, measure_idx):
#         return [part.measures[measure_idx] for part in self.parts]

#     def get_feature(self, part_idx):
#         # (index, pitch, duration, offset, dynamic), 
#         part = self.parts[part_idx]
#         measure_set = []
#         for measure_idx, measure in enumerate(part.measures):
#             each_measure = []
#             for note in measure:
#                 # note_measure_offset, dynamic = self.get_offset_and_dynamics(note, part_idx, measure_idx) 
#                 each_measure.append([part_idx, note.pitch, note.duration, note.measure_offset, note.dynamic])
#             # if len(each_measure) > 0:
#             measure_set.append(each_measure)
#         return measure_set
    
#     def slice_by_valid_length(self):
#         valid_cut_point = self.valid_measure_numbers[0::self.slice_measure_number]
#         if self.slice_measure_number == 2:
#             valid_idx_list = [32, 33, 34, 35] 
#         elif self.slice_measure_number == 4:
#             valid_idx_list = [16, 17]
#         # ----------------------------- #
#         if self.is_valid:
#             i_list = valid_idx_list 
#         else:
#             i_list = [i for i in range(len(valid_cut_point)-1) if i not in valid_idx_list]
            
#         sliced_features = [] 
#         for j in i_list:
#             sliced_part_list = [part[valid_cut_point[j]:valid_cut_point[j+1]] for part in self.measure_features]
#             uncomplete_part_idxs = []
#             for i, part in enumerate(sliced_part_list):
#                 if len(part) == 0 or  min([len(measure) for measure in part]) == 0:
#                     uncomplete_part_idxs.append(i)
#             for i in uncomplete_part_idxs:
#                 sliced_part_list[i] = []
#             sliced_features.append(sliced_part_list)
#         return sliced_features
    
#     def modify_pitch(self, target_list):
#         ratio = 0.3
#         modified_list = []
#         for note in target_list[1:-1]:
#             if random.random() < ratio:
#                 pitch = random.choice(self.vocab['pitch'][3:])
#             else:
#                 pitch = note[1]
#             # new_note = [note[0], pitch, note[2], note[3], note[4], [note[5]]]
#             new_note = [note[0], pitch, note[2], note[3], note[4]]
#             modified_list.append(new_note)
#         completed_list = [target_list[0]] + modified_list + [target_list[-1]]
#         return completed_list       
    
#     def get_processed_feature(self, front_part_idx, back_part_idx, measure_number):
#         # print(front_part_idx, back_part_idx, measure_number)
#         source_start_token = [front_part_idx, 'start', 'start', 'start', 'start']
#         source_end_token = [front_part_idx, 'end', 'end', 'end', 'end']
#         target_start_token = [back_part_idx, 'start', 'start', 'start', 'start']
#         target_end_token = [back_part_idx, 'end', 'end', 'end', 'end']

#         original_source_list = []
#         for item in self.sliced_features[measure_number][front_part_idx]:
#             if isinstance(item, list):
#                 original_source_list.extend(item)
#             else:
#                 original_source_list.append(item)
#         original_target_list = []
#         for item in self.sliced_features[measure_number][back_part_idx]:
#             if isinstance(item, list):
#                 original_target_list.extend(item)
#             else:
#                 original_target_list.append(item)

#         # 여기서 measure_diff 및 measure_idx를 집어넣자!
        
#         source_list = [source_start_token] + original_source_list + [source_end_token]
#         target_list = [target_start_token] + original_target_list
#         if self.use_pitch_modification and not self.is_valid:
#             target_list = self.modify_pitch(target_list)
#         shifted_target_list = original_target_list + [target_end_token]
#         # print(target_list, shifted_target_list)
#         source = [self.tokenizer(note_feature) for note_feature in source_list]
#         target = [self.tokenizer(note_feature) for note_feature in target_list]
#         shifted_target = [self.tokenizer(note_feature) for note_feature in shifted_target_list]
        
#         return torch.LongTensor(source), torch.LongTensor(target), torch.LongTensor(shifted_target)

    
#     def __getitem__(self, idx):
#         sample_success = False
#         while not sample_success:
#           if self.is_valid:
#               front_part_idx, back_part_idx, measure_number = self.valid_index[idx]
#           else:    
#               front_part_idx = random.choice(range(len(self.parts)-1))
#               # back_part_idx should be bigger than front_part_idx
#               back_part_idx = random.randint(front_part_idx + 1, min(len(self.parts) - 1, front_part_idx + 2))
#               measure_number = random.randint(0, len(self.sliced_features)-1) #하드코딩이긴 함...
#               # print(f"front_part_idx={front_part_idx}, back_part_idx={back_part_idx}, measure_number={measure_number}")
#           src, tgt, shifted_tgt = self.get_processed_feature(front_part_idx, back_part_idx, measure_number)
#           if len(src) > 2 and len(tgt) > 1:
#             sample_success = True
#         return src, tgt, shifted_tgt    

# ------------------------------------------------------------------------------------------------------


# class ShiftedAlignedScore(AlignedScore):
#     def __init__(self, xml_path='0_edited.musicxml', slice_measure_num = 4, sample_len = 1000, is_valid = True, use_pitch_modification=False) -> None:
#         super().__init__(xml_path, slice_measure_num, sample_len, is_valid, use_pitch_modification)

    
#     def shift_condition(self, note_list, shift_idxs=(3,4)):
#         # note_list = [ [part_idx, pitch, duration, offset, dynamic], ... ]
#         non_shift_idxs = [i for i in range(len(note_list[0])) if i not in shift_idxs]
#         pred_features = [ [x[i] for i in non_shift_idxs] for x in note_list]
#         conditions = [ [x[i] for i in shift_idxs] for x in note_list] 
        
#         pred_features = [[note_list[0][0], 'start', 'start']] + pred_features
#         conditions = conditions +  [[0, 'strong']]
        
#         combined = [ pred_features[i] + conditions[i] for i in range(len(pred_features))] + [[note_list[-1][0], 'end', 'end', 'end', 'end']]
        
#         return combined
    
#     def get_processed_feature(self, front_part_idx, back_part_idx, measure_number):
#         source_start_token = [front_part_idx, 'start', 'start', 'start', 'start']
#         source_end_token = [front_part_idx, 'end', 'end', 'end', 'end']
#         # target_start_token = [back_part_idx, 'start', 'start', 0, 'strong']
#         # target_end_token = [back_part_idx, 'end', 'end', 0, 'strong']

#         original_source_list = []
#         for item in self.sliced_features[measure_number][front_part_idx]:
#             if isinstance(item, list):
#                 original_source_list.append(item)
#         if len(original_source_list) == 0:
#             return torch.LongTensor([[]]), torch.LongTensor([[]]), torch.LongTensor([[]])
    
#         original_target_list = []
#         for item in self.sliced_features[measure_number][back_part_idx]:
#             if isinstance(item, list):
#                 original_target_list.append(item)
#         if len(original_target_list) == 0:
#             return torch.LongTensor([[]]), torch.LongTensor([[]]), torch.LongTensor([[]])

#         condition_shifted_target = self.shift_condition(original_target_list)
#         target_list = condition_shifted_target[: -1]
        
#         if self.use_pitch_modification and not self.is_valid:
#             target_list = self.modify_pitch(target_list)
#         shifted_target_list = condition_shifted_target[1:]
        
#         source_list = [source_start_token] + original_source_list + [source_end_token]
        
#         source = [self.tokenizer(note_feature) for note_feature in source_list]
#         target = [self.tokenizer(note_feature) for note_feature in target_list]
#         shifted_target = [self.tokenizer(note_feature) for note_feature in shifted_target_list]
        
#         return torch.LongTensor(source), torch.LongTensor(target), torch.LongTensor(shifted_target)


if __name__ == "__main__": 
    # from omegaconf import OmegaConf
    # config = OmegaConf.load('yamls/baseline.yaml')
    train_dataset = ShiftedAlignedScore(is_valid = False)
    val_dataset = ShiftedAlignedScore(is_valid = True, slice_measure_num=10)

    for i in range(len(val_dataset)):
        out = val_dataset[i]
        if len(out[0]) < 4 or len(out[1]) < 4 or len(out[2]) < 4:
          print(f"{i}: {len(out[0])}, {len(out[1])}, {len(out[2])}")

    for i in range(len(train_dataset)):
        out = train_dataset[i]
        if len(out[0]) < 4 or len(out[1]) < 4 or len(out[2]) < 4:
          print(f"{i}: {len(out[0])}, {len(out[1])}, {len(out[2])}")

