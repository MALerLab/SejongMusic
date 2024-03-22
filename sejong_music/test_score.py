import copy

import torch

from .yeominrak_processing import AlignedScore
from .utils import SimpleNote, Part, Gnote, make_offset_set, as_fraction, FractionEncoder, pack_collate, pad_collate, round_number

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

