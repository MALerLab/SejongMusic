import copy
import math
import json
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
        # self.part = part
        self.time_signature = None
        
        # self.flattened_notes = [note for note in self.part.flat.notes if note.duration.isGrace == False]
        self.flattened_notes = [note for note in part.flat.notes]
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

def make_offset_set(value, beat_type='orchestration'): # beat_type = ['orchestration', 'era', 'basic']
  daegang_offset = [0.0, 6.0, 10.0, 14.0]
  if beat_type == 'orchestration' :
    ratio = 1.5 # 0.0, 9.0, 15.0, 21.0
  else: 
    ratio = 0.5  
  daegang_offset = [i*ratio for i in daegang_offset]
  new_offset = []
  
  # daegang
  daegang_idx = 0
  for i, offset in enumerate(daegang_offset):
    if value >= offset:
      daegang_idx = i
  new_offset.append(daegang_idx)

  # jeonggan
  jeonggan_value = value - daegang_offset[daegang_idx]
  jeonggan_offset = [i*ratio for i in range(0, 6)]
  for i, offset in enumerate(jeonggan_offset):
    if jeonggan_value >= offset:
      jeonggan_idx = i
  new_offset.append(jeonggan_idx)

  # beat
  injeonggan_value = jeonggan_value - jeonggan_offset[jeonggan_idx]
  injeonggan_offset = [i/6*ratio for i in range(0, 6)]
  for i, offset in enumerate(injeonggan_offset):
    if injeonggan_value >= offset:
      injeonggan_idx = i
  new_offset.append(injeonggan_idx)
  return new_offset


def as_fraction(dct):
    if "numerator" in dct and "denominator" in dct:
        return Fraction(dct["numerator"], dct["denominator"])
    return dct

class FractionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Fraction):
            return {'numerator': obj.numerator, 'denominator': obj.denominator}
        # Let the base class default method raise the TypeError
        return super().default(obj)

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