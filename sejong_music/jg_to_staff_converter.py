import re
from fractions import Fraction
from collections import OrderedDict, Counter
from typing import Any, List, Union, Tuple

import music21
from music21 import note as mnote, stream as stream, meter as mmeter, key as mkey, pitch as mpitch

from .constants import POSITION, PITCH, DURATION
from .jeonggan_utils import JGConverter, GencodeConverter
from .abc_utils import ABCNote

class JGCodeToOMRDecoder:
  pos_tokens = POSITION
  ignore_tokens = ('start', 'end', 'pad', '')
  
  one_third_pos = (':1', ':2', ':3')
  two_third_pos = (':4', ':5', ':6')
  three_third_pos = (':7', ':8', ':9')
  
  one_half_pos = (':12', ':10', ':13')
  two_half_pos = (':14', ':11', ':15')
  
  
  third_pos = one_third_pos + two_third_pos + three_third_pos
  half_pos = one_half_pos + two_half_pos
  
  pos_order = [':1',':12', ':2',':10',':13', ':3', ':4', ':5', ':6', ':14', ':11',  ':15', ':7', ':8', ':9']

  
  @staticmethod
  def get_token_idx_by_pos(splitted_tokens:List[str], pos:str):
    '''
    example:
    splitted_tokens = ['태_너녜:2', '쉼표:8']
    pos = ':8'
    
    return 1
    '''
    for idx, token in enumerate(splitted_tokens):
      if pos in token:
        return idx
  
  @classmethod
  def sort_tokens_by_pos(cls, splitted_tokens:List[str]):
    '''
    example:
    splitted_tokens = ['태_너녜:2', '쉼표:8', '-:5']
    '''
    return sorted(splitted_tokens, key=lambda x: cls.pos_order.index(re.findall(r'(?::\d+)', x)[0]))
  
  @classmethod
  def convert_jeonggan_to_omr_label(cls, jg_token_str:str):
    '''
    jg_token_str: str (example: 태_너녜:2 쉼표:8)
    
    '''
    if jg_token_str == ' ':
      return '-:5'
    
    
    if ':0' in jg_token_str:
      return jg_token_str.replace(':0', ':5')
    
    pos_token_in_jg = re.findall(r'(?::\d+)', jg_token_str)
    splitted_tokens = jg_token_str.split(' ')
    splitted_tokens = [x for x in splitted_tokens if x]
    is_three_col = any(pos in pos_token_in_jg for pos in cls.third_pos)
    is_two_col = any(pos in pos_token_in_jg for pos in cls.half_pos)
    
    if is_three_col:
      if not any(pos in pos_token_in_jg for pos in cls.one_third_pos): splitted_tokens.append('-:2')
      if not any(pos in pos_token_in_jg for pos in cls.two_third_pos): splitted_tokens.append('-:5')
      if not any(pos in pos_token_in_jg for pos in cls.three_third_pos): splitted_tokens.append('-:8')
    elif is_two_col:
      if not any(pos in pos_token_in_jg for pos in cls.one_half_pos): splitted_tokens.append('-:10')
      if not any(pos in pos_token_in_jg for pos in cls.two_half_pos): splitted_tokens.append('-:11')
        
    if ':1' in pos_token_in_jg and ':3' not in pos_token_in_jg: splitted_tokens.append('-:3')
    if ':3' in pos_token_in_jg and ':1' not in pos_token_in_jg: splitted_tokens.append('-:1')
    if ':4' in pos_token_in_jg and ':6' not in pos_token_in_jg: splitted_tokens.append('-:6')
    if ':6' in pos_token_in_jg and ':4' not in pos_token_in_jg: splitted_tokens.append('-:4')
    if ':7' in pos_token_in_jg and ':9' not in pos_token_in_jg: splitted_tokens.append('-:9')
    if ':9' in pos_token_in_jg and ':7' not in pos_token_in_jg: splitted_tokens.append('-:7')
    if ':10' in pos_token_in_jg and ':12' not in pos_token_in_jg: splitted_tokens.append('-:12')
    if ':12' in pos_token_in_jg and ':10' not in pos_token_in_jg: splitted_tokens.append('-:10')
    
    jg_token_str = ' '.join(cls.sort_tokens_by_pos(splitted_tokens))
    return jg_token_str
  
  def group_by_position(self, tokens):
    total_tokens = []
    prev_tokens = []
    
    for token in tokens:
      if token in self.pos_tokens and prev_tokens:
        prev_token_reversed = '_'.join(prev_tokens[1:]) + prev_tokens[0]
        total_tokens.append(prev_token_reversed)
        prev_tokens = [token]
        continue
      prev_tokens.append(token)
    total_tokens.append('_'.join(prev_tokens[1:]) + prev_tokens[0])

    return ' '.join(total_tokens)
  
  def fix_gak_str_list(self, gak_strs:List[str]):
    out = [self.fix_single_gak_str(gak) for gak in gak_strs]  
    return ' \n '.join(out)
  
  def fix_single_gak_str(self, gak_str:str):
    splitted = gak_str.split('|')
    out = [self.convert_jeonggan_to_omr_label(jg) for jg in splitted]
    return '|'.join(out)
  
  
  def __call__(self, tokens:List[str]):
    assert isinstance(tokens, list)
    if isinstance(tokens[0], list):
      tokens = [x[0] for x in tokens]
    out_tokens = [x for x in tokens if x not in self.ignore_tokens]
    reversed_token_str = self.group_by_position(out_tokens)
    str_in_gaks = reversed_token_str.split('\n')
    return self.fix_gak_str_list(str_in_gaks)
    # jg_splitted_tokens = re.split('\n|\|', reversed_token_str)
    # return '|'.join([self.convert_jeonggan_to_omr_label(x) for x in jg_splitted_tokens])
    # return jg_splitted_tokens
    
  def convert_multi_inst_str(self, multi_inst_str:str):
    assert isinstance(multi_inst_str, str)
    parts = multi_inst_str.split('\n\n')
    outputs = []
    for part in parts:
      outputs.append(self(part.split(' ')))
    return '\n\n'.join(outputs)



class Note:
  pos_to_beat_offset = {':0': 0,
                        ':1': 0,
                        ':2': 0,
                        ':3': Fraction(1, 6),
                        ':4': Fraction(1, 3),
                        ':5': Fraction(1, 3),
                        ':6': Fraction(1, 2),
                        ':7': Fraction(2, 3),
                        ':8': Fraction(2, 3),
                        ':9': Fraction(5, 6),
                        ':10': 0,
                        ':11': Fraction(1, 2),
                        ':12': 0,
                        ':13': Fraction(1, 4),
                        ':14': Fraction(1, 2),
                        ':15': Fraction(3, 4),
                        ':16': Fraction(1, 9),
                        ':17': Fraction(4, 9),
                        ':18': Fraction(7, 9),
                        ':19': Fraction(1, 6),
                        ':20': Fraction(2, 3),
                        ':21': Fraction(2, 9),
                        ':22': Fraction(5, 9),
                        ':23': Fraction(8, 9)}
  def __init__(self, pitch:List[str], pos:str, global_jg_offset:int, jg_offset:int, gak_offset:int, duration:float=None) -> None:
    assert type(pitch) == list
    self.pitch = pitch[0]
    self.pos = pos
    self.global_jg_offset = global_jg_offset
    self.jg_offset = jg_offset
    self.gak_offset = gak_offset
    self.ornaments = pitch[1:]
    self.duration = None
    self.midi_pitch = None
    self.m21_notes = []
  def __repr__(self) -> str:
    return f"Note {self.pitch}({self.midi_pitch})_{'_'.join(self.ornaments)} {self.pos} @ {self.global_jg_offset}+{self.beat_offset} / {self.duration}"
  
  @property
  def beat_offset(self):
    try:
      return self.pos_to_beat_offset[self.pos]
    except:
      print(f"Unknown pos {self.pos}, {self.pitch}, {self.ornaments}, {self.global_jg_offset}")
  
  @property
  def int_pos(self):
    return int(self.pos[1:])
  
  @property
  def offset(self):
    return self.global_jg_offset + self.beat_offset
  

class ThreeColumnDetector:
  pos2column = {':1': 0, ':2': 0, ':3':0,
                ':4': 1, ':5': 1, ':6': 1,
                ':7': 2, ':8': 2, ':9': 2,
                ':10': 3, ':12': 3, ':13': 3,
                ':11': 4, ':14': 4, ':15': 4,
                }
  def __init__(self) -> None:
    self.cur_jg_offset = 0
    self.pos_in_cur_jg = []
  
  
  
  def _check_note_in_three_col_middle(self, note:Note):
    if note.int_pos in (2, 5, 8) and (note.int_pos-1) in self.pos_in_cur_jg: return True
    elif note.int_pos == 10 and 12 in self.pos_in_cur_jg: return True
    elif note.int_pos == 11 and 14 in self.pos_in_cur_jg: return True
    return False
  
  def _check_note_in_three_col_end(self, note:Note):
    if note.int_pos in (3, 6, 9, 13, 15) and (self.pos2column[note.pos]+16) in self.pos_in_cur_jg: return True
    return False

  def __call__(self, note:Note):
    if self.cur_jg_offset != note.global_jg_offset:
      self.cur_jg_offset = note.global_jg_offset
      self.pos_in_cur_jg = []
    if self._check_note_in_three_col_middle(note):
      note.pos = f':{16+self.pos2column[note.pos]}'
    elif self._check_note_in_three_col_end(note):
      note.pos = f':{21+self.pos2column[note.pos]}'
    self.pos_in_cur_jg.append(note.int_pos)


def define_pitch2midi():
  pitch2midi = {}
  pitch_name = ["황" ,'대' ,'태' ,'협' ,'고' ,'중' ,'유' ,'임' ,'이' ,'남' ,'무' ,'응']
  octave_name = {"하하배": -3, "하배": -2, "배":-1, '':0, '청':+1, '중청':+2}

  for o in octave_name.keys():
    for i, p in enumerate(pitch_name):
      pitch2midi[o+p] = i + 63 + octave_name[o] * 12
  return pitch2midi

PITCH2MIDI = define_pitch2midi()


class SigimsaeConverter:
  def __init__(self, scale=['황', '태', '중', '임', '남'], exceptional_pitches=[]):
    self.scale = scale
    self.pitch2midi = OrderedDict({scale: PITCH2MIDI[scale] for scale in self.scale})
    octave_name = {"하하배": -3, "하배": -2, "배":-1, '':0, '청':+1, '중청':+2}
    for o in octave_name.keys():
      for p in self.scale:
        self.pitch2midi[o+p] = self.pitch2midi[p] + octave_name[o] * 12
    self.pitch2midi = OrderedDict(sorted(self.pitch2midi.items(), key=lambda x: x[1]))
    self.pitch2up =  {p: list(self.pitch2midi.keys())[i+1] for i, p in enumerate(list(self.pitch2midi.keys())[:-1])}
    self.pitch2down = {p: list(self.pitch2midi.keys())[i-1] for i, p in enumerate(list(self.pitch2midi.keys())[1:])}
    self.midi_scales = list(self.pitch2midi.values())

    
    for o in octave_name.keys():
      for p in exceptional_pitches:
        self.pitch2midi[o+p] = PITCH2MIDI[p] + octave_name[o] * 12
        
    self.exceptional_midi_scales = [self.pitch2midi[p] for p in exceptional_pitches]
    

  def up(self, pitch):
    if type(pitch) == int:
      if pitch in self.midi_scales:
        return self.midi_scales[self.midi_scales.index(pitch)+1]
      elif pitch in self.exceptional_midi_scales:
        for _ in range(10):
          pitch += 1
          if pitch in self.midi_scales:
            return pitch
      else:
        print(f'pitch {pitch} not in midi scales')
        return pitch
    return self.pitch2up[pitch]
  
  def down(self, pitch):
    if type(pitch) == str:
      return self.pitch2down[pitch]  
    if pitch in self.midi_scales:
      return self.midi_scales[self.midi_scales.index(pitch)-1]
    if pitch in self.exceptional_midi_scales:
      for _ in range(10):
        pitch -= 1
        if pitch in self.midi_scales:
          return pitch
    else:
      print(f'pitch {pitch} not in midi scales')
      return pitch

  def up2(self, pitch, n=2):
    if type(pitch) == str:
      for _ in range(n):
        pitch = self.up(pitch)
      return pitch

    if pitch in self.midi_scales:
      return self.midi_scales[self.midi_scales.index(pitch)+n]
    
    found = 0
    if pitch in self.exceptional_midi_scales:
      for _ in range(10):
        pitch += 1
        if pitch in self.midi_scales:
          found += 1
        if found == n:
          return pitch
  
  def down2(self, pitch, n=2):
    if type(pitch) == str:
      for _ in range(n):
        pitch = self.down(pitch)
      return pitch
    if pitch in self.midi_scales:
      return self.midi_scales[self.midi_scales.index(pitch)-n]
    found = 0
    if pitch in self.exceptional_midi_scales:
      for _ in range(10):
        pitch -= 1
        if pitch in self.midi_scales:
          found += 1
        if found == n:
          return pitch


class JGToStaffConverter:
  pos_tokens = POSITION[2:] # exclude '|', '\n'
  pitch_converter = SigimsaeConverter()
  pitch_names = ["황" ,'대' ,'태' ,'협' ,'고' ,'중' ,'유' ,'임' ,'이' ,'남' ,'무' ,'응']
  dur_tokens = DURATION

  
  def __init__(self, dur_ratio=1.5, is_abc=False) -> None:
    self.dur_ratio = dur_ratio
    self.is_abc = is_abc

  @staticmethod
  def _append_note(prev_note, prev_pos, global_jg_offset, jg_offset, gak_offset, total_notes):
    if prev_note:
      total_notes.append(Note(prev_note, prev_pos, global_jg_offset, jg_offset, gak_offset))
    return []
  
  @classmethod
  def convert_to_notes(cls, tokens:List[str]):
    global_jg_offset = 0
    jg_offset = 0
    gak_offset = 0
    prev_pos = None
    prev_note = []
    total_notes = []
    for token in tokens:
      if token == '' or token in ('pad', 'start', 'end'): continue
      if token in cls.pos_tokens:
        prev_note = cls._append_note(prev_note, prev_pos, global_jg_offset, jg_offset, gak_offset, total_notes)
        prev_pos = token
      elif token in ('|', '\n'):
        prev_note = cls._append_note(prev_note, prev_pos, global_jg_offset, jg_offset, gak_offset, total_notes)
        global_jg_offset += 1
        if token == '\n':
          jg_offset = 0
          gak_offset += 1
        else:
          jg_offset += 1
      else:
        prev_note.append(token)
    prev_note = cls._append_note(prev_note, prev_pos, global_jg_offset, jg_offset, gak_offset, total_notes)
    total_notes[-1].duration = max(global_jg_offset - total_notes[-1].global_jg_offset, 1) - total_notes[-1].beat_offset
    return total_notes

  @staticmethod
  def _fix_three_col_division(notes:List[Note]):
    three_col_detector = ThreeColumnDetector()
    for note in notes:
      three_col_detector(note)

  @staticmethod
  def get_duration_of_notes(notes:List[Note]):
    filtered_notes = [note for note in notes if (note.pitch != '-' or '노니로' in note.ornaments)]
    for i, note in enumerate(filtered_notes[:-1]):
      note.duration = filtered_notes[i+1].offset - note.offset
    if filtered_notes[-1].duration is None:
      filtered_notes[-1].duration = 1 - filtered_notes[-1].beat_offset
  
  def make_m21_note(self, pitch:int, duration:float, grace:bool=False):
    if isinstance(pitch, str):
      pitch = self.pitch_converter.pitch2midi[pitch]
    m21_note = mnote.Note(pitch=pitch, duration=music21.duration.Duration(duration * self.dur_ratio))
    if grace or duration == 0:
      m21_note = m21_note.getGrace()
    return m21_note
    
  def create_m21_notes(self, notes:List[Note], verbose=False):
    prev_pitch:int = 0
    for note in notes:
      if note.pitch in self.pitch_converter.pitch2midi:
        note.midi_pitch = self.pitch_converter.pitch2midi[note.pitch]
        note.m21_notes.append(self.make_m21_note(note.midi_pitch, note.duration))
      elif note.pitch == '같은음표':
        note.midi_pitch = prev_pitch
        note.m21_notes.append(self.make_m21_note(prev_pitch, note.duration))
      elif note.pitch == '노':
        note.midi_pitch = self.pitch_converter.down(prev_pitch)
        note.m21_notes.append(self.make_m21_note(note.midi_pitch, note.duration))
      elif note.pitch == '니':
        note.midi_pitch = self.pitch_converter.up(prev_pitch)
        note.m21_notes.append(self.make_m21_note(note.midi_pitch, note.duration))
      elif note.pitch =="로":
        note.midi_pitch = self.pitch_converter.down2(prev_pitch)
        note.m21_notes.append(self.make_m21_note(note.midi_pitch, note.duration))
      elif note.pitch == "리":
        note.midi_pitch = self.pitch_converter.up2(prev_pitch)
        note.m21_notes.append(self.make_m21_note(note.midi_pitch, note.duration))
      elif note.pitch == "니나*":
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.up(prev_pitch), note.duration/2))
        note.m21_notes.append(self.make_m21_note(prev_pitch, note.duration/2))
        note.midi_pitch = prev_pitch
      elif note.pitch == "느나":
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.down(prev_pitch), note.duration/2))
        note.m21_notes.append(self.make_m21_note(prev_pitch, note.duration/2))
        note.midi_pitch = prev_pitch
      elif note.pitch == "노라":
        note.midi_pitch = self.pitch_converter.down2(prev_pitch)
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.down(prev_pitch), note.duration/2))
        note.m21_notes.append(self.make_m21_note(note.midi_pitch, note.duration/2))
      elif note.pitch == "느니":
        note.midi_pitch = self.pitch_converter.up2(prev_pitch)
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.up(prev_pitch), note.duration/2))
        note.m21_notes.append(self.make_m21_note(note.midi_pitch, note.duration/2))
      elif note.pitch == "니레나":
        note.midi_pitch = prev_pitch
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.up2(prev_pitch), note.duration/3))
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.up(prev_pitch), note.duration/3))
        note.m21_notes.append(self.make_m21_note(prev_pitch, note.duration/3))
      elif note.pitch == "네로나":
        note.midi_pitch = self.pitch_converter.down2(prev_pitch, 3)
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.down(prev_pitch), note.duration/3))
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.down2(prev_pitch), note.duration/3))
        note.m21_notes.append(self.make_m21_note(prev_pitch, note.duration/3))
      elif note.pitch == '니로나':
        note.midi_pitch = self.pitch_converter.down(prev_pitch)
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.up(prev_pitch), note.duration/3))
        note.m21_notes.append(self.make_m21_note(prev_pitch, note.duration/3))
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.down(prev_pitch), note.duration/3))
      elif note.pitch == '니느라니':
        note.midi_pitch = prev_pitch
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.up(prev_pitch), note.duration/4))
        note.m21_notes.append(self.make_m21_note(prev_pitch, note.duration/4))
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.down(prev_pitch), note.duration/4))
        note.m21_notes.append(self.make_m21_note(prev_pitch, note.duration/4))
      elif note.pitch == '느나니나':
        note.midi_pitch = prev_pitch
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.down(prev_pitch), note.duration/4))
        note.m21_notes.append(self.make_m21_note(prev_pitch, note.duration/4))
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.up(prev_pitch), note.duration/4))
        note.m21_notes.append(self.make_m21_note(prev_pitch, note.duration/4))
      elif note.pitch == '느나르나니':
        note.midi_pitch = prev_pitch
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.down(prev_pitch), note.duration/5))
        note.m21_notes.append(self.make_m21_note(prev_pitch, note.duration/5))
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.up(prev_pitch), note.duration/5))
        note.m21_notes.append(self.make_m21_note(prev_pitch, note.duration/5))
        note.m21_notes.append(self.make_m21_note(self.pitch_converter.down(prev_pitch), note.duration/5))
      elif note.pitch == '쉼표':
        note.m21_notes.append(mnote.Rest(duration=music21.duration.Duration(note.duration * self.dur_ratio)))
      elif note.pitch == '-':
        note.midi_pitch = prev_pitch
        pass
      else:
        if verbose: print(f"Unknown pitch {note.pitch} in {note}")
        note.m21_notes.append(mnote.Rest(duration=music21.duration.Duration(note.duration * self.dur_ratio)))
      
      if len(note.ornaments) > 0:
        for orn in note.ornaments:
          if note.midi_pitch is None:
            print(f"note.midi_pitch is None in {note}")
          if orn == '니레':
            grace_pitch = self.pitch_converter.up(note.midi_pitch)
            note.m21_notes = [self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)] + note.m21_notes
          elif orn == '니나':
            grace_pitch = self.pitch_converter.up2(note.midi_pitch)
            note.m21_notes = [self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)] + note.m21_notes
          elif orn == '니로':
            grace_pitch = self.pitch_converter.up2(note.midi_pitch, 3)
            note.m21_notes = [self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)] + note.m21_notes
          elif orn == '노네':
            grace_pitch = self.pitch_converter.down(note.midi_pitch)
            note.m21_notes = [self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)] + note.m21_notes
          elif orn == '너녜':
            grace_pitch = self.pitch_converter.down2(note.midi_pitch)
            note.m21_notes = [self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)] + note.m21_notes
          elif orn == '노니로':
            grace_pitch = self.pitch_converter.up(note.midi_pitch)
            note.m21_notes = [self.make_m21_note(note.midi_pitch, Fraction(1,3), grace=True),
                              self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)] + note.m21_notes
          elif orn == '노리노':
            grace_pitch = self.pitch_converter.up2(note.midi_pitch)
            note.m21_notes = [self.make_m21_note(note.midi_pitch, Fraction(1,3), grace=True),
                              self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)] + note.m21_notes
          elif orn == '네로네':
            grace_pitch = self.pitch_converter.down(note.midi_pitch)
            note.m21_notes = [self.make_m21_note(note.midi_pitch, Fraction(1,3), grace=True),
                              self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)] + note.m21_notes
          elif orn == '느네느':
            grace_pitch = self.pitch_converter.up2(note.midi_pitch, 3)
            note.m21_notes = [self.make_m21_note(note.midi_pitch, Fraction(1,3), grace=True),
                              self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)] + note.m21_notes
          elif orn == '나니로':
            grace_pitch = self.pitch_converter.down(note.midi_pitch)
            grace_pitch2 = self.pitch_converter.up(note.midi_pitch)
            note.m21_notes = [self.make_m21_note(grace_pitch, Fraction(1,3), grace=True),
                              self.make_m21_note(grace_pitch2, Fraction(1,3), grace=True)] + note.m21_notes
          elif orn == '로니로':
            grace_pitch = self.pitch_converter.down2(note.midi_pitch)
            grace_pitch2 = self.pitch_converter.up(note.midi_pitch)
            note.m21_notes = [self.make_m21_note(grace_pitch, Fraction(1,3), grace=True),
                              self.make_m21_note(grace_pitch2, Fraction(1,3), grace=True)] + note.m21_notes
          elif orn == '느로니르':
            grace_pitch = self.pitch_converter.down(note.midi_pitch)
            grace_pitch2 = self.pitch_converter.up(note.midi_pitch)
            note.m21_notes = [self.make_m21_note(note.midi_pitch, Fraction(1,3), grace=True),
                              self.make_m21_note(grace_pitch, Fraction(1,3), grace=True),
                              self.make_m21_note(grace_pitch2, Fraction(1,3), grace=True)] + note.m21_notes
          elif orn == '느니-르':
            grace_pitch = self.pitch_converter.down(note.midi_pitch)
            note.m21_notes = [self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)] \
                                + note.m21_notes + [self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)]
          elif orn == '니루-니':
            grace_pitch = self.pitch_converter.up(note.midi_pitch)
            note.m21_notes = [self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)] \
                            + note.m21_notes + [self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)]
          elif orn == '나니나':
            assert len(note.m21_notes) == 1, '나니나를 처리하는 중에 본음이 한 개 더 남아있음'
            grace_pitch = self.pitch_converter.up(note.midi_pitch)
            note.m21_notes = [
              self.make_m21_note(note.midi_pitch, note.duration/3),
              self.make_m21_note(grace_pitch, note.duration/3),
              self.make_m21_note(note.midi_pitch, note.duration/3),
            ]
          elif orn in ('싸랭', '슬기둥1', '슬기둥2'):
            grace_pitch = self.pitch_converter.pitch2midi['하배황']
            note.m21_notes = [self.make_m21_note(grace_pitch, Fraction(1,3), grace=True)] + note.m21_notes
          else:
            if verbose: print(f"Unknown ornament {orn} in {note}")
            
      if note.midi_pitch is not None:
        prev_pitch = note.midi_pitch
      
    pass
  
  def convert_m21_notes_to_stream(self, notes:List[Note], time_signature='3/8', key_signature=-4):
    stream = music21.stream.Stream()
    stream.append(music21.meter.TimeSignature(time_signature))
    current_key = music21.key.KeySignature(key_signature)
    stream.append(current_key)
    for note in notes:
      for m21_note in note.m21_notes:
        if isinstance(m21_note, mnote.Note):
          m21_note.pitch = music21.pitch.simplifyMultipleEnharmonics(pitches=[m21_note.pitch], keyContext=current_key)[0]
          if m21_note.pitch.accidental and m21_note.pitch.accidental.alter == 0:
            m21_note.pitch.accidental = None
        stream.append(m21_note)
    return stream
  
  def get_scale(self, notes:List[Note]):
    pitch_counter = Counter([note.pitch[-1:] for note in notes if note.pitch[-1:] in self.pitch_names])
    most_five_common = [x[0] for x in pitch_counter.most_common(5)]
    most_five_common.sort(key=lambda x: self.pitch_names.index(x))
    
    exceptional_pitches = [x for x in pitch_counter.keys() if x not in most_five_common]
    return most_five_common, exceptional_pitches
  
  def parse_scale(self, scale:Union[List[str], str]) -> List[str]:
    if scale is None:
      return scale
    if isinstance(scale, str):
      scale = [x for x in scale]
    scale.sort(key=lambda x: self.pitch_names.index(x))
    return scale
  
  def __call__(self, 
               tokens:Union[List[str], str], 
               time_signatures:str='3/8', 
               key_signature:int=-4,
               scale:Union[str, List[str]]=None,
               verbose=False) -> Tuple[List[Note], music21.stream.Stream]:

    if isinstance(tokens, str):
      tokens = tokens.split(' ')

    notes = self.convert_to_notes(tokens)
    scale = self.parse_scale(scale)
    if scale is None:
      scale, exceptional_pitches = self.get_scale(notes)
    if len(scale) < 4:
      self.pitch_converter = SigimsaeConverter()
    else:
      self.pitch_converter = SigimsaeConverter(scale=scale, exceptional_pitches=exceptional_pitches)
    self._fix_three_col_division(notes)
    self.get_duration_of_notes(notes)
    self.create_m21_notes(notes, verbose=verbose)
    stream = self.convert_m21_notes_to_stream(notes)
    return notes, stream
  
  def convert_abc_tokens(self, tokens:List[str],
                         time_signatures:str='3/8', 
                         key_signature:int=-4,
                         scale=None,
                         verbose=False):
    notes = self.convert_abc_tokens_to_notes(tokens)
    scale = self.parse_scale(scale)
    if scale is None:
      scale, exceptional_pitches = self.get_scale(notes)
    if len(scale) < 4:
      self.pitch_converter = SigimsaeConverter()
    else:
      self.pitch_converter = SigimsaeConverter(scale=scale, exceptional_pitches=exceptional_pitches)

    self.create_m21_notes(notes, verbose=verbose)
    stream = self.convert_m21_notes_to_stream(notes)

    return notes, stream
  
  def convert_abc_tokens_to_notes(self, tokens:List[str]):
    global_offset = 0
    jg_offset = 0
    gak_offset = 0
    prev_note = []
    total_notes = []

    for token in tokens:
      if token == '' or token in ('pad', 'start', 'end'): continue
      if token in self.dur_tokens:
        duration = token
        if prev_note:
          total_notes.append(ABCNote(prev_note, duration, global_offset, jg_offset//1, gak_offset))
          prev_note = []
        global_offset += duration
        jg_offset += duration
      elif token  == '\n':
        gak_offset += 1
        jg_offset = 0
      else:
        prev_note.append(token)
    return total_notes

  
  
  def convert_multi_track(self, tokens, scale=None, verbose=False):
    if isinstance(tokens, str):
      assert '\n\n' in tokens, 'tokens가 str인 경우에는 두 개 이상의 트랙이 있어야 함'
      token_by_inst = tokens.split('\n\n')
    else:
      assert len(tokens) > 1, 'tokens가 list인 경우에는 두 개 이상의 트랙이 있어야 함'
      token_by_inst = tokens
    
    entire_notes = []
    entire_stream = music21.stream.Score()
    for inst_tokens in token_by_inst:
      notes, stream = self(inst_tokens, scale=scale, verbose=verbose)
      entire_notes.append(notes)
      entire_stream.insert(0, stream)
    return entire_notes, entire_stream

  def convert_inference_result(self,
                               inf_out:List[List[str]], 
                               source:List[List[str]]=None,
                               gt:List[str]=None):
    output_str = [x[0] for x in inf_out]
    output_note, output_stream = self(output_str)
    source_stream = music21.stream.Score()
    note_dict = {'output': output_note}
    if source is not None:
      source_insts_total = [x[-1] for x in source]
      source_insts = list(set(source_insts_total))
      source_insts.sort(key=lambda x: source_insts_total.index(x))
      source_strs = []
      for inst in source_insts:
          source_strs.append(' '.join([x[0] for x in source if x[-1] == inst]))
      source_strs = ' \n\n '.join(source_strs)
      note_dict['source'], source_stream = self.convert_multi_track(source_strs)
      
      
    if gt is not None:
      note_dict['gt'], gt_stream = self(gt)
      source_stream.insert(0, gt_stream)
    source_stream.insert(0, output_stream)
    
    return note_dict, source_stream

class Symbol:
  def __init__(self, text:str, offset=0):
    # example of text: '배임:5' or '-:8', '임_느니르:3'
    self.offset = offset
    note_and_ornament = text.split(':')[0]
    self.note = note_and_ornament.split('_')[0]
    self.ornament = note_and_ornament.split('_')[1:] if '_' in note_and_ornament else None
    self.duration = None
    self.global_offset = None
    if self.note in PITCH2MIDI:
      self.midi = PITCH2MIDI[self.note]
    else:
      self.midi = 0
  
  def __repr__(self) -> str:
    return f'{self.note}({self.midi}) - {self.ornament}, duration:{self.duration} @ {self.offset}, {self.global_offset}'
  
  def __str__(self) -> str:
    ornaments = '_'.join(self.ornament) if self.ornament is not None else ''
    duration = self.duration if self.duration is not None else ''
    output =  f'{self.note}({self.midi}):{ornaments}:{duration}:{self.offset}:{self.global_offset}'
    if len(output.split(':')) != 5: 
      print(f'output: {output}')
    return output
  



class SymbolReader:
  def __init__(self) -> None:
    self.prev_pitch = 0
    self.prev_offset = 0
    self.dur_ratio = 1.5
    self.sigimsae_conv = SigimsaeConverter()
    self.entire_notes = []
  
  def __call__(self, i, symbol):
      # def handle_symbol(self, i, symbol, prev_pitch, prev_offset, dur_ratio, entire_notes):
    current_offset = (i + symbol.offset) * self.dur_ratio
    if symbol.note == '-':
      return
    if symbol.note == '같은음표':
      symbol.midi = self.prev_pitch
    elif symbol.note == '노':
      symbol.midi = self.sigimsae_conv.down(self.prev_pitch)
    elif symbol.note == '니':
      symbol.midi = self.sigimsae_conv.up(self.prev_pitch)
    elif symbol.note =="로":
      symbol.midi = self.sigimsae_conv.down2(self.prev_pitch)
    elif symbol.note == "리":
      symbol.midi = self.sigimsae_conv.up2(self.prev_pitch)
    elif symbol.note == '니나':
      pass
      # if prev_offset != current_offset:
    if symbol.midi != 0:
      if self.prev_pitch != 0:
        new_note = mnote.Note(self.prev_pitch, quarterLength=current_offset - self.prev_offset)
        self.entire_notes.append(new_note)
      self.prev_offset = current_offset
      self.prev_pitch = symbol.midi
      if symbol.ornament is not None:
        if '니레' in symbol.ornament:
          grace_pitch = self.sigimsae_conv.up(symbol.midi)
          self.entire_notes.append(mnote.Note(grace_pitch, quarterLength=0.5).getGrace())
        if '니나' in symbol.ornament:
          grace_pitch = self.sigimsae_conv.up2(symbol.midi)
          self.entire_notes.append(mnote.Note(grace_pitch, quarterLength=0.5).getGrace())
        if '노네' in symbol.ornament:
          grace_pitch = self.sigimsae_conv.down(symbol.midi)
          self.entire_notes.append(mnote.Note(grace_pitch, quarterLength=0.5).getGrace())
        if '노니로' in symbol.ornament:
          self.entire_notes.append(mnote.Note(symbol.midi, quarterLength=0.5).getGrace())
          grace_pitch = self.sigimsae_conv.up(symbol.midi)
          self.entire_notes.append(mnote.Note(grace_pitch, quarterLength=0.5).getGrace())

  def handle_remaining_note(self, i):
    current_offset = i * self.dur_ratio
    new_note = mnote.Note(self.prev_pitch, quarterLength=current_offset - self.prev_offset)
    self.entire_notes.append(new_note)

class JeongganboParser:
  def __init__(self) -> None:
    self.beat_template = self.define_beat_template()
    self.error_text_templates = self.define_error_text_templates()

    self.num_jg_in_gak = 20
    self.num_sharp = -4
    pass
  
  @staticmethod
  def define_beat_template():
    template = {}
    template[(5,)] = [0]
    template[(10, 11)] = [0, Fraction(1,2)]
    template[(10, 14, 15)] = [0, Fraction(1,2), Fraction(3,4)]
    template[(12, 13, 11)] = [0, Fraction(1,4), Fraction(1,2)]
    template[(12, 13, 14, 15)] = [0, Fraction(1,4), Fraction(1,2), Fraction(3,4)]
    template[(2, 5, 8)] = [0, Fraction(1,3), Fraction(2,3)]
    template[(2, 5, 7, 9)] = [0, Fraction(1,3), Fraction(2,3), Fraction(5,6)]
    template[(2, 4, 6, 8)] = [0, Fraction(1,3), Fraction(1,2), Fraction(2,3)]
    template[(2, 4, 6, 7, 9)] = [0, Fraction(1,3), Fraction(1,2), Fraction(2,3), Fraction(5,6)]
    template[(1, 3, 5, 8)] = [0, Fraction(1,6), Fraction(1,3), Fraction(2,3)]
    template[(1, 3, 5, 7, 9)] = [0, Fraction(1,6), Fraction(1,3), Fraction(2,3), Fraction(5,6)]
    template[(1, 3, 4, 6, 8)] = [0, Fraction(1,6), Fraction(1,3), Fraction(1,2), Fraction(2,3)]
    template[(1, 3, 4, 6, 7, 9)] = [0, Fraction(1,6), Fraction(1,3), Fraction(1,2), Fraction(2,3), Fraction(5,6)]

    return template

  @staticmethod
  def define_error_text_templates():
    '''
    hard-coded error text templates
    '''
    error_text_templates = {}
    error_text_templates['고_니레:2 -:5 니나:8'] = '청태_니레:2 -:5 니나:8'
    return error_text_templates

  @staticmethod
  def get_position_tuple(omr_text:str):
    return tuple([int(y) for y in  re.findall(r':(\d+)', omr_text)])



  # def piece_to_score(self, piece:Piece):
  #   sb_reader = SymbolReader()
  #   for i, jeonggan in enumerate(piece.jeonggans):
  #     if i == 1920: # In Yeominlak, this is where the tempo changes
  #       sb_reader.handle_remaining_note(i)
  #       sb_reader.prev_pitch = 0
  #       sb_reader.dur_ratio = 3.0
  #       sb_reader.prev_offset = sb_reader.prev_offset * 2
  #     for symbol in jeonggan.symbols:
  #       sb_reader(i, symbol)
  #   sb_reader.handle_remaining_note(i+1)
  #   score = self.make_score_from_notes(sb_reader.entire_notes)
  #   return score

  def make_score_from_notes(self, entire_notes: List[mnote.Note]):
    score = stream.Stream()
    score.append(mmeter.TimeSignature(f'{int(self.num_jg_in_gak*3)}/8'))
    current_key = mkey.KeySignature(self.num_sharp)
    score.append(current_key)

    for note in entire_notes:
      note.pitch = mpitch.simplifyMultipleEnharmonics(pitches=[note.pitch], keyContext=current_key)[0]
      if note.pitch.accidental.alter == 0:
        note.pitch.accidental = None # delete natural
      score.append(note)
    return score
  
  def parse_omr_results(self, jg_str):
    # text = jeonggan.omr_text.replace(':5:5', ':5').replace(':1:1', ':1') # Hard-coded fix for a specific error
    notes = jg_str.split(' ')
    notes = [x for x in notes if len(x)>2]
    text = ' '.join(notes)
    position_tuple = self.get_position_tuple(text)
    if position_tuple not in self.beat_template:
      print(f'position_tuple {position_tuple} not in template', text)
      return
    by_note_position = self.beat_template[position_tuple]
    out = [Symbol(notes[i], offset=by_note_position[i]) for i in range(len(notes))]
    return out


  @staticmethod
  def parse_duration_and_offset(jeonggans: List[str]):
    '''
    Parse duration and offset of each symbol in jeonggans
    '''
    prev_offset = 0 
    prev_pitch = 0
    prev_symbol = None
    for i, jeonggan in enumerate(jeonggans):
      for symbol in jeonggan.symbols:
        symbol.global_offset = i + symbol.offset
        if symbol.note == '-':
          continue
        if symbol.note == '같은음표':
          symbol.midi = prev_pitch
        if prev_symbol is not None:
          prev_symbol.duration = symbol.global_offset - prev_offset
        prev_offset = symbol.global_offset
        prev_pitch = symbol.midi
        prev_symbol = symbol
    prev_symbol.duration = len(jeonggans) - prev_offset
    return None
  
  def __call__(self, token_str:str):
    sb_reader = SymbolReader()
    for i, jeonggan in enumerate(piece.jeonggans):
      if i == 1920: # In Yeominlak, this is where the tempo changes
        sb_reader.handle_remaining_note(i)
        sb_reader.prev_pitch = 0
        sb_reader.dur_ratio = 3.0
        sb_reader.prev_offset = sb_reader.prev_offset * 2
      for symbol in jeonggan.symbols:
        sb_reader(i, symbol)
    sb_reader.handle_remaining_note(i+1)
    score = self.make_score_from_notes(sb_reader.entire_notes)

    return

def piece_to_txt(piece):
  symbols_in_gaks = [','.join([str(symbol) for jeonggan in gak.jeonggans for symbol in jeonggan.symbols]) for gak in piece.gaks if not gak.is_jangdan]
  symbols_in_gaks = '\n'.join(symbols_in_gaks)
  return symbols_in_gaks

class ABCtoGenConverter:
  def __init__(self):
    self.to_omr_converter = JGConverter(jeonggan_quarter_length=Fraction(1.0))
    self.gencode_converter = GencodeConverter()
    self.decoder = JGToStaffConverter()
  
  @staticmethod
  def group_by_attribute(alist:List, attribute:str):
    outputs = []
    prev_attribute = None
    for a in alist:
      if prev_attribute is None:
        prev_attribute = getattr(a, attribute)
        outputs.append([a])
      elif getattr(a, attribute) == prev_attribute:
        outputs[-1].append(a)
      else:
        prev_attribute = getattr(a, attribute)
        outputs.append([a])
    return outputs
  
  def __call__(self, tokens:List[str]):
    notes, _ = self.decoder.convert_abc_tokens(tokens)
    note_by_measure = self.group_by_attribute(notes, 'gak_offset')
    part_text = []
    for i, note_in_measure in enumerate(note_by_measure):
      conv_jgs = self.to_omr_converter.list_of_abc_notes_to_jeonggan(note_in_measure)
      text_jgs = self.to_omr_converter.jeonggan_note_to_text(conv_jgs)
      part_text.append('|'.join(text_jgs))
    return self.gencode_converter.convert_lines_to_gencode(part_text)