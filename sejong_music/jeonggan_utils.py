from typing import List
import copy
import music21
from fractions import Fraction
from .utils import apply_tie, Gnote
from .abc_utils import ABCNote

class JGConverter:
  def __init__(self, jeonggan_quarter_length=1.5, num_jeonggan_per_gak=20):
    self.PITCH2MIDI = self.define_pitch2midi()
    self.MIDI2PITCH = {v:k for k,v in self.PITCH2MIDI.items()}
    self.jql = jeonggan_quarter_length
    self.num_jeonggan_per_gak = num_jeonggan_per_gak

  
  def __call__(self, score: music21.stream.Score, duration_multiplier=1.0):
    output = []
    if hasattr(score, 'parts'):
      parts = score.parts
    else:
      parts = [score]
    for idx, part in enumerate(parts):
      num_measures = len(part)
      part_text = []
      for i in range(1, num_measures):
        notes_in_measure = list(part.measures(i, i).flat.notesAndRests)
        tie_cleaned_notes = apply_tie(notes_in_measure, 7)
        for note in tie_cleaned_notes:
          for mnote in note.note:
            mnote.duration.quarterLength *= duration_multiplier
          note.offset *= duration_multiplier
        conv_jgs = self.list_of_m21_notes_to_jeonggan(tie_cleaned_notes)
        text_jgs = self.jeonggan_note_to_text(conv_jgs)
        part_text.append('|'.join(text_jgs))
      output.append('\n'.join(part_text))
      
    return output


  def list_of_m21_notes_to_jeonggan(self, sel_meas:List[Gnote]): 
    conv_jgs = [[] for _ in range(self.num_jeonggan_per_gak)]
    # if len(sel_meas) != 0 and sel_meas[0].pitch == 0:
    #   sel_meas = sel_meas[1:]
    for note in sel_meas:
      cur_jg = conv_jgs[int(note.offset//self.jql)]
      jg_offset = note.offset % self.jql
      cur_jg.append(((self.MIDI2PITCH[note.pitch]), note.duration, jg_offset))
      # cur_jg.append(((self.MIDI2PITCH[note.pitch.midi]), note.duration.quarterLength, jg_offset))
    return conv_jgs

  def list_of_abc_notes_to_jeonggan(self, sel_meas:List[ABCNote]): 
    conv_jgs = [[] for _ in range(self.num_jeonggan_per_gak)]
    for note in sel_meas:
      cur_jg = conv_jgs[int(note.offset//self.jql)]
      beat_offset = note.global_offset % self.jql
      cur_jg.append(  ('_'.join([note.pitch] + note.ornaments), note.duration, Fraction(beat_offset).limit_denominator(6) )  )
      # cur_jg.append(((self.MIDI2PITCH[note.pitch.midi]), note.duration.quarterLength, jg_offset))
    return conv_jgs

  def jeonggan_note_to_text(self, conv_jgs):
    text_jgs = ['' for _ in range(len(conv_jgs))] 
    for i, jg in enumerate(conv_jgs):
      if len(jg) == 0:
        text_jgs[i] = '-:5'

      if len(jg) != 0 and jg[0][2] != 0:
        start_pos = jg[0][2]
        if start_pos == self.jql / 6:
          text_jgs[i] = '-:1'
        elif start_pos == self.jql / 4:
          text_jgs[i] = '-:12'
        elif start_pos == self.jql / 3:
          text_jgs[i] = '-:2'
        elif start_pos == self.jql / 2:
          text_jgs[i] = '-:10'
        elif start_pos == self.jql / 3 * 2:
          text_jgs[i] = '-:2 -:5'
        elif start_pos == self.jql / 6 * 5:
          text_jgs[i] = '-:2 -:5 -:7'
        else:
          print(f"None of the start_pos is matched: {start_pos}")

      ornament_on = False
      for note in jg:

        if note[1] == 0 and note[0] =='하배황':
          ornament_on = True
          continue

        if note[2] == 0:
          if note[1] >= self.jql:
            text_jgs[i] = f'{note[0]}:5'
          elif note[1] == self.jql / 3 * 2:
            text_jgs[i] += f'{note[0]}:2 -:5'
          elif note[1] == self.jql / 2:
            text_jgs[i] += f'{note[0]}:10'
          elif note[1] == self.jql / 3:
            text_jgs[i] += f'{note[0]}:2'
          elif note[1] == self.jql / 4:
            text_jgs[i] += f'{note[0]}:12'
          elif note[1] == self.jql / 6:
            text_jgs[i] += f'{note[0]}:1'
          elif note[1] == self.jql * 5 / 6:
            text_jgs[i] += f'{note[0]}:2 -:5 -:7'
          else:
            print(f"None of the note[1] is matched while note[2]==0: {note}")
        elif note[2] == self.jql / 6:
          if note[1] >= self.jql / 6 * 5:
            text_jgs[i] += f' {note[0]}:3 -:5 -:8'
          elif note[1] == self.jql / 3 * 2:
            text_jgs[i] += f' {note[0]}:3 -:5 -:7'
          elif note[1] == self.jql / 2:
            text_jgs[i] += f' {note[0]}:3 -:5'
          elif note[1] == self.jql / 3:
            text_jgs[i] += f' {note[0]}:3 -:4'
          elif note[1] == self.jql / 6:
            text_jgs[i] += f' {note[0]}:3'
          else:
            print(f"None of the note[1] is matched while note[2]==0.25: {note}")
        elif note[2] == self.jql / 4: # 0.375
          if note[1] >= self.jql / 4 * 3:
            text_jgs[i] += f' {note[0]}:13 -:11'
          elif note[1] == self.jql / 2:
            text_jgs[i] += f' {note[0]}:13 -:14'
          elif note[1] == self.jql / 4:
            text_jgs[i] += f' {note[0]}:13'
          else:
            print(f"None of the note[1] is matched while note[2]==0.375: {note}")
        elif note[2] == self.jql / 3:
          if note[1] >= self.jql / 3 * 2:
            text_jgs[i] += f' {note[0]}:5 -:8'
          elif note[1] == self.jql / 2:
            text_jgs[i] += f' {note[0]}:5 -:7'
          elif note[1] == self.jql / 3:
            text_jgs[i] += f' {note[0]}:5'
          elif note[1] == self.jql / 6:
            text_jgs[i] += f' {note[0]}:4'
          else:
            print(f"None of the note[1] is matched while note[2]==0.5: {note}")
        elif note[2] == self.jql / 2:
          if note[1] >= self.jql / 2:
            if text_jgs[i][-2:] == ':4':
              text_jgs[i] += f' {note[0]}:6 -:8' 
            else:
              text_jgs[i] += f' {note[0]}:11'
          elif note[1] == self.jql / 4:
            text_jgs[i] += f' {note[0]}:14'
          elif note[1] == self.jql / 3:
            if text_jgs[i][-3:] == ':10':
              text_jgs[i] = text_jgs[i].replace(':10', ':2 -:4')
            text_jgs[i] += f' {note[0]}:6 -:7'
          elif note[1] == self.jql / 6:
            if text_jgs[i][-3:] == ':10':
              text_jgs[i] = text_jgs[i].replace(':10', ':2 -:4')
            text_jgs[i] += f' {note[0]}:6'
          else:
            print(f"None of the note[1] is matched while note[2]==0.75: {note}")
        elif note[2] == self.jql / 3 * 2:
          if note[1] >= self.jql / 3:
            text_jgs[i] += f' {note[0]}:8'
          elif note[1] == self.jql / 6:
            text_jgs[i] += f' {note[0]}:7'
          else:
            print(f"None of the note[1] is matched while note[2]==1.0: {note}")
        elif note[2] == self.jql / 6 * 5:
          if note[1] >= self.jql / 6:
            text_jgs[i] += f' {note[0]}:9'
          else:
            print(f"None of the note[1] is matched while note[2]==1.25: {note}")
        else:
          print(f"None of the note[2] is matched: {note}")
        
        if ornament_on:
          new_text = ':'.join(text_jgs[i].split(':')[:-1]) + '_슬기둥1:' + text_jgs[i].split(':')[-1]
          text_jgs[i] = new_text
          ornament_on = False
    return text_jgs

  @staticmethod
  def define_pitch2midi():
    pitch2midi = {}
    pitch_name = ["황" ,'대' ,'태' ,'협' ,'고' ,'중' ,'유' ,'임' ,'이' ,'남' ,'무' ,'응']
    octave_name = {"하하배": -3, "하배": -2, "배":-1, '':0, '청':+1, '중청':+2}

    for o in octave_name.keys():
      for i, p in enumerate(pitch_name):
        pitch2midi[o+p] = i + 63 + octave_name[o] * 12
    pitch2midi['쉼표'] = 0
    return pitch2midi


class GencodeConverter:
  @staticmethod
  def split_line_to_jg(line):
    jgs = line.split('|')
    cleaned_jgs = []
    for jg in jgs:
      if 'OR' in jg:
        cleaned_jgs.append(jg.split('OR')[1])
      else:
        cleaned_jgs.append(jg)
    return cleaned_jgs

  @staticmethod
  def split_jg_to_notes(jg):
    notes = jg.split(' ')
    notes = [note for note in notes if len(note) > 0]
    return notes

  @classmethod
  def convert_jg_to_gencode(cls, jg):
    notes = cls.split_jg_to_notes(jg)
    pitches = [x.split(':')[0] for x in notes]
    positions = [x.split(':')[1] for x in notes]
    if set(pitches) == {'-'}:
      return ''
    if len(positions) == 1:
      positions[0] = '0'
    outputs = []
    for pos, pitch in zip(positions, pitches):
      if pitch == '-':
        continue
      outputs.append(f":{pos} {pitch.replace('_', ' ')}")
    return ' '.join(outputs)

  @classmethod
  def convert_line_to_gencode(cls, line:str):
    return ' | '.join([cls.convert_jg_to_gencode(jg) for jg in cls.split_line_to_jg(line)])
    
  @classmethod
  def convert_lines_to_gencode(cls, lines:List[str]):
    return ' \n '.join([cls.convert_line_to_gencode(line) for line in lines])
  
  @classmethod
  def convert_txt_to_gencode(cls, txt_fn, multi_inst=False):
    if not txt_fn.endswith('.txt'):
      lines = txt_fn
    else:
      with open(txt_fn, 'r') as f:
        lines = f.read()
    if multi_inst:
      insts = lines.split('\n\n')
      insts = [inst for inst in insts if len(inst)>0]
      return '\n\n'.join([cls.convert_lines_to_gencode(inst.split('\n')) for inst in insts])
    lines = lines.split('\n')
    return cls.convert_lines_to_gencode(lines)

  def reverse_convert(self, gencode):
    return


class RollToJGConverter(JGConverter):
  PITCH2MIDI = JGConverter.define_pitch2midi()
  MIDI2PITCH = {v:k for k,v in PITCH2MIDI.items()}

  def __init__(self, jql=1.5):
    self.jql=jql
    pass

  @staticmethod
  def extract_notes(roll):
    notes = []
    prev_frame = 0
    for i, frame in enumerate(roll):
      if frame[0] in ('start'): 
        prev_frame = i
        continue
      if frame[0] in ('end', 'mask', 'pad'):
        prev_frame += 1
        continue
      if frame[0] in ('-', '비어있음') and frame[1] == '비어있음': continue
      if frame[0] != '-':
        if notes: 
          for j in range(1, 5):
            if notes[-j][0] != '-':
              notes[-j].append(f'dur:{i-prev_frame}')
              break
            else:
              notes[-j].append(f'dur:{0}')
        prev_frame = i
      notes.append(frame)
    notes[-1].append(f'dur:{len(roll)-prev_frame}')

    return notes
  
  @staticmethod
  def get_jg_idx(note:List[str]):
    return int(note[3][3:])
  
  @staticmethod
  def get_beat_offset(note:List[str]):
    return int(note[2][5:])/6* 1.5
  
  @staticmethod
  def get_duration(note:List[str]):
    return int(note[6][4:])/6 * 1.5
  
  @staticmethod
  def make_pitch_orn_string(note:List[str]):
    pitch = note[0]
    ornament = note[1]
    if ornament=='비어있음':
      return pitch
    return f'{pitch}_{ornament}'
  
  def list_of_str_notes_to_jeonggan(self, notes:List[List[str]], max_jg_per_gak:int): 
    conv_jgs = [[] for _ in range(max_jg_per_gak)]
    # if len(sel_meas) != 0 and sel_meas[0].pitch == 0:
    #   sel_meas = sel_meas[1:]
    for note in notes:
      cur_jg = conv_jgs[self.get_jg_idx(note)]
      jg_offset = self.get_beat_offset(note)
      cur_jg.append((self.make_pitch_orn_string(note), self.get_duration(note), jg_offset))
      # cur_jg.append(((self.MIDI2PITCH[note.pitch.midi]), note.duration.quarterLength, jg_offset))
    return conv_jgs

  
  def __call__(self, roll:List[List[str]]):
    max_jg_per_gak = max([self.get_jg_idx(x) for x in roll if x[3][:3]=='jg:'])+1
    notes = self.extract_notes(copy.copy(roll))
    gak_ids = sorted(list(set([note[4] for note in notes])))
    out = []
    for gak_id in gak_ids:
      gak_notes = [x for x in notes if x[4]==gak_id]
      conv_jgs = self.list_of_str_notes_to_jeonggan(gak_notes, max_jg_per_gak)
      text_jgs = self.jeonggan_note_to_text(conv_jgs)
      out.append('|'.join(text_jgs))
    return '\n'.join(out)
