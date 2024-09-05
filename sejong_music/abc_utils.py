from typing import Any, List, Union, Tuple
from fractions import Fraction

class ABCNote:
  def __init__(self, pitch:List[str], duration:float, global_offset:float, jg_offset:int=0, gak_offset:int=0):
    self.pitch = pitch[0]
    self.ornaments = pitch[1:]
    self.duration = duration
    self.global_offset = global_offset
    self.jg_offset = jg_offset
    self.gak_offset = gak_offset
    self.is_rest = self.pitch == '쉼표'
    self.midi_pitch = None
    self.m21_notes = []

  def __repr__(self) -> str:
    return f"ABCNote({self.pitch}_{'_'.join(self.ornaments)}, dur:{self.duration}, {self.global_offset}, gak:{self.gak_offset}, jg:{self.jg_offset})"

  @property
  def offset(self):
    return self.jg_offset


class BeatToken:
  def __init__(self, beat_offset) -> None:
    self.beat = beat_offset
    self.gen = None
  def __repr__(self) -> str:
    return f"beat:{self.beat} gen:{self.gen}"
  
class PosToken:
  def __init__(self, pos, int_pos) -> None:
    self.pos = pos
    self.adjust_pos = f":{int_pos}"
  
  def __repr__(self) -> str:
    return f"pos:{self.pos} adjust_pos:{self.adjust_pos}"
  
class TokenList:
  def __init__(self) -> None:
    self.tokens = []
  
  def __len__(self):
    return len(self.tokens)
  
  def append(self, token):
    self.tokens.append(token)
  
  @property
  def last_pos_token(self):
    for token in reversed(self.tokens):
      if isinstance(token, PosToken):
        return token
  
  @property
  def penultimate_pos_token(self):
    num_pos_founded = 0
    for token in reversed(self.tokens):
      if isinstance(token, PosToken):
        num_pos_founded += 1
      if num_pos_founded == 2:
        return token
  
  @property
  def last_beat_token(self):
    for token in reversed(self.tokens):
      if isinstance(token, BeatToken):
        return token
  
  @property
  def penultimate_beat_token(self):
    num_beat_founded = 0
    for token in reversed(self.tokens):
      if isinstance(token, BeatToken):
        num_beat_founded += 1
      if num_beat_founded == 2:
        return token
  
  @property
  def gen_tokens(self):
    assert all([token.gen for token in self.tokens if isinstance(token, BeatToken)]), f"gen_tokens not resolved: {self.tokens}"
    return [token.gen if isinstance(token, BeatToken) else token for token in self.tokens]
  
  def __iter__(self):
    return iter(self.tokens)
  

def convert_beat_jg_to_gen(tokens:List[str]) -> List[str]:
  clean_tokens = TokenList()
  for token in tokens:
    if token[:4] == 'beat':
      if clean_tokens.last_beat_token and clean_tokens.last_beat_token.gen is None:
        if clean_tokens.last_beat_token.beat == 'beat:0':
          if token == 'beat:1/2':
            clean_tokens.last_beat_token.gen = ':10'
          elif token == 'beat:2/3':
            clean_tokens.last_beat_token.gen = ':2'
          elif token == 'beat:1/3':
            clean_tokens.last_beat_token.gen = ':2'
          elif token == 'beat:1/6':
            clean_tokens.last_beat_token.gen = ':1'
          elif token == 'beat:5/6':
            clean_tokens.last_beat_token.gen = ':2'
          elif token == 'beat:3/4':
            clean_tokens.last_beat_token.gen = ':10'
          elif token == 'beat:1/4':
            clean_tokens.last_beat_token.gen = ':12'
          elif token == 'beat:1/9':
            clean_tokens.last_beat_token.gen = ':1'
          elif token == 'beat:0':
            clean_tokens.last_beat_token.gen = ':1'
          elif token in ('beat:4/9', 'beat:5/9', 'beat:7/9', 'beat:8/9'):
            clean_tokens.last_beat_token.gen = ':2'
            
        elif clean_tokens.last_beat_token.beat == 'beat:1/6':
            clean_tokens.last_beat_token.gen = ':3'
        elif clean_tokens.last_beat_token.beat == 'beat:1/3':
          if token in ('beat:2/3', 'beat:5/6', 'beat:7/9', 'beat:8/9'):
            clean_tokens.last_beat_token.gen = ':5'
          elif token == 'beat:1/2':
            clean_tokens.last_beat_token.gen = ':4'
        elif clean_tokens.last_beat_token.beat == 'beat:1/2':
          if token == 'beat:2/3':
            clean_tokens.last_beat_token.gen = ':6'
        elif clean_tokens.last_beat_token.beat == 'beat:2/3':
          if token in ('beat:5/6', 'beat:7/9', 'beat:8/9'):
            clean_tokens.last_beat_token.gen = ':7'
        elif clean_tokens.last_beat_token.beat == 'beat:1/4':
          clean_tokens.last_beat_token.gen = ':13'
        elif clean_tokens.last_beat_token.beat == 'beat:1/9':
          clean_tokens.last_beat_token.gen = ':2'
        elif clean_tokens.last_beat_token.beat == 'beat:2/9':
          clean_tokens.last_beat_token.gen = ':3'
        elif clean_tokens.last_beat_token.beat == 'beat:5/6':
          clean_tokens.last_beat_token.gen = ':9'
      if token in ('beat:2/3', 'beat:5/6', 'beat:7/9', 'beat:8/9'):
        if clean_tokens.penultimate_beat_token and clean_tokens.penultimate_beat_token.gen and \
          clean_tokens.penultimate_beat_token.gen == ':10' and clean_tokens.penultimate_beat_token.beat == 'beat:0':
            clean_tokens.penultimate_beat_token.gen = ':2'
      clean_tokens.append(BeatToken(token))
      if token == 'beat:7/9': clean_tokens.last_beat_token.gen = ':8'
      elif token == 'beat:8/9': clean_tokens.last_beat_token.gen = ':9'
      elif token == 'beat:4/9': clean_tokens.last_beat_token.gen = ':5'
      elif token == 'beat:5/9': clean_tokens.last_beat_token.gen = ':6'
      elif token == 'beat:1/6': clean_tokens.last_beat_token.gen = ':3'

    elif token.strip() in ('|' '\n'):
      # if clean_tokens.penultimate_beat_token and clean_tokens.penultimate_beat_token.gen and \
      #   clean_tokens.penultimate_beat_token.gen == ':10' and clean_tokens.penultimate_beat_token.beat == 'beat:0':
      #     clean_tokens.penultimate_beat_token.gen == ':2'

      clean_tokens.append(token)
      if clean_tokens.last_beat_token is None or clean_tokens.last_beat_token.gen:
        continue
      if clean_tokens.last_beat_token.beat == 'beat:0':
        clean_tokens.last_beat_token.gen = ':0'
      elif clean_tokens.last_beat_token.beat == 'beat:1/6':
        clean_tokens.last_beat_token.gen = ':1'
      elif clean_tokens.last_beat_token.beat == 'beat:1/3':
        clean_tokens.last_beat_token.gen = ':5'
      elif clean_tokens.last_beat_token.beat == 'beat:2/3':
        clean_tokens.last_beat_token.gen = ':8'
      elif clean_tokens.last_beat_token.beat == 'beat:5/6':
        clean_tokens.last_beat_token.gen = ':9'
      elif clean_tokens.last_beat_token.beat == 'beat:1/2':
        if clean_tokens.penultimate_beat_token.gen in (':0', ':10', ':11'):
          clean_tokens.last_beat_token.gen = ':11'
        elif clean_tokens.penultimate_beat_token.gen in (':2', ':3', ':4'):
          clean_tokens.last_beat_token.gen = ':6'
        else:
          clean_tokens.last_beat_token.gen = ':11'
      elif clean_tokens.last_beat_token.beat == 'beat:3/4':
        clean_tokens.last_beat_token.gen = ':15'
    else:
      clean_tokens.append(token)
  return clean_tokens.gen_tokens

