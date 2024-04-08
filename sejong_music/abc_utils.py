from typing import Any, List, Union, Tuple


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

