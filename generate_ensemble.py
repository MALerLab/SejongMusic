import argparse
from pathlib import Path
from omegaconf import OmegaConf
from fractions import Fraction
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any

import torch
from music21 import converter, stream, note as m21_note 

from sejong_music import model_zoo, jg_code, inference
from sejong_music.yeominrak_processing import OrchestraScoreSeq, ShiftedAlignedScore, Tokenizer
from sejong_music.model_zoo import JeongganTransSeq2seq
from sejong_music.decode import MidiDecoder, OrchestraDecoder
from sejong_music.inference import JGInferencer
from sejong_music.jg_to_staff_converter import JGToStaffConverter, JGCodeToOMRDecoder, ABCtoGenConverter
from sejong_music.jg_code import JeongganDataset, JeongganTokenizer, JeongganPiece, ABCPiece, ABCDataset
from sejong_music.jeonggan_utils import JGConverter, GencodeConverter
from sejong_music.abc_utils import convert_beat_jg_to_gen
from sejong_music.full_inference import Generator



if __name__ == '__main__':
  config = OmegaConf.load('yamls/gen_settings/jg_cph.yaml')
  out_dir = Path('gen_results/')
  out_dir.mkdir(parents=True, exist_ok=True)
  name = 'chihwapyeong_bert_orchestra_20beat_down_org'
  
  # inst_cycles = ['piri', 'daegeum',  'haegeum', 'geomungo', 'gayageum', 'ajaeng']
  inst_cycles = ['piri', 'geomungo', 'gayageum', 'ajaeng', 'haegeum', 'daegeum']

  gen = Generator(config)
  gen.model.to('cuda')
  # txt_fn = 'music_score/chwipunghyeong_bert_20beat_gen.txt'
  # txt_fn = 'music_score/chwipunghyeong_bert_gen.txt'
  # txt_fn = 'music_score/chihwapyeong_bert_20beat_gen.txt'
  txt_fn = 'music_score/chihwapyeong_down_bert_20beat_gen.txt'
  
  output_str = gen.inference_from_gen_code(txt_fn, inst_cycles)
  with open(out_dir / f'{name}_gen.txt', 'w') as f:
    f.write(output_str)

  jg_omr_str = gen.jg_to_omr_converter.convert_multi_inst_str(output_str)
  with open(out_dir / f'{name}_omr.txt', 'w') as f:
    f.write(jg_omr_str)

  notes, score = gen.jg_to_staff_converter.convert_multi_track(output_str, time_signature='60/8')

  txt_fn = out_dir / f'{name}_gen.txt'
  output_str = gen.cycle_inference_from_gen_code(txt_fn, inst_cycles=inst_cycles, num_cycles=6)
  with open(out_dir / f'{name}_cycle_gen.txt', 'w') as f:
    f.write(output_str)

  jg_omr_str_cycle = gen.jg_to_omr_converter.convert_multi_inst_str(output_str)
  with open(out_dir / f'{name}_cycle_omr.txt', 'w') as f:
    f.write(jg_omr_str_cycle)
  
  cycle_notes, cycle_score = gen.jg_to_staff_converter.convert_multi_track(output_str, time_signature='60/8')
  cycle_score.write('musicxml', out_dir / f'{name}_cycle.musicxml')
  score.write('musicxml', out_dir / f'{name}.musicxml' )
  
