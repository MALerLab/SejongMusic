import argparse
import datetime
import copy
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


class Generator:
  def __init__(self, config, model=None, output_dir=None, inferencer=None, is_abc=False):
    self.is_abc = is_abc
    self.jg_decoder = ABCtoGenConverter()
    if config:
      self.config = config
      self.output_dir = Path(config.output_dir)
      self.output_dir.mkdir(parents=True, exist_ok=True)
      self.model = self._load_model()
      self.tokenizer = self.model.tokenizer
      self.inferencer = self._get_inferencer()
      self.use_offset = config.use_offset
    else:
      self.model = model
      self.tokenizer = self.model.tokenizer
      self.inferencer = inferencer
      self.output_dir = output_dir
      self.use_offset = 'beat:0' in self.tokenizer.vocab
    self.jg_converter = JGConverter(jeonggan_quarter_length=1.5)
    self.gen_converter = GencodeConverter()
    self.jg_to_staff_converter = JGToStaffConverter()
    self.jg_to_omr_converter = JGCodeToOMRDecoder()
    self.beat_to_gen_converter = convert_beat_jg_to_gen
    
  
  def _load_model(self):
    model_state_path = Path(self.config.inst_state_dir) / 'iter34800_model.pt'
    tokenizer_vocab_path = Path(self.config.inst_state_dir) / 'tokenizer_vocab.json'
    model_config_path = Path(self.config.inst_state_dir) / 'config.yaml'
    tokenizer:JeongganTokenizer = getattr(jg_code, self.config.class_names.tokenizer)(None, None, json_fn=tokenizer_vocab_path)
    model_config = OmegaConf.load(model_config_path).model
    model:JeongganTransSeq2seq = getattr(model_zoo, self.config.class_names.model)(tokenizer, model_config)
    model.load_state_dict(torch.load(model_state_path))
    model.eval()
    return model
  
  def _get_inferencer(self):
    inferencer:JGInferencer = getattr(inference, self.config.class_names.inferencer)(self.model, 
                                                                                     self.tokenizer, 
                                                                                     is_orch=True,
                                                                                     temperature=self.config.temperature,
                                                                                     top_p=self.config.top_p,
                                                                                     is_abc=self.is_abc)
    return inferencer

  def convert_xml_to_gen_code(self, score_fn, part_idx=-1):
    score = converter.parse(score_fn)
    part = score.parts[part_idx]

    jg_str = self.jg_converter(part, duration_multiplier=3.0)
    gen_str = self.gen_converter.convert_txt_to_gencode(jg_str[0])
    gen_str = ' \n '.join(gen_str.split(' \n ')[:10])
    return gen_str

  def prepare_dataset(self, gen_str, inst_list:List[str]):
    if self.is_abc:
      piece = ABCPiece(None, gen_str=gen_str, inst_list=inst_list)
      dataset = ABCDataset(piece_list=[piece], tokenizer=self.tokenizer)
    else:
      piece = JeongganPiece(None, gen_str=gen_str, inst_list=inst_list, use_offset=self.use_offset)
      dataset = JeongganDataset(piece_list=[piece], tokenizer=self.tokenizer, use_offset=self.use_offset)
    
    return dataset
  
  
  def make_full_inference_on_dataset(self, 
                                     dataset:JeongganDataset, 
                                     target_inst:str, 
                                     condition_instruments:List[str]) -> torch.Tensor:
    outputs = []
    prev_generation = None

    for i in tqdm(range(len(dataset))):
    # for i in tqdm(range(10)):
      sample, _, _ = dataset.get_processed_feature(condition_instruments, condition_instruments[0], i, force_target_inst=target_inst)  
      sample = torch.LongTensor(sample)
      while True:
        try:        
          _, output_decoded, (attention_map, output, new_out) = self.inferencer.inference(sample, target_inst, prev_generation=prev_generation)
          break
        except Exception as e:
          print('에러가 났습니다. 다시 시도합니다.')
        
      if i == 0:
        sel_out = torch.cat([output[0:1]] + [self.get_measure_specific_output(output, self.tokenizer.tok2idx[f'gak:{i}']) for i in range(0,3)], dim=0)
      else:
        sel_out = self.get_measure_specific_output(output, self.tokenizer.tok2idx[f'gak:2']) 
      # pp.pprint(output_decoded[:5])
      if self.is_abc:
        prev_generation = self.get_measure_shifted_output(output, self.tokenizer.tok2idx['gak:1'], self.tokenizer.tok2idx['gak:3'], self.tokenizer.tok2idx['beat:0'])
      else: 
        prev_generation = self.get_measure_shifted_output(output, self.tokenizer.tok2idx['gak:1'], self.tokenizer.tok2idx['gak:3'], self.tokenizer.tok2idx['prev|'])
      outputs.append(sel_out)
    outputs.append(self.get_measure_specific_output(output, self.tokenizer.tok2idx[f'gak:3'])) # add last measure
    outputs_tensor = torch.cat(outputs, dim=0)

    return outputs_tensor
  
  def make_inst_sequential_inference(self, gen_str, target_order:List[str]):
    '''
    target_order: ['geomungo', 'gayageum', 'ajaeng', 'piri', 'daegeum']
    '''
    for i, inst in enumerate(target_order[1:], start=1):
      inst_list = list(reversed(target_order[:i]))
      print(inst_list)
      dataset = self.prepare_dataset(gen_str, inst_list=inst_list) # reverse order
      outputs_tensor = self.make_full_inference_on_dataset(dataset, 
                                                           target_inst=inst, 
                                                           condition_instruments=target_order[:i])
      
      if self.use_offset:
        beat_tokens = self.tokenizer.decode(outputs_tensor[:, 0])
        new_gen_str = self.beat_to_gen_converter(beat_tokens)
        gen_str = new_gen_str + ' \n\n ' + gen_str
      elif self.is_abc:
        decoded_str = self.jg_decoder(outputs_tensor)
        gen_str = decoded_str[:,0] + ' \n\n ' + gen_str
      else:
        gen_str = ' '.join(self.tokenizer.decode(outputs_tensor[:,0])) + ' \n\n ' + gen_str
    return gen_str
    

  def inference_from_xml(self, score_fn, target_order:List[str], part_idx=-1):
    '''
    target_order: list of instrument in order of inference. Has to include first given instrument.
    '''
    gen_str = self.convert_xml_to_gen_code(score_fn, part_idx)
    gen_str = self.make_inst_sequential_inference(gen_str, target_order)
    return gen_str
  
  def inference_from_gen_code(self, text_fn, target_order:List[str]):
    with open(text_fn, 'r') as f:
      gen_str = f.read()
    gen_str = self.make_inst_sequential_inference(gen_str, target_order)
    return gen_str
  
  def cycle_inference_from_gen_code(self, text_fn, inst_cycles:List[str]=['piri', 'geomungo', 'gayageum', 'ajaeng', 'haegeum', 'daegeum'], num_cycles:int=6):
    with open(text_fn, 'r') as f:
      gen_str = f.read()
      
    for i in range(num_cycles):
      gen_str = '\n\n'.join(gen_str.split('\n\n')[:-1])
      inst_list = list(reversed(inst_cycles[1:]))
      target_inst = inst_cycles[0]
      print(inst_list, target_inst)
      dataset = self.prepare_dataset(gen_str, inst_list=inst_list) # reverse order
      outputs_tensor = self.make_full_inference_on_dataset(dataset, 
                                                            target_inst=target_inst, 
                                                            condition_instruments=inst_list)
      gen_str = ' '.join(self.tokenizer.decode(outputs_tensor[:,0])) + ' \n\n ' + gen_str
      new_cycles = copy.copy(inst_cycles[1:]) + copy.copy([inst_cycles[0]])
      inst_cycles = new_cycles
    
    return gen_str

  @staticmethod
  def get_measure_specific_output(output:torch.LongTensor, measure_idx_in_token:int):
    corresp_ids = torch.where(output[:,-2] == measure_idx_in_token)[0] + 1
    if corresp_ids[-1] >= len(output):
      print(f"Warning: next measure of {measure_idx_in_token} is not in the output.")
      corresp_ids[-1] = len(output) - 1
    return output[corresp_ids]

  @staticmethod
  def get_measure_shifted_output(output: torch.Tensor, measure_idx_start:int, measure_idx_end:int, initial_prev_pos:int):
    for i, token in enumerate(output):
      if token[-2] == measure_idx_start: # new measure started:
        break
    for j in range(i, len(output)):
      if output[j][-2] == measure_idx_end:
        break
    output_tokens_from_second_measure = output[i:j].clone()
    output_tokens_from_second_measure[0, 0] = 1 # start token의 인덱스
    output_tokens_from_second_measure[0, 1] = initial_prev_pos #beat 0의 인덱스
    output_tokens_from_second_measure[:, -2] -= 1 # gak_idx is -2
    return output_tokens_from_second_measure
  



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
  
  # score_fn = 'gen_results/CHP_scoreCPH_from_1_0308-1209.musicxml'
  # output_str = gen.inference_from_xml(score_fn, ['geomungo', 'gayageum', 'ajaeng', 'haegeum', 'piri', 'daegeum'])

  # with open('gen_results/CHP_scoreCPH_from_1_0308-1209_gencode.txt', 'w') as f:
  #   f.write(output_str)
  
  # jg_omr_str = gen.jg_to_omr_converter.convert_multi_inst_str(output_str)
  # with open('gen_results/CHP_scoreCPH_from_1_0308-1209_omr.txt', 'w') as f:
  #   f.write(jg_omr_str)
  
  # notes, score = gen.jg_to_staff_converter(output_str)

  # score.write('musicxml', 'gen_results/CHP_scoreCPH_from_1_0308-1209_write.musicxml')
  