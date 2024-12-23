import copy
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from fractions import Fraction
from music21 import stream
from tqdm.auto import tqdm
from typing import List, Set, Dict, Tuple, Union

from sejong_music import yeominrak_processing, model_zoo
from sejong_music.model_zoo import Seq2seq, QkvAttnSeq2seqOrch, get_emb_total_size, TransSeq2seq, JeongganTransSeq2seq
from sejong_music.utils import convert_note_to_sampling, convert_onset_to_sustain_token, make_offset_set, pack_collate
from sejong_music.sampling_utils import nucleus
from sejong_music.constants import get_dynamic_template_for_orch, get_dynamic, POSITION, DURATION, PITCH
from sejong_music.jg_to_staff_converter import ABCtoGenConverter
from sejong_music.abc_utils import convert_beat_jg_to_gen

class Inferencer:
  def __init__(self, 
               model: Seq2seq, 
               is_condition_shifted: bool, 
               is_orch: bool, 
               is_sep: bool, 
               temperature: float = 1.0,
               top_p: float = 1.0):
    self.model = model
    self.is_condition_shifted = is_condition_shifted
    self.is_orch = is_orch
    self.is_sep = is_sep
    self.converter = self.model.converter
    self.tokenizer = self.model.tokenizer
    self.vocab_size_dict = self.tokenizer.vocab_size_dict
    self.vocab_size = self.model.vocab_size
    if self.is_orch:
      self.dynamic_template = get_dynamic_template_for_orch()
    self.temperature = temperature
    self.top_p = top_p
  
  
  @property
  def device(self):
    return self.model.parameters().__next__().device
  
  def get_start_token(self, part_idx):
    num_features = len(self.vocab_size_dict) - 1
    # token_features = [self.tokenizer.tok2idx['index'][part_idx]]
    token_features = [part_idx]
    if 'pitch' in self.vocab_size_dict:
      token_features.append(self.tokenizer.tok2idx['pitch']['start'])
    if 'duration' in self.vocab_size_dict:
      token_features.append(self.tokenizer.tok2idx['duration']['start'])
    
    if self.is_condition_shifted:
      if 'offset' in self.vocab_size_dict:
        token_features.append(self.tokenizer.tok2idx['offset'][0]) 
      if 'daegang_offset' in self.vocab_size_dict:
        token_features.append(self.tokenizer.tok2idx['daegang_offset'][0])
      if 'jeonggan_offset' in self.vocab_size_dict:
        token_features.append(self.tokenizer.tok2idx['jeonggan_offset'][0])
      if 'beat_offset' in self.vocab_size_dict:
        token_features.append(self.tokenizer.tok2idx['beat_offset'][0])  
      if 'dynamic' in self.vocab_size_dict:
        token_features.append(self.tokenizer.tok2idx['dynamic']['strong'])
      if 'measure_change' in self.vocab_size_dict:
        token_features.append(self.tokenizer.tok2idx['measure_change'][1])
      if 'measure_idx' in self.vocab_size_dict:
        token_features.append(self.tokenizer.tok2idx['measure_idx'][0])      
      return torch.LongTensor([token_features])
    
    if 'offset' in self.vocab_size_dict:
      token_features.append(self.tokenizer.tok2idx['offset']['start'])
    if 'daegang_offset' in self.vocab_size_dict:
        token_features.append(self.tokenizer.tok2idx['daegang_offset']['start'])
    if 'jeonggan_offset' in self.vocab_size_dict:
        token_features.append(self.tokenizer.tok2idx['jeonggan_offset']['start'])
    if 'beat_offset' in self.vocab_size_dict:
        token_features.append(self.tokenizer.tok2idx['beat_offset']['start'])  
    if 'dynamic' in self.vocab_size_dict:
      token_features.append(self.tokenizer.tok2idx['dynamic']['start'])
    if 'measure_change' in self.vocab_size_dict:
      token_features.append(self.tokenizer.tok2idx['measure_change']['start'])
    if 'measure_idx' in self.vocab_size_dict:
      token_features.append(self.tokenizer.tok2idx['measure_idx']['start'])
    return torch.LongTensor([token_features])
    # return torch.LongTensor([[part_idx] + [1] * num_features])

  def _get_measure_duration(self, part_idx):
    if self.is_orch:
      return 30.0
    return self.tokenizer.measure_duration[part_idx]
  # def inference(self, sample: torch.Tensor, part_idx: int):
  
  def encode_condition_token(self, part_idx, current_beat, current_measure_idx, measure_start):
    condition_features = []
    if 'offset' in self.tokenizer.tok2idx:
      if self.is_orch:
        measure_offset_idx = self.tokenizer.tok2idx['offset'][float(current_beat)]
        condition_features.append(measure_offset_idx)
      else:
        if isinstance(current_beat, float):
          current_beat = Fraction(current_beat).limit_denominator(3)
        elif isinstance(current_beat, Fraction):
          current_beat = current_beat.limit_denominator(3)
        if isinstance(current_beat, Fraction) and current_beat.denominator == 3:
          measure_offset_idx = self.tokenizer.tok2idx['offset_fraction'][current_beat]
        else:
          try:
            measure_offset_idx = self.tokenizer.tok2idx['offset'][float(current_beat)]
          except Exception as e:
            print(f"Error Occured in current beat: {e}")
            measure_offset_idx = self.tokenizer.tok2idx['offset'][0.0]
        condition_features.append(measure_offset_idx)
        
    if 'daegang_offset' in self.tokenizer.tok2idx:
      daegang, jeonggan, beat = make_offset_set(current_beat)
      daegang_offset = self.tokenizer.tok2idx['daegang_offset'][daegang]
      jeonggan_offset = self.tokenizer.tok2idx['jeonggan_offset'][jeonggan]
      beat_offset = self.tokenizer.tok2idx['beat_offset'][beat]
      condition_features += [daegang_offset, jeonggan_offset, beat_offset]
    
    elif 'jeonggan_offset' in self.tokenizer.tok2idx:
      jeonggan_value = current_beat // 1
      jeonggan_offset = self.tokenizer.tok2idx['jeonggan_offset'][jeonggan_value]
      beat_offset_value = Fraction(current_beat % 1).limit_denominator(3)
      beat_offset = self.tokenizer.tok2idx['beat_offset'][beat_offset_value]
      condition_features += [jeonggan_offset, beat_offset]
      
    if 'dynamic' in self.tokenizer.tok2idx:
      if self.is_orch:
        dynamic = self.dynamic_template.get(float(current_beat), 'none')
        dynamic_idx = self.tokenizer.tok2idx['dynamic'][dynamic]
        condition_features.append(dynamic_idx)
      else:
        dynamic = get_dynamic(current_beat, part_idx)
        dynamic_idx = self.tokenizer.tok2idx['dynamic'][dynamic]
        condition_features.append(dynamic_idx)
        
    if 'measure_idx' in self.tokenizer.tok2idx:
      measure_idx = self.tokenizer.tok2idx['measure_idx'][current_measure_idx]
      condition_features.append(measure_idx)
    return condition_features
  
  def _apply_softmax(self, logit):
    pitch_output = torch.softmax(logit[...,:self.vocab_size[1]], dim=-1)
    dur_output = torch.softmax(logit[...,self.vocab_size[1] :], dim=-1)
    return torch.cat([pitch_output, dur_output], dim=-1) 

  def sampling_process(self, logit):
    logit = logit / self.temperature
    prob = self._apply_softmax(logit)
    selected_token = self._select_token(prob)
    return selected_token
    
  def _select_token(self, prob: torch.Tensor):
    tokens = []
    # print(prob.shape)
    pitch_prob = prob[0, :, :self.vocab_size[1]]
    dur_prob = prob[0, :, self.vocab_size[1]:]
    if self.top_p != 1.0:
      pitch_token = nucleus(pitch_prob, self.top_p)
      dur_token = nucleus(dur_prob, self.top_p)
    else:
      pitch_token = pitch_prob.multinomial(num_samples=1)
      dur_token = dur_prob.multinomial(num_samples=1)
    # pitch_token = torch.argmax(prob[0, :, :self.vocab_size[1]], dim=-1, keepdim=True)
    # dur_token = torch.argmax(prob[0, :, self.vocab_size[1]:], dim=-1, keepdim=True)
    return torch.cat([pitch_token, dur_token], dim=-1)
 
  def _handle_triplet(self, selected_token, current_dur, triplet_sum):
    if Fraction(current_dur).limit_denominator(3).denominator == 3: # limit_denominator는 분모를 3 아래로 제한, 근사를 해주는 것.
      triplet_sum += Fraction(current_dur).limit_denominator(3).numerator
    elif triplet_sum != 0:
      # print("Triplet has to appear but not", current_dur,)
      # print(current_dur, selected_token)
      current_dur = Fraction(1,3)
      selected_token = torch.LongTensor([[selected_token[0,0], self.tokenizer.tok2idx['duration'][current_dur]]]).to(self.model.device)
      triplet_sum += 1
      # print(current_dur, selected_token)   
    if triplet_sum % 3 == 0:
      triplet_sum = 0
    return selected_token, current_dur, triplet_sum  
  
  def update_condition_info(self, current_beat, current_measure_idx, measure_duration, current_dur):
    if current_dur in ('start', 'pad'):
      current_dur = 0.0
    if self.is_orch:
      current_beat += current_dur
      if float(current_beat) >= measure_duration:
        current_measure_idx += 1 #마디 수 늘려주기
        measure_changed = 1
        current_beat = current_beat - measure_duration
      else:
        measure_changed = 0
    else:  
      current_beat += Fraction(current_dur).limit_denominator(3)
      
      if float(current_beat) + 0.1 >= measure_duration:
        current_measure_idx += 1 #마디 수 늘려주기
        measure_changed = 1
        current_beat = current_beat - measure_duration
        if abs(current_beat) < 0.01:
          current_beat = 0
      else:
        measure_changed = 0

    return current_beat, current_measure_idx, measure_changed  
  
  def _decode_inference_result(self, src, output, other_out):
    if hasattr(self.model, 'enc_converter'):
      return self.model.enc_converter(src[1:-1]), self.converter(output), other_out
    src_decoded = self.converter(src[1:-1])
    out_decoded = self.converter(output)
    if sum(type(y[2])==str for y in out_decoded) > 0:
      print('Warning: There is a string in the output!')
      print(out_decoded)
    return src_decoded, out_decoded, other_out

  @torch.inference_mode()
  def inference(self, src, part_idx, prev_generation=None, fix_first_beat=False, compensate_beat=(0.0, 0.0)):
    dev = self.device
    src = src.to(dev)

    # Setup for 0th step
    # start_token = torch.LongTensor([[part_idx, 1, 1, 3, 3, 4]]) # start token idx is 1
    start_token = self.get_start_token(part_idx).to(dev)
    assert src.ndim == 2 # sequence length, feature length
    measure_duration = self._get_measure_duration(part_idx)
    current_measure_idx = 0
    measure_changed = 1
    
    encoder_output: dict = self.model.run_encoder(src)

    final_tokens = [start_token] if isinstance(self.model, TransSeq2seq) else []
    if prev_generation is not None:
      final_tokens, current_measure_idx, encoder_output = self.model._apply_prev_generation(prev_generation, final_tokens, encoder_output)
    selected_token = start_token
    
    current_beat = 0.0 if self.is_orch else Fraction(0, 1)
    triplet_sum = 0 

    total_attention_weights = []
    # while True:
    condition_tokens = self.encode_condition_token(part_idx, current_beat, current_measure_idx, measure_changed)

    for i in range(300):
      input_token = torch.cat(final_tokens, dim=0) if isinstance(self.model, TransSeq2seq) else selected_token
      logit, encoder_output, attention_weight = self.model._run_inference_on_step(input_token, encoder_output)
      selected_token = self.sampling_process(logit)
      if fix_first_beat and current_beat - compensate_beat[1] == 0.0:
        for src_token in src:
          if src_token[-1] == condition_tokens[-1] and self.tokenizer.vocab['offset'][src_token[3].item()] - compensate_beat[0] == 0.0:
            # if measure_idx is same and offset is same
            selected_token[0, 0] = src_token[1] # get pitch from src
            break
      
      if selected_token[0, 0]==self.tokenizer.tok2idx['pitch']['end'] or selected_token[0, 1]==self.tokenizer.tok2idx['duration']['end']:
        break
      
      # update beat position and beat strength
      current_dur = self.tokenizer.vocab['duration'][selected_token[0, 1]]
      if not self.is_orch:  
        selected_token, current_dur, triplet_sum = self._handle_triplet(selected_token, current_dur, triplet_sum)
        
        
      current_beat, current_measure_idx, measure_changed = self.update_condition_info(current_beat, current_measure_idx, measure_duration, current_dur)
      if 'measure_idx' in self.tokenizer.tok2idx and current_measure_idx > self.tokenizer.vocab['measure_idx'][-1]:
        break
      if float(current_beat) == 10.0 and not self.is_orch:
        print(f"Current Beat ==10.0 detected! cur_beat: {current_beat}, measure_dur: {measure_duration}, cur_measure_idx: {current_measure_idx}, cur_dur: {current_dur}, triplet_sum: {triplet_sum}")
      condition_tokens = self.encode_condition_token(part_idx, current_beat, current_measure_idx, measure_changed)
      
      # make new token for next rnn timestep
      selected_token = torch.cat([torch.LongTensor([[part_idx]]).to(dev), selected_token, torch.LongTensor([condition_tokens]).to(dev)], dim=1)
      
      final_tokens.append(selected_token)
      total_attention_weights.append(attention_weight[0,:,0])
    if len(total_attention_weights) == 0:
      attention_map = None
    else:
      attention_map = torch.stack(total_attention_weights, dim=1)
    if isinstance(self.model, TransSeq2seq): final_tokens = final_tokens[1:]
    cat_out = torch.cat(final_tokens, dim=0)
    if isinstance(self.model, TransSeq2seq) and prev_generation is not None:
      newly_generated = cat_out[len(prev_generation):]
    else:
      newly_generated = cat_out.clone()
    if prev_generation is not None and not isinstance(self.model, TransSeq2seq):
      cat_out = torch.cat([prev_generation[1:], cat_out], dim=0)
    #  self.converter(src[1:-1]), self.converter(torch.cat(final_tokens, dim=0)), attention_map

    return self._decode_inference_result(src, cat_out, (attention_map, cat_out, newly_generated))
    


class JGInferencer(Inferencer):
  def __init__(self, model: JeongganTransSeq2seq, 
               is_condition_shifted: bool, 
               is_orch: bool, 
               temperature: float = 1, 
               top_p: float = 1,
               is_abc: bool = False):
    super().__init__(model, is_condition_shifted, is_orch, is_sep=False, temperature=temperature, top_p=top_p)
    self.use_offset = 'beat:0' in self.tokenizer.vocab
    if self.use_offset:
      self.beat2gen = convert_beat_jg_to_gen
    
  def get_start_token(self, inst:str):
    return torch.LongTensor(self.tokenizer(['start', 'prev|', 'jg:0', 'gak:0', inst])).unsqueeze(0)
  
  
  def encode_condition_token(self, prev_pos_token, current_jg_idx, current_gak_idx, inst_name):
    return self.tokenizer([f'prev{prev_pos_token}', f'jg:{current_jg_idx}', f'gak:{current_gak_idx}', inst_name])
    
  def _decode_inference_result(self, src, output, other_out):
    src_decoded = self.tokenizer.decode(src[1:-1])
    out_decoded = self.tokenizer.decode(output)
    if self.use_offset:
      converted = self.beat2gen([x[0] for x in out_decoded]) + ['\n']
      converted = [x for x in converted if x != ''] 
      shorter_len = min(len(converted), len(out_decoded))
      out_decoded = [[converted[i]] + out_decoded[i][1:] for i in range(shorter_len)]
    return src_decoded, out_decoded, other_out
  
  def _apply_softmax(self, logit):
    prob = torch.softmax(logit, dim=-1)
    return prob
  
  def _select_token(self, prob: torch.Tensor):
    if prob.ndim == 3:
      prob = prob[:, 0]
    return nucleus(prob, self.top_p) if self.top_p != 1.0 else prob.multinomial(num_samples=1)
  
  @torch.inference_mode()
  def inference(self, src, inst_name:str, prev_generation=None, fix_first_beat=False, compensate_beat=(0.0, 0.0)):
    dev = self.device
    src = src.to(dev)
        
    # Setup for 0th step
    # start_token = torch.LongTensor([[part_idx, 1, 1, 3, 3, 4]]) # start token idx is 1
    start_token = self.get_start_token(inst_name).to(dev)
    assert src.ndim == 2 # sequence length, feature length
    current_gak_idx = 0
    current_jg_idx = 0
    prev_pos_token = '|'
    
    encoder_output: dict = self.model.run_encoder(src)

    final_tokens = [start_token]
    if prev_generation is not None:
      final_tokens, current_gak_idx, encoder_output = self.model._apply_prev_generation(prev_generation, final_tokens, encoder_output)
    selected_token = start_token

    total_attention_weights = []
    # while True:
    condition_tokens = self.encode_condition_token(prev_pos_token, current_jg_idx, current_gak_idx, inst_name)

    for i in range(2000):
      input_token = torch.cat(final_tokens, dim=0) if isinstance(self.model, JeongganTransSeq2seq) else selected_token
      logit, encoder_output, attention_weight = self.model._run_inference_on_step(input_token, encoder_output)
      selected_token = self.sampling_process(logit)
      decoded_token = self.tokenizer.vocab[selected_token[0]]
      
      if decoded_token == 'end': break
      
      if decoded_token in self.tokenizer.pos_vocab:
        prev_pos_token = decoded_token

      if decoded_token == '|':
        current_jg_idx += 1

      if decoded_token == '\n':
        current_gak_idx += 1
        current_jg_idx = 0
      
      if f'jg:{current_jg_idx}' not in self.tokenizer.vocab or f'gak:{current_gak_idx}' not in self.tokenizer.vocab:
        break
        
      condition_tokens = self.encode_condition_token(prev_pos_token, current_jg_idx, current_gak_idx, inst_name)
      
      # make new token for next rnn timestep
      selected_token = torch.cat([selected_token, torch.LongTensor([condition_tokens]).to(dev)], dim=1)
      
      final_tokens.append(selected_token)
      total_attention_weights.append(attention_weight[0,:,0])
    if len(total_attention_weights) == 0:
      attention_map = None
    else:
      attention_map = torch.stack(total_attention_weights, dim=1)
    if isinstance(self.model, JeongganTransSeq2seq): final_tokens = final_tokens[1:]
    cat_out = torch.cat(final_tokens, dim=0)
    if isinstance(self.model, JeongganTransSeq2seq) and prev_generation is not None:
      newly_generated = cat_out[len(prev_generation):]
    else:
      newly_generated = cat_out.clone()
    if prev_generation is not None and not isinstance(self.model, JeongganTransSeq2seq):
      cat_out = torch.cat([prev_generation[1:], cat_out], dim=0)
    #  self.converter(src[1:-1]), self.converter(torch.cat(final_tokens, dim=0)), attention_map

    return self._decode_inference_result(src, cat_out, (attention_map, cat_out, newly_generated))
    

class ABCInferencer(JGInferencer):
  def __init__(self, model: JeongganTransSeq2seq, 
              is_condition_shifted: bool, 
              is_orch: bool, 
              temperature: float = 1, 
              top_p: float = 1):
    super().__init__(model, is_condition_shifted, is_orch, temperature=temperature, top_p=top_p)
    self.jg_decoder = ABCtoGenConverter()
    
  def get_start_token(self, inst:str):
    return torch.LongTensor(self.tokenizer(['start', 'beat:0', 'jg:0', 'gak:0', inst])).unsqueeze(0)
  
  def inference(self, src, inst_name:str, prev_generation=None, fix_first_beat=False, compensate_beat=(0.0, 0.0)):
  
    dev = self.device
    src = src.to(dev)
    decoded_src = self.tokenizer.decode(src[1:-1])
    first_measure_duration, second_measure_duration = self.calculate_measure_duration(decoded_src)
    
    # Setup for 0th step
    # start_token = torch.LongTensor([[part_idx, 1, 1, 3, 3, 4]]) # start token idx is 1
    start_token = self.get_start_token(inst_name).to(dev)
    assert src.ndim == 2 # sequence length, feature length
    current_gak_idx = 0
    current_jg_idx = 0
    prev_pos_token = 'beat:0'
    
    current_beat = 0.0 if self.is_orch else Fraction(0, 1)
    encoder_output: dict = self.model.run_encoder(src)

    final_tokens = [start_token]
    if prev_generation is not None:
      final_tokens, current_gak_idx, encoder_output = self.model._apply_prev_generation(prev_generation, final_tokens, encoder_output, is_abc=True)
    selected_token = start_token

    total_attention_weights = []
    # while True:
    condition_tokens = self.encode_condition_token(0.0, current_gak_idx, inst_name)
    first_gak = True

    for i in tqdm(range(3000), leave=False):
      input_token = torch.cat(final_tokens, dim=0) if isinstance(self.model, JeongganTransSeq2seq) else selected_token
      logit, encoder_output, attention_weight = self.model._run_inference_on_step(input_token, encoder_output)
      selected_token = self.sampling_process(logit)
      decoded_token = self.tokenizer.vocab[selected_token[0]]
      
      measure_duration = first_measure_duration if first_gak else second_measure_duration
      
      if decoded_token == 'end': break

      if current_beat >= measure_duration:
          decoded_token = '\n'
          selected_token = torch.LongTensor([[self.tokenizer.tok2idx['\n']]]).to(dev)
          
      if decoded_token == '\n':
        current_gak_idx += 1
        current_jg_idx = 0
        current_beat = 0.0
        first_gak = False
          
      if decoded_token in self.tokenizer.dur_vocab:
        current_beat += decoded_token
        # if first_gak:
        #   if current_beat >= first_measure_duration:
        #     current_jg_idx += 1
        #     current_beat = 0.0
        # elif current_beat >= second_measure_duration:
        #   current_gak_idx += 1
        #   current_jg_idx = 0
        #   current_beat = 0.0
      # print(decoded_token, current_beat, current_gak_idx)
      if f'jg:{str(int(current_beat//1))}' not in self.tokenizer.vocab or f'gak:{current_gak_idx}' not in self.tokenizer.vocab:
        break
        
      condition_tokens = self.encode_condition_token(current_beat, current_gak_idx, inst_name)
      
      # make new token for next rnn timestep
      selected_token = torch.cat([selected_token, torch.LongTensor([condition_tokens]).to(dev)], dim=1)
      
      final_tokens.append(selected_token)
      total_attention_weights.append(attention_weight[0,:,0])
    if len(total_attention_weights) == 0:
      attention_map = None
    else:
      attention_map = torch.stack(total_attention_weights, dim=1)
    if isinstance(self.model, JeongganTransSeq2seq): final_tokens = final_tokens[1:]
    cat_out = torch.cat(final_tokens, dim=0)
    if isinstance(self.model, JeongganTransSeq2seq) and prev_generation is not None:
      newly_generated = cat_out[len(prev_generation):]
    else:
      newly_generated = cat_out.clone()
    if prev_generation is not None and not isinstance(self.model, JeongganTransSeq2seq):
      cat_out = torch.cat([prev_generation[1:], cat_out], dim=0)
    #  self.converter(src[1:-1]), self.converter(torch.cat(final_tokens, dim=0)), attention_map

    return self._decode_inference_result(src, cat_out, (attention_map, cat_out, newly_generated))

  def encode_condition_token(self, current_beat, current_gak_idx, inst_name):
    current_beat = round(current_beat, 5)
    jg_offset = int(current_beat // 1)
    beat_offset = Fraction(current_beat % 1).limit_denominator(6)
    return self.tokenizer([f'beat:{beat_offset}', f'jg:{jg_offset}', f'gak:{current_gak_idx}', inst_name])

  @staticmethod
  def calculate_measure_duration(decoded_src:List[list]):
    first_duration_list = []
    second_duration_list = []
    stand = True
    for note in decoded_src:
      if note[0] == '\n' and not stand:
        break
      if note[0] == '\n':
        stand = False
      if note[0] in DURATION and stand:
        first_duration_list.append(Fraction(note[0]))
      if note[0] in DURATION and not stand:
        second_duration_list.append(Fraction(note[0]))
    return sum(first_duration_list), sum(second_duration_list)
  
  def _decode_inference_result(self, src, output, other_out):

    src_decoded = [[token[0], token[-1]] for token in self.tokenizer.decode(src[1:-1])] #tokens = [x[0] for x in output]
    out_decoded = [[token[0], token[-1]] for token in self.tokenizer.decode(output)]
    src_decoded = self._split_by_inst_to_decode(src_decoded)
    out_decoded = self._split_by_inst_to_decode(out_decoded)

    return src_decoded, out_decoded, other_out
  
  def _split_by_inst_to_decode(self, tokens:List):
    inst_list = []
    for token in tokens:
      if token[1] not in inst_list:
        inst_list.append(token[1])
    all_tokens = []
    for inst in inst_list:
      inst_tokens = [token[0] for token in tokens if token[1] == inst]
      decoded = self.jg_decoder(inst_tokens).split(' ') 
      decoded = [[note, inst] for note in decoded]
      all_tokens += decoded
    return all_tokens    
  

def sequential_inference(model:Seq2seq, sample:torch.Tensor, decoder):
  s = stream.Score(id='mainScore')
  
  # output_part_idx = 1
  for i in range(7):
    if model.is_condition_shifted:
      src, output, attention_map = model.shifted_inference(sample, i+1)
    else:
      src, output, attention_map = model.inference(sample, i+1)    
    if i == 0:
      merged_part = stream.Part() 
      src_midi = decoder(src)
      for element in src_midi:
        merged_part.append(element)
      s.insert(0, merged_part)
    output_midi = decoder(output)
    merged_part = stream.Part() 
    for element in output_midi:
      merged_part.append(element)
    s.insert(0, merged_part)

    if model.is_condition_shifted:
      pred, condition = [note[:3] for note in output], [note[3:] for note in output]
      initial_condition_token = []
      if 'offset' in model.tokenizer.key_types:
        initial_condition_token.append(0)
      if 'dynamic' in model.tokenizer.key_types:
        initial_condition_token.append('strong')
      if 'measure_idx' in model.tokenizer.key_types:
        initial_condition_token.append(0)
      if 'measure_changed' in model.tokenizer.key_types:
        initial_condition_token.append(1)

      output = [pred[0] + initial_condition_token] + [pr + con for pr, con in zip(pred[1:], condition[:-1])] + [[i+1] + ['end'] * (len(model.tokenizer.key_types)-1)]
      output = [[i+1] +  ['start'] * (len(model.tokenizer.key_types)-1)] + output
      sample = [model.tokenizer(note) for note in output]
    elif hasattr(model, 'sampling_rate'):
      converted_output = convert_note_to_sampling(output, model.dynamic_template_list, beat_sampling_num=model.sampling_rate)
      converted_output = convert_onset_to_sustain_token(converted_output)
      sample = [model.tokenizer(item) for item in converted_output]

    else:
      output = [[i+1] +  ['start'] * (len(model.tokenizer.key_types)-1)] + output + [[i+1] + ['end'] * (len(model.tokenizer.key_types)-1)]
      sample = [model.tokenizer(note) for note in output]

    sample = torch.Tensor(sample).long()

  return s

class JGSimpleInferencer(JGInferencer):
  def __init__(self, model: JeongganTransSeq2seq, 
              is_condition_shifted: bool, 
              is_orch: bool, 
              temperature: float = 1, 
              top_p: float = 1):
    super().__init__(model, is_condition_shifted, is_orch, temperature=temperature, top_p=top_p)
  
  def get_start_token(self, inst:str):
    return torch.LongTensor(self.tokenizer(['start', inst])).unsqueeze(0)


  @torch.inference_mode()
  def inference(self, src, inst_name:str, prev_generation=None, fix_first_beat=False, compensate_beat=(0.0, 0.0)):
    dev = self.device
    src = src.to(dev)


    # Setup for 0th step
    # start_token = torch.LongTensor([[part_idx, 1, 1, 3, 3, 4]]) # start token idx is 1
    start_token = self.get_start_token(inst_name).to(dev)
    assert src.ndim == 2 # sequence length, feature length

    encoder_output: dict = self.model.run_encoder(src)

    final_tokens = [start_token]
    if prev_generation is not None:
      final_tokens, current_gak_idx, encoder_output = self.model._apply_prev_generation(prev_generation, final_tokens, encoder_output)
    selected_token = start_token

    total_attention_weights = []
    # while True:

    for i in tqdm(range(500), leave=False):
      
      input_token = torch.cat(final_tokens, dim=0) if isinstance(self.model, JeongganTransSeq2seq) else selected_token
      # print(input_token)
      logit, encoder_output, attention_weight = self.model._run_inference_on_step(input_token, encoder_output)
      selected_token = self.sampling_process(logit)
      decoded_token = self.tokenizer.vocab[selected_token[0]]
      if decoded_token == 'end': break
      selected_token = torch.cat([selected_token, torch.LongTensor([self.tokenizer([inst_name])]).to(dev)], dim=1)
      # print(selected_token)
      final_tokens.append(selected_token)
      total_attention_weights.append(attention_weight[0,:,0])
      
    if len(total_attention_weights) == 0:
      attention_map = None
    else:
      attention_map = torch.stack(total_attention_weights, dim=1)
    if isinstance(self.model, JeongganTransSeq2seq): final_tokens = final_tokens[1:]
    cat_out = torch.cat(final_tokens, dim=0)
    if isinstance(self.model, JeongganTransSeq2seq) and prev_generation is not None:
      newly_generated = cat_out[len(prev_generation):]
    else:
      newly_generated = cat_out.clone()
    if prev_generation is not None and not isinstance(self.model, JeongganTransSeq2seq):
      cat_out = torch.cat([prev_generation[1:], cat_out], dim=0)
    #  self.converter(src[1:-1]), self.converter(torch.cat(final_tokens, dim=0)), attention_map

    return self._decode_inference_result(src, cat_out, (attention_map, cat_out, newly_generated))
  

if __name__ == "__main__":
  print("Inference Test!")
  config = OmegaConf.load("/home/danbi/userdata/DANBI/gugakwon/SejongMusic/outputs/2024-03-22/01-29-11/wandb/run-20240322_012912-3a7w91xw/files/checkpoints/config.yaml")
  dataset_class = getattr(yeominrak_processing, config.dataset_class)
  model_class = getattr(model_zoo, config.model_class)
  val_dataset = dataset_class(is_valid= True,
                              # valid_measure_num = [i for i in range(93, 99)],
                              xml_path='music_score/FullScore_edit5.musicxml', 
                              use_pitch_modification=False, 
                              slice_measure_num=config.data.max_meas,
                              min_meas=config.data.min_meas,
                              feature_types=copy.copy(config.model.features),
                              sampling_rate=config.data.sampling_rate,
                              target_instrument=config.data.target_instrument)
  valid_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=pack_collate, drop_last=True)
  
  device = 'cuda'
  encoder_tokenizer = val_dataset.tokenizer
  model = model_class(encoder_tokenizer, config.model).to(device)
  model.is_condition_shifted = isinstance(val_dataset, yeominrak_processing.ShiftedAlignedScore)
  model.state_dict(torch.load('/home/danbi/userdata/DANBI/gugakwon/SejongMusic/outputs/2024-03-22/01-29-11/wandb/run-20240322_012912-3a7w91xw/files/checkpoints/best_model.pt'))
  
  inferencer = Inferencer(model, model.is_condition_shifted, True, True)
  src = valid_loader.dataset[0][0].to(device)
  inferencer.shifted_inference(src, 1)
  
  