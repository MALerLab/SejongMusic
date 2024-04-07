import argparse
import datetime
import copy
from pathlib import Path
from omegaconf import OmegaConf
from fractions import Fraction
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Any, Union

import torch
import music21
from music21 import converter, stream, note as m21_note 
import numpy as np

from . import model_zoo, jg_code, inference
from .yeominrak_processing import OrchestraScoreSeq, ShiftedAlignedScore, Tokenizer
from .model_zoo import JeongganTransSeq2seq, JeongganBERT
from .decode import MidiDecoder, OrchestraDecoder
from .inference_utils import prepare_input_for_next_part, get_measure_specific_output, fix_measure_idx, fill_in_source, get_measure_shifted_output, recover_beat, round_number
from .inference import JGInferencer
from .jg_to_staff_converter import JGToStaffConverter, JGCodeToOMRDecoder
from .jg_code import JeongganDataset, JeongganTokenizer, JeongganPiece
from .jeonggan_utils import JGConverter, GencodeConverter, RollToJGConverter
from .mlm_utils import Augmentor
from .era_align_utils import AlignPredictor
from .tokenizer import SingleVocabTokenizer

class EraTransformer:
  roll2omr = RollToJGConverter()
  omr2gen = GencodeConverter
  gen2staff = JGToStaffConverter()

  def __init__(self, state_dir:str, device='cuda'):
    self.state_dir = Path(state_dir)
    self.model, self.tokenizer, self.config = self.load_model_tokenizer_from_state_dir(state_dir)
    self.model.to(device)
    self.device = device
    
    self.mask_id = self.tokenizer.tok2idx['mask']
    self.sus_id = self.tokenizer.tok2idx['-']
    self.non_orn_id = self.tokenizer.tok2idx['비어있음']
    self.masker = EraTransformMasker(self.mask_id, self.sus_id)
    
    
  @staticmethod
  def load_model_tokenizer_from_state_dir(state_dir:str):
    state_dir = Path(state_dir)
    config_fn = state_dir / 'config.yaml'
    json_fn = state_dir / 'tokenizer_vocab.json'
    ckpt_fn = state_dir / 'inst_0' / 'iter46400_model.pt'

    config = OmegaConf.load(config_fn)
    tokenizer = JeongganTokenizer(None, None, is_roll=True, json_fn=json_fn)
    model = JeongganBERT(tokenizer=tokenizer, config=config.model)
    model.load_state_dict(torch.load(ckpt_fn, map_location='cpu'))
    model.eval()
    return model, tokenizer, config
  
  def convert_to_input_format(self, roll_array:np.ndarray):
    assert roll_array.ndim == 2
    roll_list = roll_array.tolist()
    roll_list = [ ['start'] * len(roll_list[0])] + roll_list + [ ['end'] * len(roll_list[0])]
    tokens = self.tokenizer(roll_list)
    return torch.LongTensor(tokens)

  @staticmethod
  def aug_form_to_input_form(x:torch.Tensor):
    assert x.ndim == 3
    return x.permute(2,0,1).flatten(0,1)
  
  def unmask_tokens(self,
                    model:JeongganBERT, 
                    x:torch.Tensor, 
                    mask:torch.Tensor, 
                    target_id:int, 
                    n:int,
                    onset_ratio:float=None):
    mask = mask[:, target_id]
    mask_remaining = mask.sum()

    if onset_ratio:
      non_onset_token = self.sus_id if target_id == 0 else self.non_orn_id
      num_initial_onsets =  sum( (x[:, target_id] != non_onset_token) * (x[:, target_id] != self.mask_id))
    while mask_remaining > 0:
      pred, _ = model(x.unsqueeze(0))
      pred = pred[0] # (T, C)
      pitch_pred = pred[:, target_id]

      is_masked = mask==1
      pitch_masked_idx = torch.where(is_masked)[0]
      
      masked_pitch_prob = pitch_pred[is_masked].softmax(dim=-1)
      high_prob_idx = masked_pitch_prob.max(dim=-1).values.argsort(descending=True)[:n]
      high_prob_pitches = masked_pitch_prob[high_prob_idx].argmax(dim=-1)
      
      selected_idx = pitch_masked_idx[high_prob_idx]
      x[selected_idx, target_id] = high_prob_pitches 
      mask[selected_idx] = 0
      mask_remaining = mask.sum()
      
      if onset_ratio:
        num_onsets =  sum( (x[:, target_id] != non_onset_token)* (x[:, target_id] != self.mask_id))
        if num_onsets > num_initial_onsets * onset_ratio:
          print(f"Early stopping at {num_onsets} onsets")
          x[:,0][x[:,0] == self.mask_id] = self.sus_id
          return x
        
    return x

  def decode_result(self, x:torch.Tensor):
    roll = self.tokenizer.decode(x)
    omr_str = self.roll2omr(roll)
    gen_str = self.omr2gen.convert_lines_to_gencode(omr_str.split('\n')) + ' \n'
    notes, score = self.gen2staff(gen_str, time_signatures='30/8')
    return notes, score
    
  
  @torch.inference_mode()
  def unmask_pitch_and_ornaments(self, x, loss_mask, n, onset_ratio):
    x = self.unmask_tokens(self.model, x, loss_mask, 0, n, onset_ratio)
    x = self.unmask_tokens(self.model, x, loss_mask, 1, n)
    return x
  
  def infer_on_gen_str(self, gen_str:str, inst:str, idx:int=0, onset_ratio=None):
    piece = JeongganPiece(None, gen_str=gen_str, inst_list=[inst])
    x = piece.convert_tokens_to_roll(piece.sliced_parts_by_measure[idx][inst], inst)
    x = self.convert_to_input_format(x)
    x, loss_mask = self.masker.mask_except_note_onset(x)
    x, loss_mask = self.aug_form_to_input_form(x), self.aug_form_to_input_form(loss_mask)
    x = self.unmask_pitch_and_ornaments(x.to(self.device), loss_mask.to(self.device), 3, onset_ratio=onset_ratio)
    notes, score = self.decode_result(x)
    
    org_token = piece.sliced_parts_by_measure[idx][inst]
    org_notes, org_score = self.gen2staff(org_token)

    total_score = music21.stream.Score()
    total_score.insert(0, score)
    total_score.insert(0, org_score)

    return (notes, org_notes), total_score


class EraTransformMasker:
  num_beat = 6
  def __init__(self, mask_id:int, sus_id:int):
    self.mask_id = mask_id
    self.sus_id = sus_id
    model_path = '/home/teo/userdata/SejongMusic/models/align_predictor.pt'
    vocab_path = '/home/teo/userdata/SejongMusic/models/align_predictor_vocab.json'  
    self.tokenizer = SingleVocabTokenizer(None, json_fn=vocab_path)
    self.align_model = AlignPredictor(len(self.tokenizer.vocab), 64)
    self.align_model.load_state_dict(torch.load(model_path))
    self.align_model.eval()
    self.update_vocab()
  
  
  def update_vocab(self):
    new_vocab = []
    for word in self.tokenizer.vocab:
      if 'pitch' not in word:
        new_vocab.append(word)
      elif word == 'pitch0':
        new_vocab.append(word)
      else:
        new_vocab.append(f'pitch{int(word[5:])+12}')
    self.tokenizer.vocab = new_vocab
    self.tokenizer.tok2idx = {tok:idx for idx, tok in enumerate(new_vocab)}
  
  def mask_except_note_onset(self, x:torch.Tensor):
    if x.ndim == 2:
      x = x.unsqueeze(-1)
    assert x.ndim == 3 # (num_frame, num_feature, num_inst)
    loss_mask = torch.ones_like(x, dtype=torch.long)
    loss_mask[:, 2:] = 0
    masked = x.clone()
    if x.shape[0] % self.num_beat == 0:
      masked[:, 0:2] = self.mask_id
      loss_mask[:, 0:2] = 1
    elif (x.shape[0]-2) % self.num_beat == 0: # start and end token appended
      masked[1:-1, 0:2] = self.mask_id
      loss_mask[1:-1, 0:2] = 1
      loss_mask[0] = 0
      loss_mask[-1] = 0

    frame_ids, inst_ids = torch.where(x[:,0,:] != self.sus_id)
    note_pair = (frame_ids, torch.zeros_like(frame_ids), inst_ids)
    masked[note_pair] = x[note_pair]
    loss_mask[note_pair] = 0
    return masked, loss_mask
  