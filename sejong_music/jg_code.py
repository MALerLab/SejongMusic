
import glob
import random
import json
from pathlib import Path
from typing import List, Set, Dict, Tuple, Union
from collections import defaultdict, Counter
from fractions import Fraction
from pathlib import Path
from tqdm.auto import tqdm
import torch
import numpy as np

from .jg_to_staff_converter import JGToStaffConverter, Note
from .constants import POSITION, PITCH, PART, DURATION, BEAT, BEAT_POSITION
from .utils import  as_fraction, FractionEncoder
from .mlm_utils import Augmentor



class JeongganPiece:
  def __init__(self, txt_fn, gen_str=None, inst_list=None, use_offset=False):
    self.use_offset = use_offset
    if gen_str:
      assert inst_list, "inst_list should be provided"
      self.name = 'gen_str'
      self.text_str = gen_str
      self.inst_list = inst_list
    else:
      self.txt_fn = txt_fn
      self.name = txt_fn.split('/')[-1].split('_')[0]
      self.text_str = self.read_txt(txt_fn)
      self.inst_list = self.txt_fn[:-4].split('/')[-1].split('_')[1:]
    self.parts, self.part_dict = self.split_to_inst()
    self.check_measure_length()
    self.tokenized_parts = [self.split_and_filter(part) for part in self.parts]
    self.token_counter = self.count_token()
    self.sliced_parts_by_inst, self.sliced_parts_by_measure = self.prepare_sliced_measures()
    
  def split_to_inst(self):
    self.daegeum, self.piri, self.haegeum, self.ajaeng, self.gayageum, self.geomungo = None, None, None, None, None, None
    parts = self.text_str.split('\n\n')
    parts = [part + ' \n ' if '\n' not in part[-5:] else part for part in parts]
    for i, inst in enumerate(self.inst_list):
      setattr(self, inst, parts[i])
    return parts, {inst: part for inst, part in zip(self.inst_list, parts)}

  def count_token(self):
    token_counter = Counter()
    for part in self.tokenized_parts:
      token_counter.update(part)
    return token_counter
    
  def split_and_filter(self, string, split_charater=' '):
    splited = string.split(split_charater)
    filtered = [x for x in splited if x != '']
    filtered = self.filter_by_ignore_token(filtered)
    if self.use_offset: filtered = self.convert_token_to_abs_offset(filtered)
    return filtered
  
  @staticmethod
  def filter_by_ignore_token(splited, ignore_token=['10', '뜰', '2', '3', '흘림표', '추성', '퇴성', '숨표'], position_token=POSITION):
    position_token = [x for x in position_token if x not in ['|', '\n']]
    filtered = []
    prev_token = splited[-1]
    for token in splited[::-1]:
      if token in ignore_token:
        prev_token = token
        continue
      elif prev_token in ignore_token:
        if token == '-' or token in position_token:
          continue
        else: 
          prev_token = token
          filtered.append(token)
      else:
        prev_token = token
        filtered.append(token)
    filtered= filtered[::-1]
    return filtered
            
  
  @staticmethod
  def cut(part, measure_boundary_idx, start_mea=0, end_mea=4):
    start_idx = measure_boundary_idx[start_mea]
    end_idx = measure_boundary_idx[end_mea]
    return part[start_idx:end_idx]

  @classmethod
  def cut_part_by_slice_len(self, part:str, slice_len:int):
    measure_boundary_idx = [0]+ [i+1 for i, cha in enumerate(part) if cha == '\n']
    # measure_boundary_idx[10] == 10th measure's start index (count from 0)
    cutted_part = [self.cut(part, measure_boundary_idx, i, i+slice_len) for i in range(len(measure_boundary_idx)-slice_len)]
    return cutted_part
    
  def check_measure_length(self):
    self.is_clean = False
    meausre_length_by_inst = [[measure.count('|') for measure in inst.split('\n')if len(measure)>2] for inst in self.parts ]
    for sublist in meausre_length_by_inst[1:]:
      if sublist != meausre_length_by_inst[0]:
        return None, None, None # 우조 경풍년, 현악취타 제외!
        print(txt_name)
        print(num_lists[0], sublist)   
    self.is_clean = True
    meausre_length_of_first_instrument = meausre_length_by_inst[0]
    self.first_measure_len = meausre_length_of_first_instrument[0]
    self.middle_measure_len = meausre_length_of_first_instrument[1] 
    self.final_measure_len = meausre_length_of_first_instrument[-1]
    
  def prepare_sliced_measures(self, slice_len=4):
    if not self.is_clean:
      return None, None
    sliced_parts = {}
    for inst, part in zip(self.inst_list, self.tokenized_parts):
      cutted_part = self.cut_part_by_slice_len(part, slice_len)
      sliced_parts[inst] = cutted_part
    sliced_by_measure = []
    for i in range(len(cutted_part)):
      sliced_by_measure.append({inst: cutted_part[i] for inst, cutted_part in sliced_parts.items()})
    return sliced_parts, sliced_by_measure
  
  @staticmethod
  def convert_tokens_to_roll(tokens:List[str], inst:str, num_frame_per_jg=6, num_features=6)->np.ndarray:
    notes:List[Note] = JGToStaffConverter.convert_to_notes(tokens)
    JGToStaffConverter.get_duration_of_notes(notes)
    num_jgs = sum([1 for x in tokens if x in ('|', '\n')])

    outputs = np.zeros((num_jgs * num_frame_per_jg, num_features), dtype=object)
    outputs[:, 2] = [f'beat:{i}' for i in range(6)] * num_jgs
    outputs[:, 5] = inst
    num_jg_per_gak = JeongganPiece.get_measure_length(tokens)
    cur_idx = 0
    for i, num_jg in enumerate(num_jg_per_gak):
      outputs[cur_idx:cur_idx+num_jg*num_frame_per_jg, 3] = [f'jg:{i}' for i in np.arange(num_jg).repeat(num_frame_per_jg)]
      outputs[cur_idx:cur_idx+num_jg*num_frame_per_jg, 4] = f'gak:{i}'
      cur_idx += num_jg*num_frame_per_jg

    for note in notes:
      jg_offset, beat_offset, duration = note.global_jg_offset, note.beat_offset, note.duration
      if duration is None:
        duration = 0
      start_frame = round((jg_offset + beat_offset) * num_frame_per_jg)
      end_frame = max(round((jg_offset + beat_offset + duration) * num_frame_per_jg), start_frame+1)
      outputs[start_frame, 0] = note.pitch
      outputs[start_frame+1:end_frame, 0] = '-'
      if note.ornaments:
        outputs[start_frame, 1] = note.ornaments[0]
    #   outputs[start_frame:end_frame, 0] = note.pitch
    #   if note.ornaments:
    #     outputs[start_frame, 1] = note.ornaments[0]
    #   else:
    #     outputs[start_frame, 1] = '음표시작'
    outputs[outputs==0] = '비어있음'
    return outputs
  
  @staticmethod
  def convert_token_to_abs_offset(tokens:List[str]):
    notes:List[Note] = JGToStaffConverter.convert_to_notes(tokens)
    JGToStaffConverter._fix_three_col_division(notes)
    note_id = 0
    new_tokens = []

    for token in tokens:
      if token in POSITION[2:]: # if token is position, not | or \n
        new_token = f"beat:{str(Fraction(notes[note_id].beat_offset))}"
        new_tokens.append(new_token)
        note_id += 1
      else:
        new_tokens.append(token)
    return new_tokens

  
  def __len__(self):
    return len(self.parts[0].split('\n'))
  
  @staticmethod
  def get_measure_length(tokens:List[str]):
    gak_start_idx = [0] + [i for i, x in enumerate(tokens) if x == '\n']
    out = []
    for start, end in zip(gak_start_idx[:-1], gak_start_idx[1:]):
      out.append(sum([1 for x in tokens[start:end] if x == '|'])+1)

    return out

  @staticmethod
  def read_txt(path):
    with open(path, 'r') as file:
      txt = file.read()
      return txt

class ABCPiece(JeongganPiece):
  def __init__(self, txt_fn, gen_str=None, inst_list=None):
    super().__init__(txt_fn, gen_str, inst_list)
    self.processed_tokens = self.get_abc_notes()
    self.sliced_parts_by_inst, self.sliced_parts_by_measure = self.prepare_sliced_measure()
    
  def get_abc_notes(self):
    abc_inst_parts = [JGToStaffConverter.convert_to_notes(part) for part in self.tokenized_parts]
    for notes in abc_inst_parts:
      # JGToStaffConverter._fix_three_col_division(notes)
      JGToStaffConverter.get_duration_of_notes(notes) 
    featured_notes = [self.make_features(notes) for notes in abc_inst_parts]
    return featured_notes

  # def count_token(self):
  #   token_counter = Counter()
  #   for part in self.processed_tokens:
  #     tokens = [note[0] for note in part]
  #     token_counter.update(tokens)
  #   return token_counter
  
  def make_features(self, notes):
    total_tokens = []
    prev_gak = notes[0].gak_offset
    for note in notes:
      if note.pitch == None or note.duration == None: #뭔지 확인하기
        continue
      if note.gak_offset != prev_gak:
        num = note.gak_offset - prev_gak
        for i in range(num):
          total_tokens.append(['\n', f"beat:{note.beat_offset}", f"jg:{note.jg_offset}", f"gak:{note.gak_offset-num+i}"])
        prev_gak = note.gak_offset
      total_tokens.append([note.pitch, f"beat:{note.beat_offset}", f"jg:{note.jg_offset}", f"gak:{note.gak_offset}"])    
      if note.ornaments:
        for orn in note.ornaments:
          total_tokens.append([orn, f"beat:{note.beat_offset}", f"jg:{note.jg_offset}", f"gak:{note.gak_offset}"])
      total_tokens.append([note.duration, f"beat:{note.beat_offset}", f"jg:{note.jg_offset}", f"gak:{note.gak_offset}"])
    total_tokens.append(['\n', f"beat:{note.beat_offset}", f"jg:{note.jg_offset}", f"gak:{note.gak_offset}"])
    return total_tokens
  
  def prepare_sliced_measure(self, slice_len=4):
    if not self.is_clean:
      return None, None
    sliced_parts = {}
    for inst, part in zip(self.inst_list, self.processed_tokens):
      cutted_part = self.by_slice_len(part, slice_len)
      sliced_parts[inst] = cutted_part
    sliced_by_measure = []
    
    for i in range(len(cutted_part)):
      # try:
      sliced_by_measure.append({inst: cutted_part[i] for inst, cutted_part in sliced_parts.items()})
      # except:
      #   print(i, [len(part) for part in sliced_parts.values()])
    return sliced_parts, sliced_by_measure

  def by_slice_len(self, part:str, slice_len:int):
    measure_boundary_idx = [0]+ [i+1 for i, cha in enumerate(part) if cha[0] == '\n']
    # measure_boundary_idx[10] == 10th measure's start index (count from 0)
    cutted_part = [self.cut(part, measure_boundary_idx, i, i+slice_len) for i in range(len(measure_boundary_idx)-slice_len)]
    new_cutted_part = []
    for phrase in cutted_part:
      first_gak = int(phrase[0][-1][4:])
      new_phrase = []
      for note_features in phrase:
        new_offset = int(note_features[-1][4:]) - first_gak
        new_note = note_features[:-1] + [f"gak:{new_offset}"]
        new_phrase.append(new_note)
      new_cutted_part.append(new_phrase)
    return new_cutted_part

  # def replace_offset



class JeongganTokenizer:
    def __init__(self, special_token, feature_types=['index', 'token', 'position'], is_roll=False, json_fn=None, use_offset=False):
        if json_fn:
          self.load_from_json(json_fn)
          return      
        self.key_types = feature_types
        pos_tokens = BEAT_POSITION if use_offset else POSITION
        existing_tokens = pos_tokens + PITCH
        special_token = [token for token in special_token if token not in existing_tokens]
        self.special_token = special_token

        if is_roll:
          self.vocab = ['pad', 'start', 'end'] + PITCH + special_token
          self.pred_vocab_size = len(self.vocab) # pad start end
          self.vocab += PART
          self.vocab += ['mask']
          self.vocab += [f'beat:{i}' for i in range(6)]
        else:
          self.pred_vocab_size = len(existing_tokens+ special_token) + 3 # pad start end
          self.vocab = ['pad', 'start', 'end'] + pos_tokens + PITCH + special_token + PART # + special_token]
          self.vocab += [f'prev{x}' for x in pos_tokens] # add prev position token
        self.vocab += [f'jg:{i}' for i in range(20)] # add jg position
        self.vocab += [f'gak:{i}' for i in range(10)] # add gak position
        # sorted([tok for tok in list(set([note for inst in self.parts for measure in inst for note in measure])) if tok not in PITCH + position_token+ ['|']+['\n']])
        self.tok2idx = {value:i for i, value in enumerate(self.vocab) }  
        self.vocab_size_dict = {'total': len(self.vocab)}
        self.pos_vocab = pos_tokens
        self.note2token = {}
    
    
    def __call__(self, note_feature:Union[List[str], str]):
        if isinstance(note_feature, list):
          return [self(x) for x in note_feature]
        return self.tok2idx[note_feature]      
              
    def save_to_json(self, json_fn):
        with open(json_fn, 'w') as f:
            json.dump(self.vocab, f, ensure_ascii=False)
    
    def load_from_json(self, json_fn):
        with open(json_fn, 'r') as f:
            self.vocab = json.load(f)
        self.tok2idx = {value:i for i, value in enumerate(self.vocab) }
        self.pred_vocab_size = self.vocab.index(PART[0])
        self.vocab_size_dict = {'total': len(self.vocab)}
        if 'beat:0' in self.vocab:
          self.pos_vocab = BEAT_POSITION
        else:
          self.pos_vocab = POSITION
        
    def decode(self, idx:Union[torch.Tensor, List[int], int]):
        if isinstance(idx, torch.Tensor):
          idx = idx.tolist()
        if isinstance(idx, list):
          return [self.decode(x) for x in idx]
        return self.vocab[idx]


# class RollJGTokenizer(JeongganTokenizer):
#   def __init__(self, special_token, feature_types=['index', 'token', 'position'], json_fn=None):
#     super().__init__(special_token, feature_types, json_fn)

#   def _update_vocab(self):
#     new_vocab = [word for word in self.vocab if 'prev' not in word]
#     new_vocab += ['음표시작', '비어있음']
#     new_vocab += [f'beat:{i}' for i in range(6)]
#     self.vocab = new_vocab
#     self.tok2idx = {value:i for i, value in enumerate(self.vocab) }
#     self.vocab_size_dict = {'total': len(self.vocab)}



class JeongganDataset:
  def __init__(self, data_path= Path('music_score/gen_code'),
              slice_measure_num = 4,
              is_valid=False,
              use_pitch_modification=False,
              pitch_modification_ratio=0.3,
              # min_meas=3,
              # max_meas=6,
              jeonggan_valid_set =['남창우조 두거', '여창계면 평거', '취타 길타령', '영산회상 중령산', '평조회상 가락덜이', '관악영산회상 염불도드리'],
              feature_types=['token', 'in_jg_position', 'jg_offset', 'gak_offset', 'inst'],
              # target_instrument='daegeum',
              num_max_inst:int=5,
              augment_param=None,
              position_tokens=POSITION,
              piece_list:List[JeongganPiece]=None,
              tokenizer:JeongganTokenizer=None, 
              is_pos_counter=True,
              use_offset=False,):
    
    data_path = Path(data_path)
    self.data_path = data_path
    self.is_valid = is_valid
    self.position_tokens = position_tokens
    self.is_pos_counter = is_pos_counter
    self.use_offset = use_offset
    self.num_max_inst = num_max_inst
    if self.use_offset: 
      self.position_tokens = BEAT_POSITION
    if self.is_pos_counter:
      self.feature_types = feature_types
    else:
      self.feature_types = [feature_types[0]]+[feature_types[-1]]
      
    if piece_list:
      self.all_pieces = piece_list
    else:
      texts = glob.glob(str(data_path / '*.txt'))
      all_pieces = [JeongganPiece(text, use_offset=use_offset) for text in texts]
      # all_pieces = [ABCPiece(text) for text in texts]
      self.all_pieces = [x for x in all_pieces if x.is_clean]

    self._get_tokenizer(feature_types, tokenizer)
    self.all_pieces = [piece for piece in self.all_pieces if (piece.name in jeonggan_valid_set) == is_valid]
    

    # self.target_instrument = target_instrument
    # self.condition_instruments = [PART[i] for i in range(PART.index(target_instrument)+1, len(PART))] 

    self.entire_segments = [segment for piece in self.all_pieces for segment in piece.sliced_parts_by_measure]


    self.all_pieces = [piece for piece in self.all_pieces if (piece.name in jeonggan_valid_set) == is_valid]
    
    if self.is_pos_counter:
      self.feature_types = feature_types
    else:
      self.feature_types = [feature_types[0]]+[feature_types[-1]]
    # self.target_instrument = target_instrument
    # self.condition_instruments = [PART[i] for i in range(PART.index(target_instrument)+1, len(PART))] 

    self.entire_segments = [segment for piece in self.all_pieces for segment in piece.sliced_parts_by_measure]

  def _get_tokenizer(self, feature_types, tokenizer:JeongganTokenizer=None):
    if tokenizer:
      self.tokenizer = tokenizer
      self.vocab = tokenizer.vocab
    else:
      unique_token = sorted(list(set([key for piece in self.all_pieces for key in piece.token_counter.keys() ])))
      self.tokenizer = JeongganTokenizer(unique_token, feature_types=feature_types, use_offset=self.use_offset)
      self.vocab = self.tokenizer.vocab

  def __len__(self):
    return len(self.entire_segments)
  
  def make_compound_word_in_order(self, token, inst, prev_position_token, current_jg_idx, current_gak_idx):
    new_token = []
    for feat in self.feature_types:
      if feat == 'token':
        new_token.append(token)
      elif feat == 'inst':
        new_token.append(inst)
      elif feat == 'in_jg_position':
        new_token.append(f'prev{prev_position_token}')
      elif feat == 'jg_offset':
        new_token.append(f'jg:{current_jg_idx}')
      elif feat == 'gak_offset':
        new_token.append(f'gak:{current_gak_idx}')
    return new_token
  
  def get_inst_and_position_feature(self, tokens:List[str], inst:str):
    new_tokens = []
    # hash_note = note_list[0][1] :0 :2
    prev_position_token = '|'
    current_jg_idx = 0
    current_gak_idx = 0
    for token in tokens:
      if self.is_pos_counter:
        expanded_token = self.make_compound_word_in_order(token, inst, prev_position_token, current_jg_idx, current_gak_idx)
        if token in self.position_tokens:
          prev_position_token = token 
        if token == '|':
          current_jg_idx += 1
        elif token == '\n':
          current_gak_idx += 1
          current_jg_idx = 0
        new_tokens.append(expanded_token)
      else: 
        expanded_token = [token, inst]
        new_tokens.append(expanded_token)
    return new_tokens  
  

  def prepare_special_tokens(self, back_part_idx):
      source_start_token = ['start'] * (len(self.feature_types) - 1) + [back_part_idx] 
      source_end_token =  ['end'] * (len(self.feature_types) - 1)  + [back_part_idx] 
      target_start_token =  ['start'] * (len(self.feature_types) - 1) + [back_part_idx] 
      target_end_token =  ['end'] * (len(self.feature_types) - 1)+ [back_part_idx] 
      return source_start_token, source_end_token, target_start_token, target_end_token

  
  def shift_condition(self, note_list:List[List[str]]):
      num_features = len(self.feature_types)
      pred_features = ['start'] + [note[0] for note in note_list] # pred feature idx is 0
      conditions = [note[1:] for note in note_list]

      last_condition = []
      last_tokens = {'inst': note_list[0][self.feature_types.index('inst')],
                     'in_jg_position': 'prev\n', 
                     'jg_offset': 'jg:0',
                      'gak_offset': f'gak:{int(note_list[-1][self.feature_types.index("gak_offset")][4:])+1}'}
      for feature in self.feature_types:
        if feature == 'token': continue
        last_condition.append(last_tokens[feature])
      conditions = conditions +  [last_condition]
      combined = [ [pred_features[i]] + conditions[i] for i in range(len(pred_features))] + [['end'] * num_features]
      
      return combined


  def get_processed_feature(self, condition_insts: List[str], target_inst: str, idx: int, force_target_inst:str=None):
      assert isinstance(condition_insts, list), "front_part_insts should be a list"
      
      if force_target_inst:
        source_start_token, source_end_token, target_start_token, target_end_token = self.prepare_special_tokens(force_target_inst)
      else:
        source_start_token, source_end_token, target_start_token, target_end_token = self.prepare_special_tokens(target_inst)
      original_source = {inst: self.entire_segments[idx][inst] for inst in condition_insts}
      original_target = self.entire_segments[idx][target_inst]

      expanded_source = [self.get_inst_and_position_feature(tokens, inst) for inst, tokens in original_source.items()]
      expanded_source = [x for sublist in expanded_source for x in sublist]

      source = [source_start_token] + expanded_source + [source_end_token]
      target = self.get_inst_and_position_feature(original_target, target_inst)
      if self.is_pos_counter:
        target = self.shift_condition(target)
      shifted_target = target[1:]
      target = target[:-1]
      shifted_target = [x[0] for x in shifted_target]
      return self.tokenizer(source), self.tokenizer(target), self.tokenizer(shifted_target)


  def __getitem__(self, idx):
    insts_of_piece = list(self.entire_segments[idx].keys())
    if self.is_valid:
      target_instrument = insts_of_piece[0]
      condition_instruments = insts_of_piece[1:]
    else:
      target_instrument = random.choice(insts_of_piece)
      condition_instruments = random.sample([inst for inst in insts_of_piece if inst != target_instrument], random.randint(1, min(len(insts_of_piece)-1, self.num_max_inst ) ))

    src, tgt, shifted_tgt = self.get_processed_feature(condition_instruments, target_instrument, idx)          
    return torch.LongTensor(src), torch.LongTensor(tgt), torch.LongTensor(shifted_tgt)

class ABCTokenizer:
    def __init__(self, special_token, feature_types=['token', 'beat_offset', 'jg_offset', 'gak_offset', 'inst'], is_roll=False, json_fn=None):
        if json_fn:
          self.load_from_json(json_fn)
          return      
        self.key_types = feature_types
        existing_tokens = PITCH + DURATION
        special_token = [token for token in special_token if token not in existing_tokens]
        self.special_token = special_token
        self.pred_vocab_size = len(existing_tokens+ special_token) + 3 # pad start end

        self.vocab = ['pad', 'start', 'end'] + DURATION + PITCH + special_token + PART # + special_token]
        self.vocab += [f'beat:{x}' for x in BEAT] # add beat position token
        self.vocab += [f'jg:{i}' for i in range(20)]  # add jg position 
        self.vocab += [f'gak:{i}' for i in range(10)] # add gak position
        # sorted([tok for tok in list(set([note for inst in self.parts for measure in inst for note in measure])) if tok not in PITCH + position_token+ ['|']+['\n']])
        self.tok2idx = {value:i for i, value in enumerate(self.vocab) }  
        self.vocab_size_dict = {'total': len(self.vocab)}
        self.dur_vocab = DURATION
        self.note2token = {}
    
          
    # def __call__(self, note_feature:Union[List[str], str]):
    #     if isinstance(note_feature, list):
    #       return [self(x) for x in note_feature]
    #     return self.tok2idx[note_feature]    
              
    def __call__(self, note_feature:Union[List[str]]):
      all_note_features = []
      for note in note_feature: 
        if isinstance(note, list):
          tokenized_note = [self.tok2idx[x] for x in note]
          all_note_features.append(tokenized_note)
        else:
          all_note_features.append(self.tok2idx[note])
      return all_note_features

    def save_to_json(self, json_fn):
      with open(json_fn, 'w') as f:
          json.dump(self.vocab, f, cls=FractionEncoder)
          
    def load_from_json(self, json_fn):
        with open(json_fn, 'r') as f:
            self.vocab = json.load(f, object_hook=as_fraction)
        print(self.vocab)
        self.tok2idx = {value:i for i, value in enumerate(self.vocab) }
        self.pred_vocab_size = self.vocab.index(PART[0])
        self.vocab_size_dict = {'total': len(self.vocab)}
        self.dur_vocab = DURATION
        
    def decode(self, idx:Union[torch.Tensor, List[int], int]):
        if isinstance(idx, torch.Tensor):
          idx = idx.tolist()
        if isinstance(idx, list):
          return [self.decode(x) for x in idx]
        return self.vocab[idx]
class ABCDataset(JeongganDataset):
  def __init__(self, data_path= Path('music_score/gen_code'),
              slice_measure_num = 4,
              is_valid=False,
              use_pitch_modification=False,
              pitch_modification_ratio=0.3,
              # min_meas=3,
              # max_meas=6,
              jeonggan_valid_set =['남창우조 두거', '여창계면 평거', '취타 길타령', '영산회상 중령산', '평조회상 가락덜이', '관악영산회상 염불도드리'],
              feature_types=['token', 'in_jg_position', 'jg_offset', 'gak_offset', 'inst'],
              # target_instrument='daegeum',
              position_tokens=POSITION,
              piece_list:List[JeongganPiece]=None,
              tokenizer:JeongganTokenizer=None, 
              is_pos_counter=True,
              augment_param=None,
              num_max_inst=6):
    super().__init__(data_path=data_path, 
                     slice_measure_num=slice_measure_num, 
                     is_valid=is_valid, 
                     use_pitch_modification=use_pitch_modification, 
                     pitch_modification_ratio=pitch_modification_ratio, 
                     jeonggan_valid_set=jeonggan_valid_set, 
                     feature_types=feature_types,
                     position_tokens=position_tokens, 
                     piece_list=piece_list, 
                     tokenizer=tokenizer, 
                     is_pos_counter=is_pos_counter)
    if self.is_pos_counter:
      self.feature_types = feature_types
    else:
      self.feature_types = [feature_types[0]]+[feature_types[-1]]
      
    if piece_list:
      self.all_pieces = piece_list
    else:
      texts = glob.glob(str(data_path / '*.txt'))
      # all_pieces = [JeongganPiece(text) for text in texts]
      all_pieces = [ABCPiece(text) for text in texts]
      self.all_pieces = [x for x in all_pieces if x.is_clean]

    self._get_tokenizer(feature_types, tokenizer)
    self.all_pieces = [piece for piece in self.all_pieces if (piece.name in jeonggan_valid_set) == is_valid]
    

    # self.target_instrument = target_instrument
    # self.condition_instruments = [PART[i] for i in range(PART.index(target_instrument)+1, len(PART))] 

    self.entire_segments = [segment for piece in self.all_pieces for segment in piece.sliced_parts_by_measure]


    self.all_pieces = [piece for piece in self.all_pieces if (piece.name in jeonggan_valid_set) == is_valid]
    
    if self.is_pos_counter:
      self.feature_types = feature_types
    else:
      self.feature_types = [feature_types[0]]+[feature_types[-1]]
    # self.target_instrument = target_instrument
    # self.condition_instruments = [PART[i] for i in range(PART.index(target_instrument)+1, len(PART))] 

    self.entire_segments = [segment for piece in self.all_pieces for segment in piece.sliced_parts_by_measure]
  
  def _get_tokenizer(self, feature_types, tokenizer:JeongganTokenizer=None):
    if tokenizer:
      self.tokenizer = tokenizer
      self.vocab = tokenizer.vocab
    else:
      unique_token = sorted(list(set([key for piece in self.all_pieces for key in piece.token_counter.keys() ])))
      self.tokenizer = ABCTokenizer(unique_token, feature_types=feature_types)
      self.vocab = self.tokenizer.vocab
      
  def get_processed_feature(self, condition_insts: List[str], target_inst: str, idx: int):
      assert isinstance(condition_insts, list), "front_part_insts should be a list"
      
      source_start_token, source_end_token, target_start_token, target_end_token = self.prepare_special_tokens(target_inst)
      original_source = {inst: self.entire_segments[idx][inst] for inst in condition_insts}
      original_target = self.entire_segments[idx][target_inst]
      expanded_source = [token+[inst] for inst, tokens in original_source.items() for token in tokens]
      expanded_target = [token+[target_inst] for token in original_target]
      # expanded_source = [x for sublist in expanded_source for x in sublist]

      source = [source_start_token] + expanded_source + [source_end_token]
      # target = self.get_inst_and_position_feature(original_target, target_inst)
      if self.is_pos_counter:
        target = self.shift_condition(expanded_target)

      shifted_target = target[1:]
      target = target[:-1]
      shifted_target = [x[0] for x in shifted_target]
      return self.tokenizer(source), self.tokenizer(target), self.tokenizer(shifted_target)
  
  def shift_condition(self, note_list:List[List[str]]):
      num_features = len(self.feature_types)
      pred_features = ['start'] + [note[0] for note in note_list] # pred feature idx is 0
      conditions = [note[1:] for note in note_list]

      last_condition = []
      last_tokens = {'inst': note_list[0][self.feature_types.index('inst')],
                     'in_jg_position': 'beat:0', 
                     'jg_offset': 'jg:0',
                      'gak_offset': f'gak:{int(note_list[-1][self.feature_types.index("gak_offset")][4:])+1}'}
      for feature in self.feature_types:
        if feature == 'token': continue
        last_condition.append(last_tokens[feature])
      conditions = conditions +  [last_condition]
      combined = [ [pred_features[i]] + conditions[i] for i in range(len(pred_features))] + [['end'] * num_features]
      
      return combined
    
  def __getitem__(self, idx):
    insts_of_piece = list(self.entire_segments[idx].keys())
    if self.is_valid:
      target_instrument = insts_of_piece[0]
      condition_instruments = insts_of_piece[1:]
    else:
      target_instrument = random.choice(insts_of_piece)
      condition_instruments = random.sample([inst for inst in insts_of_piece if inst != target_instrument], random.randint(1, len(insts_of_piece)-2))

    
    src, tgt, shifted_tgt = self.get_processed_feature(condition_instruments, target_instrument, idx)        
    return torch.LongTensor(src), torch.LongTensor(tgt), torch.LongTensor(shifted_tgt)    
class JGMaskedDataset(JeongganDataset):
  def __init__(self, data_path='music_score/gen_code', 
               slice_measure_num=4, 
               is_valid=False, 
               jeonggan_valid_set=['남창우조 두거', '여창계면 평거', '취타 길타령', '영산회상 중령산', '평조회상 가락덜이', '관악영산회상 염불도드리'], 
               feature_types=['token', 'ornaments', 'in_jg_position', 'jg_offset', 'gak_offset', 'inst'], 
               position_tokens=POSITION, 
               augment_param:dict={},
               piece_list: List[JeongganPiece] = None, 
               tokenizer: JeongganTokenizer = None,
               num_max_inst:int=6):
    super().__init__(data_path, slice_measure_num, is_valid, False, False, jeonggan_valid_set=jeonggan_valid_set, feature_types=feature_types, position_tokens=position_tokens, piece_list=piece_list, tokenizer=tokenizer, num_max_inst=num_max_inst)
    self.entire_segments = self.get_entire_segments()
    self.unique_pitches, self.unique_ornaments = self._get_unique_pitch_and_ornaments()
    self.augmentor = Augmentor(self.tokenizer, self.unique_pitches, self.unique_ornaments, **augment_param)
    self.num_max_inst = num_max_inst

  def _get_unique_pitch_and_ornaments(self):
    unique_pitches = set()
    unique_ornaments = set()
    for seg in self.entire_segments:
      for inst, roll in seg.items():
        pitches = [x[0] for x in roll]
        ornaments = [x[1] for x in roll]
        unique_pitches.update(set(pitches))
        unique_ornaments.update(set(ornaments))
    return sorted(list(unique_pitches)), sorted(list(unique_ornaments))
  
  def _get_tokenizer(self, feature_types, tokenizer:JeongganTokenizer=None):
    if tokenizer:
      self.tokenizer = tokenizer
      self.vocab = tokenizer.vocab
    else:
      unique_token = sorted(list(set([key for piece in self.all_pieces for key in piece.token_counter.keys() ]))) + ['비어있음', 'mask']
      self.tokenizer = JeongganTokenizer(unique_token, feature_types=feature_types, is_roll=True)
      self.vocab = self.tokenizer.vocab
  
  def get_entire_segments(self):
    entire_segments = []
    for piece in tqdm(self.all_pieces, desc='Converting segments to roll...'):
      entire_segments += self._get_roll_by_part(piece.sliced_parts_by_measure)
    return entire_segments
  
  def _get_roll_by_part(self, sliced_parts_by_measure:List[Dict[str, List[str]]]):
    return [{inst: JeongganPiece.convert_tokens_to_roll(tokens, inst) for inst, tokens in meas_data.items()} for meas_data in sliced_parts_by_measure]

  def prepare_special_tokens(self):
    source_start_token = ['start'] * len(self.feature_types)
    source_end_token =  ['end'] * len(self.feature_types)
    return torch.LongTensor(self.tokenizer(source_start_token)), torch.LongTensor(self.tokenizer(source_end_token))

  def get_processed_feature(self, selected_insts: List[str], idx:int):
    start_token, end_token = self.prepare_special_tokens()

    x = self.tokenizer([x.tolist() for inst, x in self.entire_segments[idx].items() if inst in selected_insts])
    x = torch.LongTensor(x).permute(1,2,0)
    correct_x = x.clone()
    x, loss_mask = self.augmentor(x)

    x = x.permute(2,0,1).flatten(0,1)
    loss_mask = loss_mask.permute(2,0,1).flatten(0,1)
    loss_mask = loss_mask[:,:2]
    correct_x = correct_x.permute(2,0,1).flatten(0,1)
    x = torch.cat([start_token.unsqueeze(0), x, end_token.unsqueeze(0)])
    correct_x = torch.cat([start_token.unsqueeze(0), correct_x, end_token.unsqueeze(0)])
    loss_mask = torch.cat([torch.zeros(1, 2), loss_mask, torch.zeros(1,2)])
    return x, correct_x, loss_mask
  
  def __getitem__(self, idx):
    insts_of_piece = list(self.entire_segments[idx].keys())
    if self.is_valid:
      insts = insts_of_piece[:self.num_max_inst]
    else:
      insts = random.sample(insts_of_piece, random.randint(1, min(len(insts_of_piece), self.num_max_inst)))

    
    masked_src, org_src, loss_mask = self.get_processed_feature(insts, idx)          
    return masked_src, org_src, loss_mask