
import glob
import random
import json
from pathlib import Path
from typing import List, Set, Dict, Tuple, Union
from collections import defaultdict, Counter
from pathlib import Path
import torch


POSITION = ['|', '\n']+ [f":{i}" for i in range(0, 16)]
PITCH = [ '하하배임','하배황', '하배태', '하배중', '하배임', '하배이', '하배남', '하배무',
          '배황', '배태', '배협', '배고', '배중', '배임', '배남', '배무', '배응',
          '황', '태', '협', '고', '중', '임', '이', '남', '무', 
          '청황', '청태', '청협', '청고', '청중', '청임', '청남', '청무',
          '중청황']
PART = ['daegeum', 'piri', 'haegeum', 'gayageum', 'geomungo', 'ajaeng']



def read_txt(path):
  with open(path, 'r') as file:
    txt = file.read()
    return txt

class JeongganPiece:
  def __init__(self, txt_fn):
    self.txt_fn = txt_fn
    self.name = txt_fn.split('/')[-1].split('_')[0]
    self.text_str = read_txt(txt_fn)
    self.inst_list = self.txt_fn[:-4].split('/')[-1].split('_')[1:]
    self.parts, self.part_dict = self.split_to_inst()
    self.check_measure_length()
    self.tokenized_parts = [self.split_and_filter(part) for part in self.parts]
    self.token_counter = self.count_token()
    self.sliced_parts_by_inst, self.sliced_parts_by_measure = self.prepare_sliced_measures()
    
  def split_to_inst(self):
    self.daegeum, self.piri, self.haegeum, self.ajaeng, self.gayageum, self.geomungo = None, None, None, None, None, None
    parts = self.text_str.split('\n\n')
    parts = [part + ' \n ' for part in parts]
    for i, inst in enumerate(self.inst_list):
      setattr(self, inst, parts[i])
    return parts, {inst: part for inst, part in zip(self.inst_list, parts)}

  def count_token(self):
    token_counter = Counter()
    for part in self.tokenized_parts:
      token_counter.update(part)
    return token_counter
    
  @staticmethod
  def split_and_filter(string, split_charater=' '):
    splited = string.split(split_charater)
    filtered = [x for x in splited if x != '']
    return filtered
  
  @staticmethod
  def cut(part, measure_boundary_idx, start_mea=0, end_mea=4):
    start_idx = measure_boundary_idx[start_mea]
    end_idx = measure_boundary_idx[end_mea]
    return part[start_idx:end_idx]

  @classmethod
  def cut_part_by_slice_len(self, part:str, slice_len:int):
    measure_boundary_idx = [0]+ [i+1 for i, cha in enumerate(part) if cha == '\n'] + [len(part)]
    cutted_part = [self.cut(part, measure_boundary_idx, i, i+slice_len) for i in range(len(measure_boundary_idx)-slice_len)]
    return cutted_part
    
  def check_measure_length(self):
    self.is_clean = False
    meausre_length_by_inst = [[measure.count('|') for measure in inst.split('\n')] for inst in self.parts ]
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
  
  def __len__(self):
    return len(self.parts[0].split('\n'))


class JeongganTokenizer:
    def __init__(self, special_token, feature_types=['index', 'token', 'position'], json_fn=None):
        if json_fn:
          self.load_from_json(json_fn)
          return      
        self.key_types = feature_types
        existing_tokens = POSITION + PITCH
        special_token = [token for token in special_token if token not in existing_tokens]
        self.special_token = special_token
        self.pred_vocab_size = len(existing_tokens+ special_token) + 3 # pad start end

        self.vocab = ['pad', 'start', 'end'] + POSITION + PITCH + special_token + PART # + special_token]
        self.vocab += [f'prev{x}' for x in POSITION] # add prev position token
        self.vocab += [f'jg:{i}' for i in range(20)] # add jg position
        self.vocab += [f'gak:{i}' for i in range(10)] # add gak position
        # sorted([tok for tok in list(set([note for inst in self.parts for measure in inst for note in measure])) if tok not in PITCH + position_token+ ['|']+['\n']])
        self.tok2idx = {value:i for i, value in enumerate(self.vocab) }  
        self.vocab_size_dict = {'total': len(self.vocab)}
        self.pos_vocab = POSITION
        self.note2token = {}
    
    # def __call__(self, note_feature:Union[List[str], str]):
    #     if isinstance(note_feature, list):
    #       if isinstance(note_feature[0], list):
    #         return [self.hash_note_feature(tuple(x)) for x in note_feature]
    #       else:
    #         return [self(x) for x in note_feature]
    #     return self.tok2idx[note_feature]
    
    def __call__(self, note_feature:Union[List[str], str]):
        if isinstance(note_feature, list):
          return [self(x) for x in note_feature]
        return self.tok2idx[note_feature]      
      
    # def hash_note_feature(self, note_feature:Tuple[Union[int, float, str]]):
    #     try:
    #         return self.note2token[note_feature]
    #     except:
    #         out = [self.tok2idx[x] for x in note_feature]
    #         self.note2token[note_feature] =  out
    #         return out
        
    def save_to_json(self, json_fn):
        with open(json_fn, 'w') as f:
            json.dump(self.vocab, f)
    
    def load_from_json(self, json_fn):
        with open(json_fn, 'r') as f:
            self.vocab = json.load(f)
        self.tok2idx = {value:i for i, value in enumerate(self.vocab) }
        self.pred_vocab_size = self.vocab.index(PART[0])
        self.vocab_size_dict = {'total': len(self.vocab)}
        self.pos_vocab = POSITION
        
    def decode(self, idx:Union[torch.Tensor, List[int], int]):
        if isinstance(idx, torch.Tensor):
          idx = idx.tolist()
        if isinstance(idx, list):
          return [self.decode(x) for x in idx]
        return self.vocab[idx]

            
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
              position_tokens=POSITION):
    
    self.data_path = data_path
    self.is_valid = is_valid
    self.position_tokens = position_tokens
    
    texts = glob.glob(str(data_path / '*.txt'))
    all_pieces = [JeongganPiece(text) for text in texts]
    self.all_pieces = [x for x in all_pieces if x.is_clean]
    unique_token = list(set([key for piece in self.all_pieces for key in piece.token_counter.keys() ]))
    self.tokenizer = JeongganTokenizer(unique_token, feature_types=feature_types)
    self.vocab = self.tokenizer.vocab
    self.all_pieces = [piece for piece in self.all_pieces if (piece.name in jeonggan_valid_set) == is_valid]
    
    self.feature_types = feature_types
    # self.target_instrument = target_instrument
    # self.condition_instruments = [PART[i] for i in range(PART.index(target_instrument)+1, len(PART))] 

    self.entire_segments = [segment for piece in self.all_pieces for segment in piece.sliced_parts_by_measure]

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
      expanded_token = self.make_compound_word_in_order(token, inst, prev_position_token, current_jg_idx, current_gak_idx)
      if token in self.position_tokens:
        prev_position_token = token 
      if token == '|':
        current_jg_idx += 1
      elif token == '\n':
        current_gak_idx += 1
        current_jg_idx = 0
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


  def get_processed_feature(self, condition_insts: List[str], target_inst: str, idx: int):
      assert isinstance(condition_insts, list), "front_part_insts should be a list"
      
      source_start_token, source_end_token, target_start_token, target_end_token = self.prepare_special_tokens(target_inst)
      original_source = {inst: self.entire_segments[idx][inst] for inst in condition_insts}
      original_target = self.entire_segments[idx][target_inst]

      expanded_source = [self.get_inst_and_position_feature(tokens, inst) for inst, tokens in original_source.items()]
      expanded_source = [x for sublist in expanded_source for x in sublist]

      source = [source_start_token] + expanded_source + [source_end_token]
      target = self.get_inst_and_position_feature(original_target, target_inst)
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
      condition_instruments = random.sample([inst for inst in insts_of_piece if inst != target_instrument], random.randint(1, len(insts_of_piece)-2))

    
    src, tgt, shifted_tgt = self.get_processed_feature(condition_instruments, target_instrument, idx)          
    return torch.LongTensor(src), torch.LongTensor(tgt), torch.LongTensor(shifted_tgt)
