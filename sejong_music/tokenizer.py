import json

from typing import List, Set, Dict, Tuple, Union

from fractions import Fraction
from collections import defaultdict
from .utils import  as_fraction, FractionEncoder
import torch

class Tokenizer:
    def __init__(self, parts, feature_types=['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx'], json_fn=None):
        if json_fn:
            self.load_from_json(json_fn)
            return      
        num_parts = len(parts)
        self.key_types = feature_types
        vocab_list = defaultdict(list)
        vocab_list['index'] = [part.index for part in parts]
        vocab_list['pitch'] = ['pad', 'start', 'end'] + sorted(list(set([note.pitch for i in range(num_parts) for note in parts[i].tie_cleaned_notes])))
        vocab_list['duration'] = ['pad', 'start', 'end'] + [8.0] + sorted(list(set([note.duration for i in range(num_parts) for note in parts[i].tie_cleaned_notes])))
        if 'offset' in feature_types:
            vocab_list['offset'] = ['pad', 'start', 'end'] + sorted(list(set([note.measure_offset for i in range(num_parts) for note in parts[i].tie_cleaned_notes])))
        if 'daegang_offset' in feature_types:
            vocab_list['daegang_offset'] = ['pad', 'start', 'end'] + [i for i in range(0, 4)]
            vocab_list['jeonggan_offset'] = ['pad', 'start', 'end'] + [i for i in range(0, 6)]
            vocab_list['beat_offset'] = ['pad', 'start', 'end'] + [i for i in range(0, 6)]
        elif 'jeonggan_offset' in feature_types: # In case using jeonggan offset without daegang offset
            vocab_list['jeonggan_offset'] = ['pad', 'start', 'end'] + [i for i in range(0, 10)]
            vocab_list['beat_offset'] = ['pad', 'start', 'end'] + [Fraction(0, 1), Fraction(1, 3),  Fraction(1,2), Fraction(2, 3)]
        if 'dynamic' in feature_types:
            vocab_list['dynamic'] = ['pad', 'start', 'end'] + ['strong', 'middle', 'weak', 'none']
        if 'measure_idx' in feature_types:
            vocab_list['measure_idx'] = ['pad', 'start', 'end'] + [i for i in range(10)]
        if 'measure_change' in feature_types:
            vocab_list['measure_change'] = ['pad', 'start', 'end']  + [0, 1, 2]
        # vocab_list['offset'] = ['pad', 'start', 'end'] + sorted(list(set([note.measure_offset for i in range(num_parts) for note in parts[i].tie_cleaned_notes])))
        # vocab_list['dynamic'] = ['pad', 'start', 'end'] + ['strong', 'middle', 'weak']
        # vocab_list['measure_idx'] = ['pad', 'start', 'end'] + [i for i in range(10)]
        # vocab_list['measure_change'] = ['pad', 'start', 'end']  + [0, 1, 2]
        # vocab_list['measure_idx'] = ['pad', 'start', 'end'] + [i for i in range(6)]
        self.measure_duration = [parts[i].measure_duration for i in range(num_parts)]
        self.vocab = vocab_list
        self.vocab_size_dict = {key: len(value) for key, value in self.vocab.items()}
        self.tok2idx = {key: {k:i for i, k in enumerate(value)} for key, value in self.vocab.items() }
        if 'offset' in feature_types:
            self.tok2idx['offset_fraction'] = {Fraction(k).limit_denominator(3):v  for k, v in self.tok2idx['offset'].items() if type(k)==float}
            self.vocab['offset_fraction'] = [Fraction(k).limit_denominator(3) for k in self.vocab['offset'] if type(k)==float]
        
        self.key2idx = {key: i for i, key in enumerate(self.key_types)}
        # self.key_types = ['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_change']

        self.note2token = {}
        
    def save_to_json(self, json_fn):
        with open(json_fn, 'w') as f:
            json.dump(self.vocab, f, cls=FractionEncoder)
    
    def load_from_json(self, json_fn):
        with open(json_fn, 'r') as f:
            self.vocab = json.load(f, object_hook=as_fraction)
        self.key_types = list(self.vocab.keys())
        self.vocab_size_dict = {key: len(value) for key, value in self.vocab.items()}
        self.tok2idx = {key: {k:i for i, k in enumerate(value)} for key, value in self.vocab.items() }
        # if 'offset' in self.key_types:
        #   self.tok2idx['offset_fraction'] = {Fraction(k).limit_denominator(3):v  for k, v in self.tok2idx['offset'].items() if type(k)==float}
        self.key2idx = {key: i for i, key in enumerate(self.key_types)}
        self.note2token = {}
        self.measure_duration = [8.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
            

    def hash_note_feature(self, note_feature:Tuple[Union[int, float, str]]):
        if note_feature in self.note2token:
            return self.note2token[note_feature]
        else:
            out = [self.tok2idx[self.key_types[i]][element] for i, element in enumerate(note_feature)]
            self.note2token[note_feature] =  out
            return out
        
    def __call__(self, note_feature:List[Union[int, float, str]]):
        # converted_lists = [self.tok2idx[self.key_types[i]][element] for i, element in enumerate(note_feature)]     
        converted_lists = self.hash_note_feature(tuple(note_feature))
        # feature_types=['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx']
        # converted_lists = [self.tok2idx[self.key_types[i]][element] for i, element in enumerate(note_feature)]
        return converted_lists
      
      

class SingleVocabTokenizer:
    def __init__(self, unique_tokens, json_fn=None):
        if json_fn:
          self.load_from_json(json_fn)
          return      

        self.vocab = ['pad', 'start', 'end'] + sorted(list(set(unique_tokens)))
        # sorted([tok for tok in list(set([note for inst in self.parts for measure in inst for note in measure])) if tok not in PITCH + position_token+ ['|']+['\n']])
        self.tok2idx = {value:i for i, value in enumerate(self.vocab) }  
        self.vocab_size_dict = {'total': len(self.vocab)}
    
    
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
        self.vocab_size_dict = {'total': len(self.vocab)}
        
    def decode(self, idx:Union[torch.Tensor, List[int], int]):
        if isinstance(idx, torch.Tensor):
          idx = idx.tolist()
        if isinstance(idx, list):
          return [self.decode(x) for x in idx]
        return self.vocab[idx]
