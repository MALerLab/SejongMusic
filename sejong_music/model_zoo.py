import copy
import yaml
from fractions import Fraction
from typing import List

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

from .constants import get_dynamic, MEAS_LEN_BY_IDX
from .yeominrak_processing import Tokenizer
from .metric import convert_dynamics_to_integer, make_dynamic_template
from .module import PackedDropout, MultiEmbedding, get_emb_total_size
from .sampling_utils import nucleus
from .transformer_module import TransEncoder, TransDecoder

import x_transformers


def read_yaml(yml_path):
    with open(yml_path, 'r') as f:
        yaml_obj = yaml.load(f, Loader=yaml.FullLoader)
    config = OmegaConf.create(yaml_obj)
    return config



# def get_measure_duration(source, vocab):
#   measure_set = defaultdict(list)
#   for i in range(1, len(source)-1):
#     measure = source[i, 2].item()
#     measure_set[measure].append(vocab.vocab['duration'][source[i, 1].item()])
#   sum_value = [sum(value) for value in measure_set.values()]
#   return max(sum_value)

class Converter:
  def __init__(self, tokenizer:Tokenizer):
    self.tokenizer = tokenizer
    
  def _convert_note(self, note:List[int]):
    key_types = self.tokenizer.key_types[:len(note)]
    return [self.tokenizer.vocab[key][note[i]] for i, key in enumerate(key_types)]

    # return [int(note[0]), self.vocab['pitch'][note[1]], self.vocab['duration'][note[2]], self.vocab['offset'][note[3]], self.vocab['dynamic'][note[4]], self.vocab['measure_change'][note[5]]]

  def __call__(self, output):
    # print(output) # tensor([[7, 6, 6, 3, 3]])

    return [self._convert_note(note) for note in output]
  
class RollConverter(Converter):
  def __init__(self, tokenizer:Tokenizer, sampling_rate=2):
    super().__init__(tokenizer)
    self.sampling_rate = sampling_rate

  def convert_to_note_event(self, frame_list):
    note_list = []
    prev_pitch = 0
    note_duration = 0
    part_idx = frame_list[0][0]
    for frame in frame_list:
      _, pitch = frame
      if pitch == 0:
        note_duration += 1
        continue
      else:
        if prev_pitch != 0:
          note_list.append([part_idx, prev_pitch, note_duration / self.sampling_rate])
        prev_pitch = pitch
        note_duration = 1
    if prev_pitch != 0:
      note_list.append([part_idx, prev_pitch, note_duration / self.sampling_rate])
    return note_list
  
  def __call__(self, output): #[pitch, in_onset]
    converted = [self._convert_note(note) for note in output]
    converted = self.convert_to_note_event(converted)
    return converted

# class RollConverter:
#   def __init__(self, vocab):
#     self.vocab = vocab 
  
#   def __call__(self, output): #[pitch, in_onset]
#     return [[int(note[0]), self.vocab['pitch'][int(note[1])], int(note[2])] for note in output]


#================ model =================

class Seq2seq(nn.Module):
  def __init__(self, tokenizer:Tokenizer, config):
    super().__init__()
    self.tokenizer = tokenizer
    self.vocab_size_dict = self.tokenizer.vocab_size_dict
    self.config = config
    self._make_encoder_and_decoder()
    self.vocab_size = [x for x in self.vocab_size_dict.values()] # {'index': 8, 'pitch': 17, 'duration': 15, 'offset': 62, 'dynamic': 6}
    self.converter = Converter(self.tokenizer)
    self.is_condition_shifted = False

  def _make_encoder_and_decoder(self):
    self.encoder = Encoder(self.vocab_size_dict, self.config)
    self.decoder = Decoder(self.vocab_size_dict, self.config)

  def forward(self, src, tgt):
    enc_out, enc_last_hidden = self.encoder(src) # enc_last_hidden : [num_layers * num_directions, batch, hidden_size]
    dec_out = self.decoder(tgt, enc_last_hidden)
    return dec_out
  
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
      if 'dynamic' in self.vocab_size_dict:
        token_features.append(self.tokenizer.tok2idx['dynamic']['strong'])
      if 'measure_change' in self.vocab_size_dict:
        token_features.append(self.tokenizer.tok2idx['measure_change'][1])
      if 'measure_idx' in self.vocab_size_dict:
        token_features.append(self.tokenizer.tok2idx['measure_idx'][0])      
      return torch.LongTensor([token_features])
    
    if 'offset' in self.vocab_size_dict:
      token_features.append(self.tokenizer.tok2idx['offset']['start'])
    if 'dynamic' in self.vocab_size_dict:
      token_features.append(self.tokenizer.tok2idx['dynamic']['start'])
    if 'measure_change' in self.vocab_size_dict:
      token_features.append(self.tokenizer.tok2idx['measure_change']['start'])
    if 'measure_idx' in self.vocab_size_dict:
      token_features.append(self.tokenizer.tok2idx['measure_idx']['start'])
    return torch.LongTensor([token_features])
    # return torch.LongTensor([[part_idx] + [1] * num_features])

  def encode_condition_token(self, part_idx, current_beat, current_measure_idx, measure_start):
    condition_features = []
    if 'offset' in self.tokenizer.tok2idx:
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
          print(f"current beat: {current_beat}")
          print(f"part_idx: {part_idx}")
          print(f"current_measure_idx: {current_measure_idx}")
          print(f"measure_start: {measure_start}")
          print(f"measure_duration: {self.tokenizer.measure_duration[part_idx]}")
          print(f"current_beat: {float(current_beat)}")
          print(f"current beat type: {type(current_beat)}")
          print(f"current beat==10.0 : {current_beat==10.0}")
          print(f"current beat + 0.1 > 10.0 : {current_beat+0.1>10.0}")
          measure_offset_idx = self.tokenizer.tok2idx['offset'][0.0]
      condition_features.append(measure_offset_idx)
    if 'dynamic' in self.tokenizer.tok2idx:
      dynamic = get_dynamic(current_beat, part_idx)
      dynamic_idx = self.tokenizer.tok2idx['dynamic'][dynamic]
      condition_features.append(dynamic_idx)
    if 'measure_idx' in self.tokenizer.tok2idx:
      measure_idx = self.tokenizer.tok2idx['measure_idx'][current_measure_idx]
      condition_features.append(measure_idx)
    if 'measure_change' in self.tokenizer.tok2idx:
      measure_start_idx = self.tokenizer.tok2idx['measure_change'][measure_start]
      condition_features.append(measure_start_idx)
    return condition_features

  def update_condition_info(self, current_beat, current_measure_idx, measure_duration, current_dur):
    if current_dur in ('start', 'pad'):
      current_dur = 0.0
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
  
  def _get_measure_duration(self, part_idx):
    return self.tokenizer.measure_duration[part_idx]
  
  def _decode_inference_result(self, src, output, other_out):
    return self.converter(src[1:-1]), self.converter(output), other_out
  
  @torch.inference_mode()
  def inference(self, src, part_idx):
    enc_output, enc_last_hidden = self.encoder(src)
    # print(enc_last_hidden.shape)
    start_token = self.get_start_token(part_idx)
    
    assert src.ndim == 2 # sequence length, feature length
    #------------
      
    not_triplet_vocab_list = [ i for i, dur in enumerate(self.tokenizer.vocab['duration'][3:], 3) if (dur%0.5) != 0.0]
    #--------
    measure_duration = self.tokenizer.measure_duration[part_idx]
    current_measure_offset_idx = 0
    current_dynamic_idx = 'strong'
    
    last_hidden = enc_last_hidden
    selected_token = start_token
    current_beat = 0.0
    measure_offset_idx = self.tokenizer.tok2idx['offset'][current_measure_offset_idx]
    dynamic_idx = self.tokenizer.tok2idx['dynamic'][current_dynamic_idx]
    
    final_tokens = []
    
    while True:
    # for _ in range(5):
      # print(selected_token.shape, last_hidden.shape)
      selected_token, last_hidden = self.decoder.forward_one_step(selected_token, last_hidden, not_triplet_vocab_list)
      selected_token = torch.cat([selected_token, torch.LongTensor([condition_features])], dim=1)
      selected_token = torch.cat([torch.LongTensor([[part_idx]]), selected_token], dim=-1)
      if selected_token[0, 1]==self.tokenizer.tok2idx['pitch']['end'] or selected_token[0, 2]==self.tokenizer.tok2idx['duration']['end']:
        break
      
      current_dur = self.tokenizer.vocab['duration'][selected_token[0, 2]]
      print('current_duration is', current_dur)
      current_beat, current_measure_idx, measure_changed = self.update_condition_info(current_beat, current_measure_idx, measure_duration, current_dur)
      condition_features = self.encode_condition_token(part_idx, current_beat, current_measure_idx, measure_changed)
      final_tokens.append(selected_token)

    return self.converter(src[1:-1]), self.converter(torch.cat(final_tokens, dim=0))
  
      # prob = PackedSequence(prob, x[1], x[2], x[3])

# class TransformerSeq2seq(Seq2seq):
#   def __init__(self, tokenizer:Tokenizer, config):
#     super().__init__(tokenizer, config)
#     self.encoder = 


class Encoder(nn.Module):
  def __init__(self, vocab_size_dict: dict, config):
    super().__init__()
    self.param = config
    self.vocab_size = [x for x in vocab_size_dict.values()]
    self.vocab_size_dict = vocab_size_dict
    self._make_embedding_layer()
    self.rnn = nn.GRU(self.param.emb.total_size, self.param.gru.hidden_size, num_layers=self.param.gru.num_layers, batch_first=True, bidirectional=True, dropout=0.2)
    self.hidden_size = self.param.gru.hidden_size
    self.num_layers = self.param.gru.num_layers
    self.return_reduced_bidirectional = True
    
  def _make_embedding_layer(self):
    # self.emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.net_param.nn_params.emb.emb_size)
    self.emb = MultiEmbedding(self.vocab_size_dict, self.param.emb)
    
  def _get_embedding(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
    else:
 
      emb = self.emb(input_seq)
    return emb

  def forward(self, input_seq):

    emb = self._get_embedding(input_seq)
    out, last_hidden = self.rnn(emb)
      
    if self.rnn.bidirectional and self.return_reduced_bidirectional:
      last_hidden = last_hidden.view(self.num_layers, 2, -1, self.hidden_size)
      last_hidden = last_hidden.mean(dim=1)
      
      if isinstance(out, PackedSequence):
        out = PackedSequence(out.data.view(out.data.shape[0], 2, -1).mean(dim=1), out[1], out[2], out[3])
      else:
        out = out.view(out.shape[0], out.shape[1], 2, -1).mean(dim=2)
    return out, last_hidden
  
class EncoderWithDropout(Encoder):
  def __init__(self, vocab_size_dict: dict, config):
    super().__init__(vocab_size_dict, config)
    self.emb_dropout = PackedDropout(config.emb_dropout)
  
  def _get_embedding(self, input_seq):
    x = super()._get_embedding(input_seq)
    return self.emb_dropout(x)

class Decoder(nn.Module):
  def __init__(self, vocab_size_dict: dict, param):
    super().__init__()
    self.param = param
    self.vocab_size = [x for x in vocab_size_dict.values()] 
    self.vocab_size_dict = vocab_size_dict
    self._make_embedding_layer()
    self.rnn = nn.GRU(self.param.emb.total_size, self.param.gru.hidden_size, num_layers=self.param.gru.num_layers, batch_first=True)
    self.hidden_size = self.param.gru.hidden_size
    self.num_layers = self.param.gru.num_layers
    self._make_projection_layer()
  
  def _make_embedding_layer(self):
    # self.emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.net_param.nn_params.emb.emb_size)
    self.emb = MultiEmbedding(self.vocab_size_dict, self.param.emb)
    
  def _make_projection_layer(self):
    # total_vocab_size = sum([x for x in self.vocab_size_dict.values()])
    self.proj = nn.Linear(self.hidden_size, self.vocab_size[1] + self.vocab_size[2])
  
  def _get_embedding(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
    else:
      emb = self.emb(input_seq)
    return emb
  
  def _apply_softmax(self, logit):
    pitch_output = torch.softmax(logit[...,:self.vocab_size[1]], dim=-1)
    # print(logit.shape)
    dur_output = torch.softmax(logit[...,self.vocab_size[1] :], dim=-1)
    # print(f"{pitch_output.shape, dur_output.shape}!!")
    return torch.cat([pitch_output, dur_output], dim=1)
    
  def forward(self, x, enc_output):
    if isinstance(x, PackedSequence):
      emb = self._get_embedding(x)
      out, _ = self.rnn(emb, enc_output)
      # out = pad_packed_sequence(out, batch_first=True)[0]
      logit = self.proj(out.data)
      prob = self._apply_softmax(logit)
      prob = PackedSequence(prob, x[1], x[2], x[3])
    return prob
  
  def _select_token(self, prob):
    tokens = []
    pitch_token = prob.data[:, :self.vocab_size[1]].multinomial(num_samples=1)
    dur_token = prob.data[:, self.vocab_size[1]:].multinomial(num_samples=1)
    # dur_token = torch.argmax(prob.data[:, model.vocab_size[0]:], dim=-1)
    return torch.cat([pitch_token, dur_token], dim=-1)
  
  def forward_one_step(self, x, last_hidden, not_triplet_vocab_list): # 처음에는 encoder_output = last_hidden
    emb = self._get_embedding(x)
    # print(emb.shape, last_hidden.shape)
    out, last_hidden = self.rnn(emb, last_hidden)
    logit = self.proj(out.data)
    not_triplet_vocab_list = [i+self.vocab_size[1] for i in not_triplet_vocab_list]
    logit[:, not_triplet_vocab_list] = -1e10
    prob = self._apply_softmax(logit)
    selected_token = self._select_token(prob)
    return selected_token, last_hidden
      
  
# ---------------Attention --------------#


class AttentionSeq2seq(Seq2seq):
  def __init__(self, vocab, config):
    super().__init__(vocab, config)
    self.decoder = AttnDecoder(self.vocab_size_dict, self.config)    

  def forward(self, src, tgt):
    enc_out, enc_last_hidden = self.encoder(src)
    dec_out = self.decoder(src, tgt, enc_out, enc_last_hidden)
     
    return dec_out  


  def inference(self, src, part_idx):
    with torch.inference_mode():
      
      encode_out, encode_last_hidden = self.encoder(src.unsqueeze(0))
      # enc_emb = self.encoder._get_embedding(src)
      # # print(enc_emb.shape)
      # enc_emb = enc_emb.unsqueeze(0)
      # encode_out, encode_last_hidden = self.encoder.rnn(enc_emb)

      # Setup for 0th step
      start_token = torch.LongTensor([[part_idx, 1, 1, 1, 1]]) # start token idx is 1
      assert src.ndim == 2 # sequence length, feature length
      #-------------
      not_triplet_vocab_list = [ i for i, dur in enumerate(self.vocab.vocab['duration'][3:], 3) if (dur%0.5) != 0.0]
      #-------------
      measure_duration = self.vocab.measure_duration[part_idx]
      current_measure_offset_idx = 0
      current_dynamic_idx = 'strong'
      
      last_hidden = encode_last_hidden
      selected_token = start_token
      current_beat = 0.0
      
      measure_offset_idx = self.vocab.tok2idx['offset'][current_measure_offset_idx]
      dynamic_idx = self.vocab.tok2idx['dynamic'][current_dynamic_idx]
      
      final_tokens = []
      total_attention_weights = []
      while True:
      # for _ in range(5):
        emb = self.decoder._get_embedding(selected_token)
        decode_out, last_hidden = self.decoder.rnn(emb.unsqueeze(0), last_hidden)
        attention_vectors, attention_weight = get_attention_vector(encode_out, decode_out)
        combined_value = torch.cat([decode_out, attention_vectors], dim=-1)
        
        selected_token = self.decoder.sampling_process(combined_value, not_triplet_vocab_list)
        selected_token = torch.cat([selected_token, torch.LongTensor([[measure_offset_idx, dynamic_idx]])], dim=1)
        selected_token = torch.cat([torch.LongTensor([[part_idx]]), selected_token], dim=-1)
        
        if selected_token[0, 1]==self.vocab.tok2idx['pitch']['end'] or selected_token[0, 2]==self.vocab.tok2idx['duration']['end']:
          break
        
        current_dur = self.vocab.vocab['duration'][selected_token[0, 2]]
        if current_dur == 'start':
          current_dur = 0.0
        current_beat += current_dur
        if current_beat >= measure_duration:
          current_beat = current_beat - measure_duration
          
        measure_offset_idx = self.vocab.tok2idx['offset'][current_beat]
        dynamic = get_dynamic(current_beat, part_idx)
        dynamic_idx = self.vocab.tok2idx['dynamic'][dynamic]
        final_tokens.append(selected_token)
        total_attention_weights.append(attention_weight[0,:,0])
      attention_map = torch.stack(total_attention_weights, dim=1)
        # print(selected_token)
      # print(self.converter(src[1:-1]))

      # while True:
      #   selected_token, last_hidden = self.decoder.forward_one_step(selected_token, last_hidden)
      #   selected_token = torch.cat([selected_token, torch.LongTensor([[measure_idx, beat_idx]])], dim=1)
      #   # print(selected_token)
      #   current_dur = vocab.vocab['duration'][selected_token[0, 1]]
      #   # print(current_beat, current_dur)
      #   current_beat += current_dur
      #   if current_beat >= measure_duration:
      #     current_measure_idx += 1
      #     current_beat = current_beat - measure_duration
      #   measure_idx = vocab.tok2idx['measure'][current_measure_idx]  
      #   beat_idx = vocab.tok2idx['beat'][current_beat]
       
      #   if selected_token[0, 0]==vocab.tok2idx['pitch']['end'] or selected_token[0, 1]==vocab.tok2idx['duration']['end']:
      #     break

      #   final_tokens.append(selected_token)
    return self.converter(src[1:-1]), self.converter(torch.cat(final_tokens, dim=0)), attention_map

class AttnDecoder(nn.Module):
  def __init__(self, vocab_size_dict: dict, param):
    super().__init__()
    self.param = param
    self.vocab_size = [x for x in vocab_size_dict.values()] 
    self.vocab_size_dict = vocab_size_dict
    self._make_embedding_layer()
    self.rnn = nn.GRU(self.param.emb.total_size, self.param.gru.hidden_size, num_layers=self.param.gru.num_layers, batch_first=True)
    self.hidden_size = self.param.gru.hidden_size
    self.num_layers = self.param.gru.num_layers
    self._make_projection_layer()
  
  def _make_embedding_layer(self):
    # self.emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.net_param.nn_params.emb.emb_size)
    self.emb = MultiEmbedding(self.vocab_size_dict, self.param.emb)
    
  def _make_projection_layer(self):
    # total_vocab_size = sum([x for x in self.vocab_size_dict.values()])
    # self.proj = nn.Linear(self.hidden_size, self.vocab_size[1] + self.vocab_size[2])
    self.proj = nn.Linear(self.hidden_size*2, self.vocab_size[1] + self.vocab_size[2])

  def _get_embedding(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
    else:
      emb = self.emb(input_seq)
    return emb
  
  def _apply_softmax(self, logit):
    pitch_output = torch.softmax(logit[...,:self.vocab_size[1]], dim=-1)
    # print(logit.shape)
    dur_output = torch.softmax(logit[...,self.vocab_size[1] :], dim=-1)
    # print(f"{pitch_output.shape, dur_output.shape}!!")
    return torch.cat([pitch_output, dur_output], dim=-1) 
  
  def run_decoder(self, x, enc_output):
    if isinstance(x, PackedSequence):
      emb = self._get_embedding(x)
      out, last_hidden = self.rnn(emb, enc_output)
    return out, last_hidden

  def forward(self, src, tgt, enc_hidden_state_by_t, enc_out):
    dec_hidden_state_by_t, dec_out = self.run_decoder(tgt, enc_out)

    mask = pad_packed_sequence(src, batch_first=True)[0][... , 1:2] != 0
    # print(mask.shape)
    encoder_hidden_states, source_lens = pad_packed_sequence(enc_hidden_state_by_t, batch_first=True)
    decoder_hidden_states, target_lens = pad_packed_sequence(dec_hidden_state_by_t, batch_first=True)
    # print(encoder_hidden_states.shape, decoder_hidden_states.shape) # [N, Ts, C], [N, Tt, C] => [N, Ts, Tt]
    # torch.Size([64, 80, 128]) torch.Size([64, 79, 128])
    
    attention_score = torch.bmm(encoder_hidden_states, decoder_hidden_states.transpose(1, 2))
    attention_score = attention_score.masked_fill(mask == 0, -float("inf"))
    attention_weight = torch.softmax(attention_score, dim=1)
    attention_vec = torch.bmm(attention_weight.transpose(1, 2), encoder_hidden_states)
    
    
    # attention_weight, attention_vec = get_attention_vector(encoder_hidden_states, decoder_hidden_states, mask)
    
    dec_output = torch.cat([decoder_hidden_states, attention_vec], dim=-1)
    # print('pass')
    logit = self.proj(dec_output)
    
    prob = self._apply_softmax(logit)
    prob = pack_padded_sequence(prob, target_lens, batch_first=True, enforce_sorted=False)
    # prob = PackedSequence(prob, tgt[1], tgt[2], tgt[3])
    return prob, attention_weight

  # def forward(self, x, enc_output):
  #   if isinstance(x, PackedSequence):
  #     out, _ = self.run_decoder(x, enc_output)
  #     # out = pad_packed_sequence(out, batch_first=True)[0]
  #     logit = self.proj(out.data)
  #     prob = self._apply_softmax(logit)
  #     prob = PackedSequence(prob, x[1], x[2], x[3])
  #   return prob
  
  def _select_token(self, prob: torch.Tensor):
    tokens = []
    # print(prob.shape)
    pitch_token = prob[0, :, :self.vocab_size[1]].multinomial(num_samples=1)
    dur_token = prob[0, :, self.vocab_size[1]:].multinomial(num_samples=1)
    # dur_token = torch.argmax(prob.data[:, model.vocab_size[0]:], dim=-1)
    return torch.cat([pitch_token, dur_token], dim=-1)
  
  def sampling_process(self, out, not_triplet_vocab_list=None): # 처음에는 encoder_output = last_hidden
    # out : 3차원 tensor
    logit = self.proj(out) 
    # print(logit.shape)
    if not_triplet_vocab_list != None:
      # print(f'model.py line 247: logit.shape={logit.shape}, vocab_size={self.vocab_size}')
      not_triplet_vocab_list = [i+self.vocab_size[1] for i in not_triplet_vocab_list]
      logit[..., not_triplet_vocab_list] = -1e10
      
    prob = self._apply_softmax(logit)
    selected_token = self._select_token(prob)
    return selected_token
    

  
def get_attention_vector(encoder_hidden_states, decoder_hidden_states, mask=None):
  
  is_packed = isinstance(encoder_hidden_states, PackedSequence)
  if is_packed:
    encoder_hidden_states, source_lens = pad_packed_sequence(encoder_hidden_states, batch_first=True)
    decoder_hidden_states, target_lens = pad_packed_sequence(decoder_hidden_states, batch_first=True)
  
  attention_score = torch.bmm(encoder_hidden_states, decoder_hidden_states.transpose(1, 2))
  if mask is not None:
    attention_score = attention_score.masked_fill(mask == 0, -float("inf"))
  attention_weight = torch.softmax(attention_score, dim=1)
  attention_vectors = torch.bmm(attention_weight.transpose(1, 2), encoder_hidden_states)
  
  if is_packed:
    attention_vectors = pack_padded_sequence(attention_vectors, target_lens, batch_first=True, enforce_sorted=False)
    
  return attention_vectors, attention_weight 


class QkvAttnSeq2seq(AttentionSeq2seq):
  def __init__(self, vocab, config):
    super().__init__(vocab, config)
    self.decoder = QkvAttnDecoder(self.vocab_size_dict, self.config)  
  


  def _run_inference_on_step(self, emb, last_hidden, encode_out):
    decode_out, last_hidden = self.decoder.rnn(emb.unsqueeze(0), last_hidden)
    attention_vectors, attention_weight = self._get_attention_vector(encode_out, decode_out)
    combined_value = torch.cat([decode_out, attention_vectors], dim=-1)
    
    return combined_value, last_hidden , attention_weight

  def _apply_prev_generation(self, prev_generation, final_tokens, last_hidden, encode_out):
    emb = self.decoder._get_embedding(prev_generation)
    _, last_hidden = self.decoder.rnn(emb.unsqueeze(0), last_hidden)
    dev = emb.device
    start_token = torch.cat([prev_generation[-1:, :3].to(dev),  torch.LongTensor([[3, 3, 5]]).to(dev)], dim=-1)
    final_tokens.append(start_token)
    current_measure_idx = 2

    return final_tokens, current_measure_idx, last_hidden
  


  @torch.inference_mode()
  def inference(self, src, part_idx):
    dev = src.device
    encode_out, encode_last_hidden = self.encoder(src.unsqueeze(0))
    # Setup for 0th step
    # start_token = torch.LongTensor([[part_idx, 1, 1, 1, 1]]) # start token idx is 1
    start_token = self.get_start_token(part_idx).to(dev)

    assert src.ndim == 2 # sequence length, feature length
    #-------------
    not_triplet_vocab_list = [ i for i, dur in enumerate(self.tokenizer.vocab['duration'][3:], 3) if (dur%0.5) != 0.0]
    #-------------
    measure_duration = self.tokenizer.measure_duration[part_idx]
    
    last_hidden = encode_last_hidden
    if last_hidden.shape[-1] != self.decoder.gru_size:
      last_hidden = self.decoder.reduce_enc(last_hidden)
    selected_token = start_token
    current_beat = Fraction(0, 1)
    measure_changed = 1
    current_measure_idx = 0
    
    condition_features = self.encode_condition_token(part_idx, current_beat, current_measure_idx, measure_changed)

    
    final_tokens = []
    total_attention_weights = []
    while True:
    # for _ in range(5):
      emb = self.decoder._get_embedding(selected_token)
      decode_out, last_hidden = self.decoder.rnn(emb.unsqueeze(0), last_hidden)
      attention_vectors, attention_weight = self._get_attention_vector(encode_out, decode_out)
      combined_value = torch.cat([decode_out, attention_vectors], dim=-1)
      selected_token = self.decoder.sampling_process(combined_value, not_triplet_vocab_list)
      
      selected_token = torch.cat([selected_token, torch.LongTensor([condition_features]).to(dev)], dim=1)
      selected_token = torch.cat([torch.LongTensor([[part_idx]]).to(dev), selected_token], dim=-1)
      
      if selected_token[0, 1]==self.tokenizer.tok2idx['pitch']['end'] or selected_token[0, 2]==self.tokenizer.tok2idx['duration']['end']:
        break
      
      current_dur = self.tokenizer.vocab['duration'][selected_token[0, 2]]
      current_beat, current_measure_idx, measure_changed = self.update_condition_info(current_beat, current_measure_idx, measure_duration, current_dur)
      if 'measure_idx' in self.tokenizer.tok2idx and current_measure_idx > self.tokenizer.vocab['measure_idx'][-1]:
        break      
      condition_features = self.encode_condition_token(part_idx, current_beat, current_measure_idx, measure_changed)

      final_tokens.append(selected_token)
      total_attention_weights.append(attention_weight[0,:,0])
    attention_map = torch.stack(total_attention_weights, dim=1)

    return self._decode_inference_result(src, torch.cat(final_tokens, dim=0), attention_map)

  @torch.inference_mode()
  def shifted_inference(self, src, part_idx, prev_generation=None, fix_first_beat=False, compensate_beat=(0.0, 0.0)):
    dev = src.device
    encode_out, encode_last_hidden = self.encoder(src.unsqueeze(0)) # 

    # Setup for 0th step
    # start_token = torch.LongTensor([[part_idx, 1, 1, 3, 3, 4]]) # start token idx is 1
    start_token = self.get_start_token(part_idx).to(dev)
    assert src.ndim == 2 # sequence length, feature length
    #-------------
    not_triplet_vocab_list = [ i for i, dur in enumerate(self.tokenizer.vocab['duration'][3:], 3) if (dur%0.5) != 0.0]
    #-------------
    measure_duration = self._get_measure_duration(part_idx)
    current_measure_idx = 0
    measure_changed = 1

    last_hidden = encode_last_hidden
    if last_hidden.shape[-1] != self.decoder.gru_size:
      last_hidden = self.decoder.reduce_enc(last_hidden)

    final_tokens = []
    if prev_generation is not None:
      final_tokens, current_measure_idx, last_hidden = self._apply_prev_generation(prev_generation, final_tokens, last_hidden, encode_out)
      # emb = self.decoder._get_embedding(prev_generation)
      # decode_out, last_hidden = self.decoder.rnn(emb.unsqueeze(0), last_hidden)
      # start_token = torch.cat([prev_generation[-1:, :3].to(dev),  torch.LongTensor([[3, 3, 5]]).to(dev)], dim=-1)
      # final_tokens.append(start_token)
      # current_measure_idx = 2
    selected_token = start_token
    current_beat = Fraction(0, 1)

    total_attention_weights = []
    # while True:
    triplet_sum = 0 
    condition_tokens = self.encode_condition_token(part_idx, current_beat, current_measure_idx, measure_changed)

    for i in range(300):
      emb = self.decoder._get_embedding(selected_token)
      # decode_out, last_hidden = self.decoder.rnn(emb.unsqueeze(0), last_hidden)
      # attention_vectors, attention_weight = self._get_attention_vector(encode_out, decode_out)
      # combined_value = torch.cat([decode_out, attention_vectors], dim=-1)
      combined_value, last_hidden, attention_weight = self._run_inference_on_step(emb, last_hidden, encode_out)
      selected_token = self.decoder.sampling_process(combined_value, not_triplet_vocab_list)
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
      if Fraction(current_dur).limit_denominator(3).denominator == 3:
        triplet_sum += Fraction(current_dur).limit_denominator(3).numerator
      elif triplet_sum != 0:
        # print("Triplet has to appear but not", current_dur,)
        triplet_sum += 1
        # print(current_dur, selected_token)
        current_dur = Fraction(1,3)
        selected_token = torch.LongTensor([[selected_token[0,0], self.tokenizer.tok2idx['duration'][current_dur]]]).to(dev)
        # print(current_dur, selected_token)
      else:
        triplet_sum = 0
      if triplet_sum == 3:
        triplet_sum = 0
      current_beat, current_measure_idx, measure_changed = self.update_condition_info(current_beat, current_measure_idx, measure_duration, current_dur)
      if 'measure_idx' in self.tokenizer.tok2idx and current_measure_idx > self.tokenizer.vocab['measure_idx'][-1]:
        break
      if float(current_beat) == 10.0:
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
    cat_out = torch.cat(final_tokens, dim=0)
    newly_generated = cat_out.clone()
    if prev_generation is not None:
      cat_out = torch.cat([prev_generation[1:], cat_out], dim=0)
    #  self.converter(src[1:-1]), self.converter(torch.cat(final_tokens, dim=0)), attention_map

    return self._decode_inference_result(src, cat_out, (attention_map, cat_out, newly_generated))


  def _get_attention_vector(self, encoder_hidden_states, decoder_hidden_states, mask=None):
    
    is_packed = isinstance(encoder_hidden_states, PackedSequence)
    if is_packed:
      encoder_hidden_states, source_lens = pad_packed_sequence(encoder_hidden_states, batch_first=True)
      decoder_hidden_states, target_lens = pad_packed_sequence(decoder_hidden_states, batch_first=True)
    # print(decoder_hidden_states.shape, encoder_hidden_states.shape)

    query = self.decoder.query(decoder_hidden_states)
    key = self.decoder.key(encoder_hidden_states)
    value = self.decoder.value(encoder_hidden_states)

    attention_score = torch.bmm(key, query.transpose(1, 2))
    if mask is not None:
      attention_score = attention_score.masked_fill(mask == 0, -float("inf"))
    attention_weight = torch.softmax(attention_score, dim=1)
    attention_vectors = torch.bmm(attention_weight.transpose(1, 2), value)
    
    if is_packed:
      attention_vectors = pack_padded_sequence(attention_vectors, target_lens, batch_first=True, enforce_sorted=False)
      
    return attention_vectors, attention_weight 


class QkvAttnSeq2seqMore(QkvAttnSeq2seq):
  def __init__(self, vocab, config):
    super().__init__(vocab, config)
    self.encoder = EncoderWithDropout(self.vocab_size_dict, self.config)
    self.decoder = QkvAttnDecoderMore(self.vocab_size_dict, self.config)

  def _run_inference_on_step(self, emb, last_hidden, encode_out):
    if not isinstance(last_hidden, list):
      last_hidden = [last_hidden, None]
    decode_out, last_hidden[0] = self.decoder.rnn(emb.unsqueeze(0), last_hidden[0])
    attention_vectors, attention_weight = self._get_attention_vector(encode_out, decode_out)
    combined_value = torch.cat([decode_out, attention_vectors], dim=-1)
    combined_value, last_hidden[1] = self.decoder.final_rnn(combined_value, last_hidden[1])
    
    return combined_value, last_hidden, attention_weight

  def _apply_prev_generation(self, prev_generation, final_tokens, last_hidden, encode_out):
    emb = self.decoder._get_embedding(prev_generation)
    decode_out, last_hidden = self.decoder.rnn(emb.unsqueeze(0), last_hidden)
    attention_vectors, attention_weight = self._get_attention_vector(encode_out, decode_out)
    combined_value = torch.cat([decode_out, attention_vectors], dim=-1)
    combined_value, last_hidden2 = self.decoder.final_rnn(combined_value)
    
    dev = emb.device
    
    
    start_token = torch.cat([prev_generation[-1:, :3].to(dev),  torch.LongTensor([[3, 3, 5]]).to(dev)], dim=-1)
    final_tokens.append(start_token)
    current_measure_idx = 2

    return final_tokens, current_measure_idx, [last_hidden, last_hidden2]


# class QkvAttnSeq2seqOrch(QkvAttnSeq2seq):
#   def __init__(self, enc_tokenizer, dec_tokenizer, config):
#     super().__init__(dec_tokenizer, config)
#     self.encoder = Encoder(enc_tokenizer.vocab_size_dict, config)  
#     self.enc_converter = Converter(enc_tokenizer)
#     self.dynamic_template = {x/2: 'weak' for x in range(0, 90, 3)}
#     self.dynamic_template[0.0] =  'strong'
#     self.dynamic_template[15.0] =  'strong'
#     self.dynamic_template[9.0] =  'middle'
#     self.dynamic_template[21.0] =  'middle'


class QkvAttnSeq2seqOrch(QkvAttnSeq2seqMore):
  def __init__(self, enc_tokenizer, dec_tokenizer, config):
    super().__init__(dec_tokenizer, config)
    self.encoder = EncoderWithDropout(enc_tokenizer.vocab_size_dict, config)  
    self.enc_converter = Converter(enc_tokenizer)
    self.dynamic_template = {x/2: 'weak' for x in range(0, 90, 3)}
    self.dynamic_template[0.0] =  'strong'
    self.dynamic_template[15.0] =  'strong'
    self.dynamic_template[9.0] =  'middle'
    self.dynamic_template[21.0] =  'middle'
  
  def _decode_inference_result(self, src, output, other_out):
    return self.enc_converter(src[1:-1]), self.converter(output), other_out
  
  def _get_measure_duration(self, part_idx):
    return 30.0 # Currently fixed to 30
  
  def update_condition_info(self, current_beat, current_measure_idx, measure_duration, current_dur):
    if current_dur in ('start', 'pad'):
      current_dur = 0.0
    current_beat += current_dur
    if float(current_beat) >= measure_duration:
      current_measure_idx += 1 #마디 수 늘려주기
      measure_changed = 1
      current_beat = current_beat - measure_duration
    else:
      measure_changed = 0

    return current_beat, current_measure_idx, measure_changed
  
  def encode_condition_token(self, part_idx, current_beat, current_measure_idx, measure_start):
    condition_features = []
    if 'offset' in self.tokenizer.tok2idx:
      measure_offset_idx = self.tokenizer.tok2idx['offset'][float(current_beat)]
      condition_features.append(measure_offset_idx)
    if 'dynamic' in self.tokenizer.tok2idx:
      dynamic = self.dynamic_template.get(float(current_beat), 'none')
      dynamic_idx = self.tokenizer.tok2idx['dynamic'][dynamic]
      condition_features.append(dynamic_idx)
    if 'measure_idx' in self.tokenizer.tok2idx:
      measure_idx = self.tokenizer.tok2idx['measure_idx'][current_measure_idx]
      condition_features.append(measure_idx)
    if 'measure_change' in self.tokenizer.tok2idx:
      measure_start_idx = self.tokenizer.tok2idx['measure_change'][measure_start]
      condition_features.append(measure_start_idx)
    return condition_features


  @torch.inference_mode()
  def shifted_inference(self, src, part_idx, prev_generation=None):
    dev = src.device
    encode_out, encode_last_hidden = self.encoder(src.unsqueeze(0)) # 

    # Setup for 0th step
    # start_token = torch.LongTensor([[part_idx, 1, 1, 3, 3, 4]]) # start token idx is 1
    start_token = self.get_start_token(part_idx).to(dev)
    assert src.ndim == 2 # sequence length, feature length
    measure_duration = self._get_measure_duration(part_idx)
    current_measure_idx = 0
    measure_changed = 1

    last_hidden = encode_last_hidden
    if last_hidden.shape[-1] != self.decoder.gru_size:
      last_hidden = self.decoder.reduce_enc(last_hidden)

    final_tokens = []
    if prev_generation is not None:
      final_tokens, current_measure_idx, last_hidden = self._apply_prev_generation(prev_generation, final_tokens, last_hidden, encode_out)
    selected_token = start_token
    current_beat = 0.0

    total_attention_weights = []
    # while True:
    for i in range(500):
      # print("selected token is", selected_token)
      emb = self.decoder._get_embedding(selected_token)
      combined_value, last_hidden, attention_weight = self._run_inference_on_step(emb, last_hidden, encode_out)
      selected_token = self.decoder.sampling_process(combined_value)

      if selected_token[0, 0]==self.tokenizer.tok2idx['pitch']['end'] or selected_token[0, 1]==self.tokenizer.tok2idx['duration']['end']:
        break
      
      # update beat position and beat strength
      current_dur = self.tokenizer.vocab['duration'][selected_token[0, 1]]
      if current_dur + current_beat > measure_duration:
        if measure_duration - current_beat in self.tokenizer.tok2idx['duration']:
          selected_token[0, 1] = self.tokenizer.tok2idx['duration'][measure_duration - current_beat]
          current_dur = measure_duration - current_beat
        elif measure_duration - current_beat > 3.0:
          selected_token[0, 1] = self.tokenizer.tok2idx['duration'][3.0]
          current_dur = 3.0
        else:
          selected_token[0, 1] = self.tokenizer.tok2idx['duration'][0.5]
          current_dur = 0.5
      current_beat, current_measure_idx, measure_changed = self.update_condition_info(current_beat, current_measure_idx, measure_duration, current_dur)
      if 'measure_idx' in self.tokenizer.tok2idx and current_measure_idx > self.tokenizer.vocab['measure_idx'][-1]:
        break
      condition_tokens = self.encode_condition_token(part_idx, current_beat, current_measure_idx, measure_changed)
      
      # make new token for next rnn timestep
      selected_token = torch.cat([torch.LongTensor([[part_idx]]).to(dev), selected_token, torch.LongTensor([condition_tokens]).to(dev)], dim=1)
      
      final_tokens.append(selected_token)
      total_attention_weights.append(attention_weight[0,:,0])
    if len(total_attention_weights) == 0:
      attention_map = None
    else:
      attention_map = torch.stack(total_attention_weights, dim=1)
    cat_out = torch.cat(final_tokens, dim=0)
    newly_generated = cat_out.clone()
    if prev_generation is not None:
      cat_out = torch.cat([prev_generation[1:], cat_out], dim=0)
    #  self.converter(src[1:-1]), self.converter(torch.cat(final_tokens, dim=0)), attention_map

    return self._decode_inference_result(src, cat_out, (attention_map, cat_out, newly_generated))


class QkvAttnDecoder(nn.Module):
  def __init__(self, vocab_size_dict: dict, param):
    super().__init__()
    self.param = param
    self.vocab_size = [x for x in vocab_size_dict.values()] 
    self.vocab_size_dict = vocab_size_dict
    self._make_embedding_layer()
    self.gru_size = self.param.gru.hidden_size
    if self.param.gru.hidden_size != self.gru_size:
      self.reduce_enc = nn.Linear(self.param.gru.hidden_size, self.gru_size)
    
    self.rnn = nn.GRU(self.param.emb.total_size, self.gru_size, num_layers=self.param.gru.num_layers, batch_first=True)
    self.hidden_size = self.param.gru.hidden_size
    self.num_layers = self.param.gru.num_layers
    self._make_projection_layer()
    self.query = nn.Linear(self.gru_size, self.hidden_size)
    self.key = nn.Linear(self.hidden_size, self.hidden_size)
    self.value = nn.Linear(self.hidden_size, self.hidden_size)
  
  def _make_embedding_layer(self):
    self.emb = MultiEmbedding(self.vocab_size_dict, self.param.emb)
    
  def _make_projection_layer(self):
    # total_vocab_size = sum([x for x in self.vocab_size_dict.values()])
    # self.proj = nn.Linear(self.hidden_size, self.vocab_size[1] + self.vocab_size[2])
    self.proj = nn.Linear(self.hidden_size + self.gru_size, self.vocab_size[1] + self.vocab_size[2])

  def _get_embedding(self, input_seq):
    if isinstance(input_seq, PackedSequence):
      emb = PackedSequence(self.emb(input_seq[0]), input_seq[1], input_seq[2], input_seq[3])
    else:
      emb = self.emb(input_seq)
    return emb
  
  def _apply_softmax(self, logit):
    pitch_output = torch.softmax(logit[...,:self.vocab_size[1]], dim=-1)
    # print(logit.shape)
    dur_output = torch.softmax(logit[...,self.vocab_size[1] :], dim=-1)
    # print(f"{pitch_output.shape, dur_output.shape}!!")
    return torch.cat([pitch_output, dur_output], dim=-1) 
  
  def run_decoder(self, x, enc_output):
    if isinstance(x, PackedSequence):
      emb = self._get_embedding(x)
      out, last_hidden = self.rnn(emb, enc_output)
    return out, last_hidden

  def forward(self, src, tgt, enc_hidden_state_by_t, enc_out):
    if enc_out.shape[-1] != self.gru_size:
      enc_out = self.reduce_enc(enc_out)
    dec_hidden_state_by_t, dec_out = self.run_decoder(tgt, enc_out)
    mask = pad_packed_sequence(src, batch_first=True)[0][... , 1:2] != 0

    encoder_hidden_states, source_lens = pad_packed_sequence(enc_hidden_state_by_t, batch_first=True)
    decoder_hidden_states, target_lens = pad_packed_sequence(dec_hidden_state_by_t, batch_first=True)

    query = self.query(decoder_hidden_states)
    # print(query.shape)
    key = self.key(encoder_hidden_states)
    value = self.value(encoder_hidden_states)
    
    attention_score = torch.bmm(key, query.transpose(1, 2))
    attention_score = attention_score.masked_fill(mask == 0, -float("inf"))
    attention_weight = torch.softmax(attention_score, dim=1)
    attention_vec = torch.bmm(attention_weight.transpose(1, 2), value)

    dec_output = torch.cat([decoder_hidden_states, attention_vec], dim=-1)

    logit = self.proj(dec_output)

    prob = self._apply_softmax(logit)
    prob = pack_padded_sequence(prob, target_lens, batch_first=True, enforce_sorted=False)
    return prob, attention_weight

  def _select_token(self, prob: torch.Tensor, top_p=0.7):
    tokens = []
    # print(prob.shape)
    pitch_prob = prob[0, :, :self.vocab_size[1]]
    dur_prob = prob[0, :, self.vocab_size[1]:]
    if top_p != 1.0:
      pitch_token = nucleus(pitch_prob, top_p)
      dur_token = nucleus(dur_prob, top_p)
    else:
      pitch_token = pitch_prob.multinomial(num_samples=1)
      dur_token = dur_prob.multinomial(num_samples=1)
    # pitch_token = torch.argmax(prob[0, :, :self.vocab_size[1]], dim=-1, keepdim=True)
    # dur_token = torch.argmax(prob[0, :, self.vocab_size[1]:], dim=-1, keepdim=True)
    return torch.cat([pitch_token, dur_token], dim=-1)
  
  def sampling_process(self, out, not_triplet_vocab_list=None): # 처음에는 encoder_output = last_hidden
    # out : 3차원 tensor
    logit = self.proj(out) 
    # print(logit.shape)
    if not_triplet_vocab_list != None:
      # print(f'model.py line 247: logit.shape={logit.shape}, vocab_size={self.vocab_size}')
      not_triplet_vocab_list = [i+self.vocab_size[1] for i in not_triplet_vocab_list]
      # logit[..., not_triplet_vocab_list] = -1e10
      
    prob = self._apply_softmax(logit)
    selected_token = self._select_token(prob)
    return selected_token

class QkvAttnDecoderMore(QkvAttnDecoder):
  def __init__(self, vocab_size_dict: dict, param):
    super().__init__(vocab_size_dict, param)
    self.final_rnn = nn.GRU(self.hidden_size + self.gru_size, self.hidden_size, batch_first=True)
    self.emb_dropout = PackedDropout(param.emb_dropout)

  def _make_projection_layer(self):
    self.proj = nn.Linear(self.hidden_size, self.vocab_size[1] + self.vocab_size[2])

  def _get_embedding(self, input_seq):
    x = super()._get_embedding(input_seq)
    return self.emb_dropout(x)

  def forward(self, src, tgt, enc_hidden_state_by_t, enc_out):
    if enc_out.shape[-1] != self.gru_size:
      enc_out = self.reduce_enc(enc_out)
    dec_hidden_state_by_t, dec_out = self.run_decoder(tgt, enc_out)
    mask = pad_packed_sequence(src, batch_first=True)[0][... , 1:2] != 0

    encoder_hidden_states, source_lens = pad_packed_sequence(enc_hidden_state_by_t, batch_first=True)
    decoder_hidden_states, target_lens = pad_packed_sequence(dec_hidden_state_by_t, batch_first=True)

    query = self.query(decoder_hidden_states)
    # print(query.shape)
    key = self.key(encoder_hidden_states)
    value = self.value(encoder_hidden_states)
    
    attention_score = torch.bmm(key, query.transpose(1, 2))
    attention_score = attention_score.masked_fill(mask == 0, -float("inf"))
    attention_weight = torch.softmax(attention_score, dim=1)
    attention_vec = torch.bmm(attention_weight.transpose(1, 2), value)

    dec_output = torch.cat([decoder_hidden_states, attention_vec], dim=-1)
    dec_output = self.emb_dropout(dec_output)  

    dec_output, _ = self.final_rnn(dec_output)
    dec_output = self.emb_dropout(dec_output)

    logit = self.proj(dec_output)

    prob = self._apply_softmax(logit)
    prob = pack_padded_sequence(prob, target_lens, batch_first=True, enforce_sorted=False)
    return prob, attention_weight

class QkvAttnDecoderMoreAttn(QkvAttnDecoder):
  def __init__(self, vocab_size_dict: dict, param):
    super().__init__(vocab_size_dict, param)
    self.final_rnn = nn.GRU(self.hidden_size + self.gru_size, self.hidden_size, batch_first=True)
    self.emb_dropout = PackedDropout(param.emb_dropout)
    
    self.query2 = nn.Linear(self.gru_size, self.hidden_size)
    self.key2 = nn.Linear(self.hidden_size, self.hidden_size)
    self.value2 = nn.Linear(self.hidden_size, self.hidden_size)
    
    self.final_rnn2 = nn.GRU(self.hidden_size + self.gru_size, self.hidden_size, batch_first=True)

  def forward(self, src, tgt, enc_hidden_state_by_t, enc_out):
    if enc_out.shape[-1] != self.gru_size:
      enc_out = self.reduce_enc(enc_out)
    dec_hidden_state_by_t, dec_out = self.run_decoder(tgt, enc_out)
    mask = pad_packed_sequence(src, batch_first=True)[0][... , 1:2] != 0

    encoder_hidden_states, source_lens = pad_packed_sequence(enc_hidden_state_by_t, batch_first=True)
    decoder_hidden_states, target_lens = pad_packed_sequence(dec_hidden_state_by_t, batch_first=True)

    query = self.query(decoder_hidden_states)
    # print(query.shape)
    key = self.key(encoder_hidden_states)
    value = self.value(encoder_hidden_states)
    
    attention_score = torch.bmm(key, query.transpose(1, 2))
    attention_score = attention_score.masked_fill(mask == 0, -float("inf"))
    attention_weight = torch.softmax(attention_score, dim=1)
    attention_vec = torch.bmm(attention_weight.transpose(1, 2), value)

    dec_output = torch.cat([decoder_hidden_states, attention_vec], dim=-1)
    dec_output = self.emb_dropout(dec_output)  

    dec_output, _ = self.final_rnn(dec_output)
    dec_output = self.emb_dropout(dec_output)
    
    
    query = self.query2(dec_output)
    key = self.key2(encoder_hidden_states)
    value = self.value2(encoder_hidden_states)

    attention_score = torch.bmm(key, query.transpose(1, 2))
    attention_score = attention_score.masked_fill(mask == 0, -float("inf"))
    attention_weight = torch.softmax(attention_score, dim=1)
    attention_vec = torch.bmm(attention_weight.transpose(1, 2), value)

    dec_output = torch.cat([dec_output, attention_vec], dim=-1)
    dec_output = self.emb_dropout(dec_output)  

    dec_output, _ = self.final_rnn2(dec_output)
    dec_output = self.emb_dropout(dec_output)

    logit = self.proj(dec_output)

    prob = self._apply_softmax(logit)
    prob = pack_padded_sequence(prob, target_lens, batch_first=True, enforce_sorted=False)
    return prob, attention_weight



class QkvRollSeq2seq(QkvAttnSeq2seq):
  def __init__(self, vocab, config):
    super().__init__(vocab, config)    
    self.encoder.return_reduced_bidirectional = False
    if hasattr(config, 'sampling_rate'):
      self.sampling_rate = config.sampling_rate 

    self.converter = RollConverter(self.tokenizer, self.sampling_rate)
    self.decoder = QkvRollDecoder(self.vocab_size_dict, self.config)
    self.dynamic_template_list = make_dynamic_template(beat_sampling_num=self.sampling_rate)
    self.dynamic_template_list = [convert_dynamics_to_integer(d) for d in self.dynamic_template_list]

  def forward(self, src, dec_input):
    enc_out, enc_last_hidden = self.encoder(src)
    dec_out = self.decoder(src, dec_input, enc_out, enc_last_hidden)
    return dec_out
  
  def make_dec_input(self, part_idx, measure_len):
    dynamic_template = self.dynamic_template_list[part_idx]
    num_frames = len(dynamic_template) * measure_len

    output = np.zeros((num_frames, 4), dtype=np.int32)
    output[:, 0] = part_idx
    output[:, 1] = 0 # add dummy note
    output[:, 2] = dynamic_template * measure_len
    output[:, 3] = np.arange(num_frames)

    output = torch.LongTensor([self.tokenizer(frame)  for frame in output] )
    return output[:, [0, 2, 3]]
    

  @torch.inference_mode()
  def inference(self, src, target_idx):
    dev = src.device
    encode_out, enc_last_hidden = self.encoder(src.unsqueeze(0)) 
    src_part_idx = int(src[0][0])

    measure_len = int(src.shape[0] / self.sampling_rate / MEAS_LEN_BY_IDX[src_part_idx])
    dec_input = self.make_dec_input(target_idx, measure_len).to(dev)
    dec_input = self.decoder._get_embedding(dec_input)
    decode_out, last_hidden = self.decoder.rnn(dec_input.unsqueeze(0), enc_last_hidden)
    attention_vectors, attention_weight = self._get_attention_vector(encode_out, decode_out)
    combined_value = torch.cat([decode_out, attention_vectors], dim=-1)
    combined_value = combined_value.squeeze(0)
    logit = self.decoder.proj(combined_value)
    selected_pitch = torch.argmax(logit, dim=-1)

    selected_token = [[target_idx, int(p)] for p in selected_pitch]

      
    return self.converter([note[:2] for note in src]), self.converter(selected_token), (attention_weight, selected_token, selected_token[-len(self.dynamic_template_list[target_idx])*2:-len(self.dynamic_template_list[target_idx])])


class QkvRollDecoder(QkvAttnDecoder):
  def __init__(self, vocab_size_dict: dict, param):
    self.param = param
    super().__init__(vocab_size_dict, param)
    self.rnn = nn.GRU(self.emb.total_size, self.hidden_size, num_layers=param.gru.num_layers, batch_first=True, bidirectional=True, dropout=param.gru.dropout)
    self.query = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.key = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.value = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.dropout = nn.Dropout(param.gru.dropout)

  def _make_embedding_layer(self): # 
    vocab_size_dict = copy.copy(self.vocab_size_dict)
    vocab_size_dict.pop('pitch')
    self.emb = MultiEmbedding(vocab_size_dict, self.param.emb)


  def _apply_softmax(self, logit):
    pitch_output = torch.softmax(logit, dim=-1)
    return pitch_output

  def _make_projection_layer(self):
    # total_vocab_size = sum([x for x in self.vocab_size_dict.values()])
    self.proj = nn.Linear(self.hidden_size * 3, self.vocab_size[1]) # pitch + is_onset
    
  def forward(self, src, dec_input, enc_hidden_state_by_t, enc_out):
    if isinstance(dec_input, PackedSequence):
      emb = self._get_embedding(dec_input.data)
      emb = PackedSequence(emb, dec_input[1], dec_input[2], dec_input[3])
      dec_hidden_state_by_t, dec_out = self.rnn(emb, enc_out)
      mask = pad_packed_sequence(src, batch_first=True)[0][... , 1:2] != 0
      # out = pad_packed_sequence(out, batch_first=True)[0]
      encoder_hidden_states, source_lens = pad_packed_sequence(enc_hidden_state_by_t, batch_first=True)
      decoder_hidden_states, target_lens = pad_packed_sequence(dec_hidden_state_by_t, batch_first=True)
      query = self.query(decoder_hidden_states)
      key = self.key(encoder_hidden_states)
      value = self.value(encoder_hidden_states)
      
      attention_score = torch.bmm(key, query.transpose(1, 2))
      attention_score = attention_score.masked_fill(mask == 0, -float("inf"))
      attention_weight = torch.softmax(attention_score, dim=1)
      attention_vec = torch.bmm(attention_weight.transpose(1, 2), value)
      dec_output = torch.cat([decoder_hidden_states, attention_vec], dim=-1)
      dec_output = self.dropout(dec_output)
      
      logit = self.proj(dec_output)
      prob = self._apply_softmax(logit)
      prob = pack_padded_sequence(prob, target_lens, batch_first=True, enforce_sorted=False)
      
    else:
      raise NotImplementedError
    return prob, attention_weight



  
  #-------------Transformer -----------------#
class TransSeq2seq(Seq2seq):
  def __init__(self, tokenizer, config):
    super().__init__(tokenizer, config)
    # self.tokenizer = tokenizer
    # self.vocab_size_dict = self.tokenizer.vocab_size_dict
    
  def _make_encoder_and_decoder(self):
    self.encoder = TransEncoder(self.vocab_size_dict, self.config)
    self.decoder = TransDecoder(self.vocab_size_dict, self.config)
  
  def forward(self, src, tgt):
    enc_out, src_mask = self.encoder(src)
    dec_out = self.decoder(tgt, enc_out, src_mask)
    return dec_out, None
  
  def shifted_inference(self, src, part_idx, prev_generation=None, fix_first_beat=False, compensate_beat=(0.0, 0.0)):
    dev = src.device
    assert src.ndim == 2
    enc_out, enc_mask = self.encoder(src.unsqueeze(0)) 

    # Setup for 0th step
    # start_token = torch.LongTensor([[part_idx, 1, 1, 3, 3, 4]]) # start token idx is 1
    start_token = self.get_start_token(part_idx).to(dev)

    #-------------
    # not_triplet_vocab_list = [ i for i, dur in enumerate(self.tokenizer.vocab['duration'][3:], 3) if (dur%0.5) != 0.0]
    #-------------
    measure_duration = self._get_measure_duration(part_idx)
    current_measure_idx = 0
    measure_changed = 1

    final_tokens = [start_token]
    if prev_generation is not None:
      final_tokens, current_measure_idx, last_hidden = self._apply_prev_generation(prev_generation, final_tokens, last_hidden, enc_out)
      # emb = self.decoder._get_embedding(prev_generation)
      # decode_out, last_hidden = self.decoder.rnn(emb.unsqueeze(0), last_hidden)
      # start_token = torch.cat([prev_generation[-1:, :3].to(dev),  torch.LongTensor([[3, 3, 5]]).to(dev)], dim=-1)
      # final_tokens.append(start_token)
      # current_measure_idx = 2
    selected_token = start_token

    current_beat = Fraction(0, 1)

    total_attention_weights = []
    # while True:
    triplet_sum = 0 
    condition_tokens = self.encode_condition_token(part_idx, current_beat, current_measure_idx, measure_changed)

    for i in range(300):
      # decode_out, last_hidden = self.decoder.rnn(emb.unsqueeze(0), last_hidden)
      # attention_vectors, attention_weight = self._get_attention_vector(encode_out, decode_out)
      # combined_value = torch.cat([decode_out, attention_vectors], dim=-1)
      # combined_value, last_hidden, attention_weight = self._run_inference_on_step(emb, last_hidden, encode_out)
      prob = self.decoder(torch.cat(final_tokens, dim=0).unsqueeze(0), enc_out, enc_mask)
      selected_token = self.decoder._select_token(prob[:, -1:])
      # for rule_base fixing
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
      if Fraction(current_dur).limit_denominator(3).denominator == 3:
        triplet_sum += Fraction(current_dur).limit_denominator(3).numerator
      elif triplet_sum != 0:
        # print("Triplet has to appear but not", current_dur,)
        triplet_sum += 1
        # print(current_dur, selected_token)
        current_dur = Fraction(1,3)
        selected_token = torch.LongTensor([[selected_token[0,0], self.tokenizer.tok2idx['duration'][current_dur]]]).to(dev)
        # print(current_dur, selected_token)
      else:
        triplet_sum = 0
      if triplet_sum == 3:
        triplet_sum = 0
      current_beat, current_measure_idx, measure_changed = self.update_condition_info(current_beat, current_measure_idx, measure_duration, current_dur)
      if 'measure_idx' in self.tokenizer.tok2idx and current_measure_idx > self.tokenizer.vocab['measure_idx'][-1]:
        break
      if float(current_beat) == 10.0:
        print(f"Current Beat ==10.0 detected! cur_beat: {current_beat}, measure_dur: {measure_duration}, cur_measure_idx: {current_measure_idx}, cur_dur: {current_dur}, triplet_sum: {triplet_sum}")
      condition_tokens = self.encode_condition_token(part_idx, current_beat, current_measure_idx, measure_changed)
      # make new token for next rnn timestep
      selected_token = torch.cat([torch.LongTensor([[part_idx]]).to(dev), selected_token, torch.LongTensor([condition_tokens]).to(dev)], dim=1)
      
      final_tokens.append(selected_token)
      
    if len(total_attention_weights) == 0:
      attention_map = None
    else:
      attention_map = torch.stack(total_attention_weights, dim=1)
      
    cat_out = torch.cat(final_tokens[1:], dim=0) #token shape: [1,6] => [24,6]
    newly_generated = cat_out.clone()
    if prev_generation is not None:
      cat_out = torch.cat([prev_generation[1:], cat_out], dim=0)
        #  self.converter(src[1:-1]), self.converter(torch.cat(final_tokens, dim=0)), attention_map
    return self._decode_inference_result(src, cat_out, (attention_map, cat_out, newly_generated))




if __name__ == "__main__":
  from yeominrak_processing import AlignedScore, ShiftedAlignedScore, pack_collate, SamplingScore, CNNScore, pad_collate
  import random as random
  from loss import nll_loss, focal_loss
  from trainer import Trainer, RollTrainer, CNNTrainer
  from torch.utils.data import DataLoader
  from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pack_padded_sequence
  from omegaconf import OmegaConf

  config = OmegaConf.load('yamls/baseline.yaml')
  config = get_emb_total_size(config)


  val_dataset = ShiftedAlignedScore(is_valid = True)
  model = QkvAttnSeq2seq(val_dataset.tokenizer, config.model)

  model.load_state_dict(torch.load('best_model.pt'))
  sample = val_dataset[45][0]
  model.shifted_inference(sample, 7)
  print(model)
  '''
  # train_dataset = AlignedScore(is_valid = False)
  # val_dataset = AlignedScore(is_valid = True)
  # train_dataset = SamplingScore(slice_measure_num = 2, sample_len = 10000, is_valid = False, is_modify_pitch=False)
  # val_dataset = SamplingScore(is_valid=True)
  train_dataset = CNNScore(slice_measure_num = 2, sample_len = 10000, is_valid = False, is_modify_pitch=False)
  val_dataset = CNNScore(slice_measure_num = 2, is_valid=True)

  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=lambda x: pad_collate(x, 300), drop_last=True)  
  valid_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=lambda x: pad_collate(x, 300), drop_last=True)

  device = 'cuda'
  # model = QkvRollSeq2seq(val_dataset.tokenizer, hidden_size=32).to(device)
  # model = RollSeq2seq(val_dataset.tokenizer, hidden_size=32).to(device)
  model = CNNModel(val_dataset.tokenizer, emb_size=64, hidden_size=128).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  loss_fn = nll_loss
  # loss_fn = focal_loss
  # trainer = RollTrainer(model, optimizer, loss_fn, train_loader, valid_loader, device)
  trainer = CNNTrainer(model, optimizer, loss_fn, train_loader, valid_loader, device)
  trainer.train_by_num_epoch(20)
  '''