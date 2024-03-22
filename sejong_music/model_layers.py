from typing import List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

from .module import PackedDropout, MultiEmbedding, get_emb_total_size, Converter
from .sampling_utils import nucleus

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
