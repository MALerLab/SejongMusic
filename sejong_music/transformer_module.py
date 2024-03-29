import torch
import x_transformers
import torch.nn as nn

from .module import SumEmbedding, SumEmbeddingSingleVocab
from x_transformers.x_transformers import AbsolutePositionalEmbedding

class TransEncoder(nn.Module):
  def __init__(self, vocab_size_dict: dict, config):
    super().__init__()
    self.param = config
    self.vocab_size = [x for x in vocab_size_dict.values()]
    self.vocab_size_dict = vocab_size_dict
    self._make_embedding_layer()
    self.layers = x_transformers.Encoder(dim=self.param.dim,
                                        depth=self.param.depth, 
                                        num_heads=self.param.num_heads,
                                        attn_dropout=config.dropout,
                                        ff_dropout = config.dropout
    )
    add_dropout_after_attn(self.layers, self.param.dropout)
    add_dropout_after_ff(self.layers, self.param.dropout)
    
    
  def _make_embedding_layer(self):
    self.embedding = SumEmbedding(self.vocab_size_dict, self.param.dim)
    
  def forward(self, x:torch.LongTensor):
    '''
    x: (batch, seq_len, num_features)
    '''
    mask = (x != 0)[..., -1] # squeeze num_features dimension
    embedding = self.embedding(x)
    return self.layers(embedding, mask=mask), mask
  
class TransDecoder(nn.Module):
  def __init__(self, vocab_size_dict: dict, config):
    super().__init__()
    self.vocab_size = [x for x in vocab_size_dict.values()]
    self.param = config
    self.embedding = SumEmbedding(vocab_size_dict, config.dim)
    self.layers = x_transformers.Decoder(dim=config.dim, 
                                        depth=config.depth, 
                                        num_heads=config.num_heads, 
                                        attn_dropout=config.dropout,
                                        ff_dropout = config.dropout,
                                        cross_attn_dropout = config.dropout,
                                        cross_attend=True)
    self._make_projection_layer()
    add_dropout_after_attn(self.layers, self.param.dropout)
    add_dropout_after_ff(self.layers, self.param.dropout)

  def forward(self, x, enc_out, src_mask, return_logits=False):
    mask = (x != 0)[..., -1]
    embedding = self.embedding(x)
    output = self.layers(embedding, context=enc_out, mask=mask, context_mask=src_mask)
    logit = self.proj(output)
    if return_logits:
      return logit
    dec_out = self._apply_softmax(logit)
    return dec_out
  
  def _make_projection_layer(self):
    # total_vocab_size = sum([x for x in self.vocab_size_dict.values()])
    self.proj = nn.Linear(self.param.dim, self.vocab_size[1] + self.vocab_size[2])

  def _apply_softmax(self, logit):
    pitch_output = torch.softmax(logit[...,:self.vocab_size[1]], dim=-1)
    # print(logit.shape)
    dur_output = torch.softmax(logit[...,self.vocab_size[1] :], dim=-1)
    # print(f"{pitch_output.shape, dur_output.shape}!!")
    return torch.cat([pitch_output, dur_output], dim=-1)
  
  def _select_token(self, prob: torch.Tensor):
    tokens = []
    # print(prob.shape)
    pitch_token = prob[0, :, :self.vocab_size[1]].multinomial(num_samples=1)
    dur_token = prob[0, :, self.vocab_size[1]:].multinomial(num_samples=1)
    # dur_token = torch.argmax(prob.data[:, model.vocab_size[0]:], dim=-1)
    return torch.cat([pitch_token, dur_token], dim=-1)


  
class JeongganTransEncoder(nn.Module):
  def __init__(self, vocab_size: int, config):
    super().__init__()
    self.param = config
    self.vocab_size = vocab_size
    self._make_embedding_layer()
    self.layers = x_transformers.Encoder(dim=self.param.dim,
                                        depth=self.param.depth, 
                                        num_heads=self.param.num_heads,
                                        attn_dropout=config.dropout,
                                        ff_dropout = config.dropout,
                                        attn_flash=True
    )
    self.encoder_pos_enc = AbsolutePositionalEmbedding(config.dim, 2000)
    add_dropout_after_attn(self.layers, self.param.dropout)
    add_dropout_after_ff(self.layers, self.param.dropout)
    
    
  def _make_embedding_layer(self):
    self.embedding = SumEmbeddingSingleVocab(self.vocab_size, self.param.dim)
    
  def forward(self, x:torch.LongTensor):
    '''
    x: (batch, seq_len, num_features)
    '''
    mask = (x != 0)[..., -1] # squeeze num_features dimension
    embedding = self.embedding(x)
    embedding += self.encoder_pos_enc(embedding)
    return self.layers(embedding, mask=mask), mask
  
class JeongganTransDecoder(nn.Module):
  def __init__(self, vocab_size: int, pred_vocab_size:int, config):
    super().__init__()
    self.param = config
    self.vocab_size = vocab_size
    self.pred_vocab_size = pred_vocab_size
    self.embedding = SumEmbeddingSingleVocab(vocab_size, config.dim)
    self.layers = x_transformers.Decoder(dim=config.dim, 
                                        depth=config.depth, 
                                        num_heads=config.num_heads, 
                                        attn_dropout=config.dropout,
                                        ff_dropout = config.dropout,
                                        cross_attn_dropout = config.dropout,
                                        cross_attend=True,
                                        attn_flash=True)
    self.decoder_pos_enc = AbsolutePositionalEmbedding(config.dim, 1000)
    self._make_projection_layer()
    add_dropout_after_attn(self.layers, self.param.dropout)
    add_dropout_after_ff(self.layers, self.param.dropout)

  def forward(self, x, enc_out, src_mask, return_logits=False):
    mask = (x != 0)[..., -1]
    embedding = self.embedding(x)
    embedding += self.decoder_pos_enc(embedding)
    output = self.layers(embedding, context=enc_out, mask=mask, context_mask=src_mask)
    logit = self.proj(output)
    if return_logits:
      return logit
    dec_out = self._apply_softmax(logit)
    return dec_out
  
  def _make_projection_layer(self):
    # total_vocab_size = sum([x for x in self.vocab_size_dict.values()])
    self.proj = nn.Linear(self.param.dim, self.pred_vocab_size)

  def _apply_softmax(self, logit):
    return torch.softmax(logit, dim=-1)
  
  def _select_token(self, prob: torch.Tensor):
    tokens = []
    # print(prob.shape)
    pitch_token = prob[0, :, :self.vocab_size[1]].multinomial(num_samples=1)
    dur_token = prob[0, :, self.vocab_size[1]:].multinomial(num_samples=1)
    # dur_token = torch.argmax(prob.data[:, model.vocab_size[0]:], dim=-1)
    return torch.cat([pitch_token, dur_token], dim=-1)
  
  
def add_dropout_after_attn(x_module:x_transformers.Encoder, dropout):
  for layer in x_module.layers:
    if 'Attention' in str(type(layer[1])): 
      if isinstance(layer[1].to_out, nn.Sequential): # if GLU
        layer[1].to_out.append(nn.Dropout(dropout))
      elif isinstance(layer[1].to_out, nn.Linear): # if simple linear
        layer[1].to_out = nn.Sequential(layer[1].to_out, nn.Dropout(dropout))
      else:
        raise ValueError('to_out should be either nn.Sequential or nn.Linear')

def add_dropout_after_ff(x_module:x_transformers.Encoder, dropout):
  for layer in x_module.layers:
    if 'FeedForward' in str(type(layer[1])):
      layer[1].ff.append(nn.Dropout(dropout))