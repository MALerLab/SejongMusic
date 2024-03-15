import torch
import x_transformers
import torch.nn as nn

from .module import SumEmbedding


class TransEncoder(nn.Module):
  def __init__(self, vocab_size_dict: dict, config):
    super().__init__()
    self.param = config
    self.vocab_size = [x for x in vocab_size_dict.values()]
    self.vocab_size_dict = vocab_size_dict
    self._make_embedding_layer()
    self.layers = x_transformers.Encoder(dim=self.param.dim ,depth=self.param.depth, num_heads=self.param.num_heads)
    
  def _make_embedding_layer(self):
    self.embedding = SumEmbedding(self.vocab_size_dict, self.param.dim)
    
  def forward(self, x:torch.LongTensor):
    '''
    x: (batch, seq_len, num_features)
    '''
    mask = (x != 0)[..., 0] # squeeze num_features dimension
    embedding = self.embedding(x)
    return self.layers(embedding, mask=mask), mask
  
class TransDecoder(nn.Module):
  def __init__(self, vocab_size_dict: dict, config):
    super().__init__()
    self.vocab_size = [x for x in vocab_size_dict.values()]
    self.param = config
    self.embedding = SumEmbedding(vocab_size_dict, config.dim)
    self.layers = x_transformers.Decoder(dim=config.dim, depth=config.depth, num_heads=config.num_heads, cross_attend=True)
    self._make_projection_layer()
    
  def forward(self, x, enc_out, src_mask):
    mask = (x != 0)[..., 0]
    embedding = self.embedding(x)
    output = self.layers(embedding, context=enc_out, mask=mask, context_mask=src_mask)
    logit = self.proj(output)
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