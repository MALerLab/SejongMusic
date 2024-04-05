import random
from typing import List
import torch


class Augmentor:
  def __init__(self, 
               tokenizer, 
               pitch_tokens:List[str], 
               ornament_tokens:List[str],
               entire_orn_r:float = 0.05,
               jg_r:float = 0.15,
               frame_replace_r = 0.05, 
               frame_rand_r = 0.05,
               mask_note_r = 0.15,
               replace_note_r =0.05,
               erase_note_r = 0.05,
               **kwargs):
    self.tokenizer = tokenizer
    self.mask_id = tokenizer.tok2idx['mask']
    self.sus_id = tokenizer.tok2idx['-']
    self.pitch_ids:List[int] = sorted(tokenizer(pitch_tokens))
    self.min_pitch = self.pitch_ids[0]
    self.max_pitch = self.pitch_ids[-1]
    self.ornament_ids:List[int] = sorted(tokenizer(ornament_tokens))
    self.num_beat = sum([1 for x in tokenizer.vocab if 'beat' in x])
    
    self.min_beat = tokenizer.tok2idx['beat:0']
    self.max_beat = tokenizer.tok2idx['beat:5']
    self.min_jg = tokenizer.tok2idx['jg:0']
    self.max_jg = tokenizer.tok2idx['jg:19']
    self.min_gak = tokenizer.tok2idx['gak:0']
    self.max_gak = tokenizer.tok2idx['gak:3']
    
    self.jg_r = jg_r
    self.entire_orn_r = entire_orn_r

    self.frame_mask_r = frame_rand_r
    self.frame_replace_r = frame_replace_r
    self.mask_note_r = mask_note_r
    self.replace_note_r = replace_note_r
    self.note_erase_r = erase_note_r
    self.total_note_r = mask_note_r + replace_note_r + erase_note_r

    
  
  def mask_ornaments(self, x:torch.Tensor, start:int, end:int):
    x[start:end, 1] = self.mask_id
    return x
  
  def mask_entire_ornaments(self, x:torch.Tensor):
    x[:, 1] = self.mask_id
  
  def mask_pitch(self, x:torch.Tensor, start:int, end:int):
    x[start:end, 0] = self.mask_id
    return x
  
  def change_pitch(self, x:torch.Tensor, start:int, end:int):
    random_pitch = random.choice(self.pitch_ids)
    x[start:end, 0] = random_pitch
    return x

  def change_ornament(self, x:torch.Tensor, start:int, end:int):
    random_orn = random.choice(self.ornament_ids)
    x[start:end, 1] = random_orn
    return x
  
  def mask_jeonggan(self, x, loss_mask, n=1):
    num_jg = x.shape[0] // self.num_beat
    num_inst = x.shape[2]
    jg_ids = random.sample(range(num_jg), n)
    inst_ids = [random.randint(0, num_inst+1) for _ in range(n)]
    for jg_id, inst_id in zip(jg_ids, inst_ids):
      start = jg_id * self.num_beat
      end = start + self.num_beat      
      if inst_id >= num_inst:
        x[start:end, 0:2] = self.mask_id
        loss_mask[start:end, 0:2] = 1
      else:
        x[start:end, 0:2, inst_id] = self.mask_id
        loss_mask[start:end, 0:2, inst_id] = 1
    return loss_mask
  
  
  def sample_rand_jg(self, max_jg):
    rand_jg = random.randint(self.min_jg, max_jg)
    rand_gak = random.randint(self.min_gak, self.max_gak)
    return rand_jg, rand_gak
  
  def _make_rand_idx_for_pitch(self, x, rand_r):
    num_frame, _, num_inst = x.shape
    n = int(num_frame * num_inst * rand_r)
    rand_idx = torch.randint(0, num_frame, (n,))
    rand_inst = torch.randint(0, num_inst, (n,))

    rand_pair = (rand_idx, torch.zeros_like(rand_idx), rand_inst)
    return rand_pair

  def _make_rand_idx_for_note(self, x, rand_r):
    frame_ids, inst_ids = torch.where(x[:,0,:] != self.sus_id)
    n = int(len(frame_ids) * rand_r)
    rand_idx = torch.randint(0, len(frame_ids), (n,))
    rand_pair = (frame_ids[rand_idx], torch.zeros_like(rand_idx), inst_ids[rand_idx])
    return rand_pair


  def mask_rand_pitch_frame(self, x:torch.Tensor, loss_mask:torch.Tensor):
    rand_pair = self._make_rand_idx_for_pitch(x, self.frame_mask_r)
    x[rand_pair] = self.mask_id
    loss_mask[rand_pair] = 1
  
  def replace_rand_pitch_frame(self, x, loss_mask):
    rand_pair = self._make_rand_idx_for_pitch(x, self.frame_replace_r)
    rand_pitches = torch.randint(self.min_pitch, self.max_pitch, (len(rand_pair[0]),))
    x[rand_pair] = rand_pitches
    loss_mask[rand_pair] = 1
    
  def mask_erase_replace_note(self, x, loss_mask):
    rand_pair = self._make_rand_idx_for_note(x, self.total_note_r)
    num_masks = int(len(rand_pair[0]) * (self.mask_note_r / self.total_note_r))
    num_replaces = int(len(rand_pair[0]) * (self.replace_note_r / self.total_note_r))
    
    x[rand_pair][:num_masks] = self.mask_id
    x[rand_pair][num_masks:num_masks+num_replaces] = torch.randint(self.min_pitch, self.max_pitch, (num_replaces,))
    x[rand_pair][num_masks+num_replaces:] = self.sus_id
    loss_mask[rand_pair] = 1
    
    
  
  def mask_rand_note(self, x):
    rand_pair = self._make_rand_idx_for_note(x, self.mask_note_r)
    x[rand_pair] = self.mask_id

  def erase_rand_note(self, x):
    rand_pair = self._make_rand_idx_for_note(x, self.note_erase_r)
    x[rand_pair] = self.sus_id
  
  def replace_rand_note_pitch(self, x):
    rand_pair = self._make_rand_idx_for_note(x, self.replace_note_r)
    rand_pitches = torch.randint(0, len(self.pitch_ids), (len(rand_pair[0]),))
    x[rand_pair] = rand_pitches
    

  def __call__(self, x:torch.Tensor):
    assert x.ndim == 3
    loss_mask = torch.zeros_like(x, dtype=torch.long)
    
    num_jg_to_mask = int(x.shape[0] // self.num_beat * self.jg_r)
    self.mask_jeonggan(x, loss_mask, num_jg_to_mask)
    self.mask_rand_pitch_frame(x, loss_mask)
    self.replace_rand_pitch_frame(x, loss_mask)
    self.mask_erase_replace_note(x, loss_mask)
    
    if random.random() < self.entire_orn_r: # 5% chance
      self.mask_entire_ornaments(x)
    
    return x, loss_mask

  def mask_every_start_beat(self, x:torch.Tensor):
    if x.ndim == 2:
      x = x.unsqueeze(-1)
    assert x.ndim == 3 # (num_frame, num_feature, num_inst)
    loss_mask = torch.zeros_like(x, dtype=torch.long)
    if x.shape[0] % self.num_beat == 0:
      x[::self.num_beat, 0:2] = self.mask_id
      loss_mask[::self.num_beat, 0:2] = 1
    elif (x.shape[0]-2) % self.num_beat == 0: # start and end token appended
      x[1::self.num_beat, 0:2] = self.mask_id
      loss_mask[1::self.num_beat, 0:2] = 1
    else:
      raise ValueError("Invalid input shape")
    return x, loss_mask
  
  def mask_except_start_beat(self, x:torch.Tensor):
    if x.ndim == 2:
      x = x.unsqueeze(-1)
    assert x.ndim == 3 # (num_frame, num_feature, num_inst)
    loss_mask = torch.ones_like(x, dtype=torch.long)
    loss_mask[:, 2:] = 0
    masked = x.clone()
    masked[:, 0:2] = self.mask_id
    
    if x.shape[0] % self.num_beat == 0:
      masked[::self.num_beat, 0:2] = x[::self.num_beat, 0:2]
      loss_mask[::self.num_beat, 0:2] = 0
    elif (x.shape[0]-2) % self.num_beat == 0: # start and end token appended
      # x[1::self.num_beat, 0:2] = self.mask_id
      masked[1::self.num_beat, 0:2] = x[1::self.num_beat, 0:2]
      loss_mask[1::self.num_beat, 0:2] = 0
      masked[0] = x[0]
      masked[-1] = x[-1]
      loss_mask[0] = 0
      loss_mask[-1] = 0
    else:
      raise ValueError("Invalid input shape")
    return masked, loss_mask