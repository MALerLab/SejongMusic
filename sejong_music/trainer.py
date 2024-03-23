import os

import torch
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
from music21 import stream, environment

from .metric import get_similarity_with_inference, get_correspondence_with_inference
from .utils import make_dynamic_template
from .constants import MEAS_LEN_BY_IDX, get_dynamic_template_for_orch
from .inference import sequential_inference, Inferencer
# from .yeominrak_processing import SamplingScore
from .evaluation import fill_pitches
from .decode import MidiDecoder, OrchestraDecoder


us=environment.UserSettings()
us['musescoreDirectPNGPath'] = '/usr/bin/mscore'
os.putenv("QT_QPA_PLATFORM", "offscreen")
os.putenv("XDG_RUNTIME_DIR", environment.Environment().getRootTempDir())



class Trainer:
  def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, device, save_dir, save_log=True, scheduler=None):
  # def __init__(self, **kwargs):
  #   for key, value in kwargs.items():
  #     setattr(self, key, value)  
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.loss_fn = loss_fn
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    self.midi_decoder = MidiDecoder(self.valid_loader.dataset.tokenizer)

    self.model.to(device)
    
    self.best_valid_accuracy = 0
    self.device = device
    self.save_log = save_log

    self.best_pitch_sim = 0
    self.best_valid_loss = 100
    
    self.training_loss = []
    self.validation_loss = []
    self.validation_acc = []
    self.pitch_acc = []
    self.dur_acc = []
    self.iteration = 0
    self.save_dir = Path(save_dir)
    self.save_dir.mkdir(exist_ok=True)
    # 여기서부터는 metric
    self.measure_match = []
    self.pitch_similarity = []
    self.dynamic_similarity = []
    
  def train_by_num_epoch(self, num_epochs):
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
      self.model.train()
      # for batch in tqdm(self.train_loader, leave=False):
      attn_map = []
      if hasattr(self.train_loader.dataset, 'update_slice_info'):
        self.train_loader.dataset.update_slice_info()
      for batch in self.train_loader:
        loss_value, train_loss_dict, attn_weight = self._train_by_single_batch(batch)
        self.training_loss.append(loss_value)
        attn_map.append(attn_weight)

      self.model.eval()
      validation_loss, validation_acc, valid_loss_dict, valid_pitch_acc, valid_dur_acc = self.validate()
      self.pitch_acc.append(valid_pitch_acc)
      self.dur_acc.append(valid_dur_acc)
      self.valid_loss_dict = valid_loss_dict
      self.validation_loss.append(validation_loss)
      self.validation_acc.append(validation_acc)
      if self.save_log:
        wandb.log({'validation.loss': validation_loss}, step=self.iteration)
        wandb.log({'validation.acc': validation_acc}, step=self.iteration)
        wandb.log({'validation.pitch_acc': valid_pitch_acc}, step=self.iteration)
        wandb.log({'validation.dur_acc': valid_dur_acc}, step=self.iteration)
        wandb.log({'attn_weight': attn_weight}, step=self.iteration)
      # valid_acc가 가장 높은 모델을 저장
      pbar.set_description(f"Acc: {validation_acc:.4f}, Loss: {validation_loss:.4f}")
      if validation_acc > self.best_valid_accuracy:
        self.best_valid_accuracy = validation_acc
        torch.save(self.model.state_dict(), self.save_dir / 'best_model.pt')
        # print(f'Best Model Saved! Accuracy: {validation_acc}')
      if validation_loss < self.best_valid_loss:
        self.best_valid_loss = validation_loss
        torch.save(self.model.state_dict(), self.save_dir / 'best_loss_model.pt')
      
      if (epoch + 1) % 100 == 0:
        torch.save(self.model.state_dict(), self.save_dir / f'epoch{epoch+1}_model.pt')
      
      if (epoch + 1) % 50 == 0 and epoch > 250:
        measure_match, similarity, dynamic_corr = self.make_inference_result()
        if similarity > self.best_pitch_sim:
          self.best_pitch_sim = similarity
          torch.save(self.model.state_dict(), self.save_dir / 'best_pitch_sim_model.pt')
        # try:
        #   self.make_inference_result(epoch)
        #   print('Inference Result Saved!')
        # except:
        #   print('Inference Result Failed to Save')
      self.model.to(self.device)
    torch.save(self.model.state_dict(), self.save_dir / 'last_model.pt')
    return attn_map

  
  def make_inference_result(self, write_png=False, loader=None):
    if loader is None:
      loader = self.valid_loader
    # self.model.to('cpu')
    # for idx in range(self.valid_loader.batch_size):
    selected_idx_list = [i for i in range(len(loader.dataset))]
    measure_match = []
    similarity = []
    dynamic_corr = []
    dynamic_templates = make_dynamic_template(offset_list=MEAS_LEN_BY_IDX)
    for idx in selected_idx_list:
      sample, _, target = loader.dataset[idx]
      # if isinstance(self.train_loader.dataset, SamplingScore):
      #   target = self.model.converter(target)
      # else:
      target = self.model.converter(target[:-1])
      input_part_idx = sample[0][0]
      target_part_idx = target[0][0]
      # if input_part_idx < 2: continue
      if self.model.is_condition_shifted:
        src, output, attn_map = self.model.shifted_inference(sample.to(self.device), target_part_idx)
      else:
        src, output, attn_map = self.model.inference(sample.to(self.device), target_part_idx)
      input_measure_len, output_measure_len = MEAS_LEN_BY_IDX[input_part_idx], MEAS_LEN_BY_IDX[target_part_idx]
      try:
        is_match = sum([note[2] for note in src])/input_measure_len == sum([note[2] for note in output])/output_measure_len
      except Exception as e:
        print(f"Error occured in inference result: {e}")
        print(src)
        print(output)
        print([note[2] for note in output])
        is_match = False
      measure_match += [is_match]
      if is_match:
        similarity += [get_similarity_with_inference(target, output, dynamic_templates)]
        dynamic_corr += [get_correspondence_with_inference(target, output, dynamic_templates)]
      
      src_midi, output_midi = self.midi_decoder(src), self.midi_decoder(output)

      merged_midi = stream.Score()
      merged_midi.insert(0, src_midi)
      merged_midi.insert(0, output_midi)
      
      
      if write_png:
        merged_midi.write('musicxml.png', fp=str(self.save_dir / f'{input_part_idx}-{target_part_idx}.png'))
        if self.save_log:
          wandb.log({f'inference_result_{idx}_from{input_part_idx}to{target_part_idx}': wandb.Image(str(self.save_dir / f'{input_part_idx}-{target_part_idx}-1.png'))},
                    step=self.iteration)

      '''
      if input_part_idx < 6: 
        if self.model.is_condition_shifted:
          src2, output2, attn_map2 = self.model.shifted_inference(sample, input_part_idx+2)
        else:
          src2, output2, attn_map2 = self.model.inference(sample, input_part_idx+2)

        output_measure_len = MEAS_LEN_BY_IDX[input_part_idx+2]
        is_match2 = sum([note[2] for note in src2])/input_measure_len == sum([note[2] for note in output2])/output_measure_len
        measure_match += [is_match2]
        # if is_match2:
        #   similarity += [get_similarity_with_inference(src2, output2, dynamic_templates)]
        #   dynamic_corr += [get_correspondence_with_inference(src2, output2, dynamic_templates)]
        
        src_midi2, output_midi2 = self.midi_decoder(src2), self.midi_decoder(output2)

        merged_midi2 = stream.Stream()
        for element in src_midi2:
          merged_midi2.append(element)
        for element in output_midi2:
          merged_midi2.append(element)

        if write_png:
          merged_midi2.write('musicxml.png', fp=str(self.save_dir / f'{input_part_idx}-{input_part_idx+2}.png'))
          if self.save_log:
            wandb.log({f'inference_result_{idx}_from{input_part_idx}to{input_part_idx+2}': wandb.Image(str(self.save_dir / f'{input_part_idx}-{input_part_idx+2}-1.png'))},
                      step=self.iteration
                      )
              '''
    measure_match = sum(measure_match)/len(measure_match)
    if len(similarity) == 0:
      similarity = 0
      dynamic_corr = 0
    else:
      similarity = sum(similarity)/len(similarity)
      dynamic_corr = sum(dynamic_corr)/len(dynamic_corr)
    print(f"Measure Match: {measure_match:.4f}, Similarity: {similarity:.4f}, Dynamic Corr: {dynamic_corr:.4f}")
    self.measure_match.append(measure_match)
    self.pitch_similarity.append(similarity)
    self.dynamic_similarity.append(dynamic_corr)
    
    if self.save_log:
      wandb.log({'measure_match': measure_match}, step=self.iteration)
      wandb.log({'pitch_similarity': similarity}, step=self.iteration)
      wandb.log({'dynamic_similarity': dynamic_corr}, step=self.iteration)
    
    self.model.to(self.device)
    return measure_match, similarity, dynamic_corr

  def make_sequential_inference_result(self, write_png=False):
    self.model.cpu()
    selected_idx_list = [i for i in range(len(self.valid_loader.dataset))][0::self.valid_loader.dataset.slice_measure_number]
    for idx in selected_idx_list:
      sample = self.valid_loader.dataset[idx][0]
      if sample[0, 0] != 0: break
      output_stream = sequential_inference(self.model, sample, self.midi_decoder)

      if write_png:
        output_stream.write('musicxml.png', fp=str(self.save_dir / f'seq_inf_sample{idx}.png'))
        if self.save_log:
          wandb.log({f'seq_inf_sample{idx}': wandb.Image(str(self.save_dir / f'seq_inf_sample{idx}-1.png'))},
                    step=self.iteration)
    self.model.to(self.device)


  
  def _train_by_single_batch(self, batch):  
    loss, _, loss_dict, attn_weight = self.get_loss_from_single_batch(batch)
    
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    if self.scheduler is not None:
      self.scheduler.step()
    if self.save_log:
        wandb.log({f"training.loss": loss}, step=self.iteration)
        
    self.iteration += 1
    return loss.item(), loss_dict, attn_weight

  def get_loss_from_single_batch(self, batch):
    src, tgt, shifted_tgt = batch
    src = src.to(self.device)
    tgt = tgt.to(self.device)
    pred, attn_weight = self.model(src, tgt)
    
    if isinstance(pred, PackedSequence):  
      pitch_loss = self.loss_fn(pred.data[:, :self.model.vocab_size[1]], shifted_tgt.data[:, 1])
      dur_loss = self.loss_fn(pred.data[:, self.model.vocab_size[1]:], shifted_tgt.data[:, 2])
    else:
      pitch_loss = self.loss_fn(pred[..., :self.model.vocab_size[1]], shifted_tgt[..., 1])
      dur_loss = self.loss_fn(pred[..., self.model.vocab_size[1]:], shifted_tgt[..., 2])
    total_loss = (pitch_loss + dur_loss) / 2
    
    loss_dict = {'pitch_loss': pitch_loss.item(), 'dur_loss': dur_loss.item(), 'total_loss': total_loss.item()}
    return total_loss, pred, loss_dict, attn_weight

  def validate(self):
    # 여기서는 그냥 loader = valid_loader로 고정!
    validation_loss = 0
    validation_acc = 0
    num_total_tokens = 0
    valid_pitch_acc = 0
    valid_dur_acc = 0

    with torch.inference_mode():
      # for batch in tqdm(self.valid_loader, leave=False):
      for batch in self.valid_loader:
        tmp_validation_acc, pitch_acc, dur_acc, tmp_validation_loss, tmp_num_total_tokens, loss_dict = self.get_valid_loss_from_single_batch(batch)
        num_total_tokens += tmp_num_total_tokens
        validation_loss += tmp_validation_loss * tmp_num_total_tokens
        validation_acc += tmp_validation_acc * tmp_num_total_tokens
        valid_pitch_acc += pitch_acc * tmp_num_total_tokens
        valid_dur_acc += dur_acc * tmp_num_total_tokens
      
    return validation_loss / num_total_tokens, validation_acc / num_total_tokens, loss_dict, valid_pitch_acc / num_total_tokens, valid_dur_acc / num_total_tokens
    
  def get_valid_loss_from_single_batch(self, batch):
    src, _, shifted_tgt = batch
    loss, pred, loss_dict, attention_weight = self.get_loss_from_single_batch(batch)
    
    if isinstance(pred, PackedSequence):
      num_tokens = pred.data.shape[0]
      pitch_acc = float(torch.sum(torch.argmax(pred.data[..., :self.model.vocab_size[1]], dim=-1) == shifted_tgt.to(self.device).data[...,1])) / num_tokens
      dur_acc = float(torch.sum(torch.argmax(pred.data[..., self.model.vocab_size[1]:], dim=-1) == shifted_tgt.to(self.device).data[...,2]))  / num_tokens
    else:
      mask = shifted_tgt[..., 1] != 0 # 2-dim
      mask = mask.to(self.device)
      num_tokens = mask.sum()
      pitch_acc = float(torch.sum((torch.argmax(pred[..., :self.model.vocab_size[1]], dim=-1) == shifted_tgt.to(self.device)[...,1])*mask)) / num_tokens
      dur_acc = float(torch.sum((torch.argmax(pred[..., self.model.vocab_size[1]:], dim=-1) == shifted_tgt.to(self.device)[...,2])*mask)) / num_tokens

      
    main_acc = (pitch_acc + dur_acc) / 2
    
    validation_loss = loss.item()

    loss_dict['pitch_acc'] = pitch_acc
    loss_dict['dur_acc'] = dur_acc
    loss_dict['main_acc'] = main_acc
        
    return main_acc, pitch_acc, dur_acc, validation_loss, num_tokens, loss_dict
  
  def load_best_model(self):
    self.model.load_state_dict(torch.load(self.save_dir/'best_pitch_sim_model.pt'))
    self.model.eval()
    print('Best Model Loaded!')

def get_pitch_dur_acc(pred, tgt, vocab_size, device):
  num_tokens = pred.data.shape[0]
  pitch_acc = float(torch.sum(torch.argmax(pred.data[:, :vocab_size[1]], dim=-1) == tgt.to(device).data[:,1])) / num_tokens
  dur_acc = float(torch.sum(torch.argmax(pred.data[:, vocab_size[1]:], dim=-1) == tgt.to(device).data[:,2]))  / num_tokens
  main_acc = (pitch_acc + dur_acc) / 2
  
  return pitch_acc, dur_acc, main_acc









# ------------------ CNNTrainer ------------------ #

class CNNTrainer(Trainer):
  def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, device):
    super().__init__(model, optimizer, loss_fn, train_loader, valid_loader, device)
  

  def get_loss_from_single_batch(self, batch):
    src, tgt = batch
    src = src.to(self.device)
    tgt = tgt.to(self.device)
    pred = self.model(src, tgt)
    
    batch_len = (tgt.shape[1] - (tgt[:, :, 1] == 0).sum(dim=1)).cpu()
    tgt = pack_padded_sequence(tgt[:, :, 1], batch_len, batch_first=True, enforce_sorted=False)
    pred = pack_padded_sequence(pred, batch_len, batch_first=True, enforce_sorted=False)

    total_loss = self.loss_fn(pred.data, tgt.data)
    
    loss_dict = {'total_loss': total_loss.item()}
    return total_loss, pred, loss_dict, None

  def get_valid_loss_from_single_batch(self, batch):
    src, tgt = batch
    loss, pred, loss_dict, attention_weight = self.get_loss_from_single_batch(batch)
    batch_len = (tgt.shape[1] - (tgt[:, :, 1] == 0).sum(dim=1)).cpu()
    tgt = pack_padded_sequence(tgt[:, :, 1], batch_len, batch_first=True, enforce_sorted=False)

    num_tokens = pred.data.shape[0]
    pitch_acc = float(torch.sum(torch.argmax(pred.data, dim=-1) == tgt.data.to(self.device))) / num_tokens
    
    validation_loss = loss.item() * num_tokens
    # num_total_tokens = num_tokens
    # # validation_acc = acc.item()
    # validation_acc = acc
    loss_dict['pitch_acc'] = pitch_acc

    # loss_dict['main_acc'] = main_acc.item()
    # loss_dict['dur_acc'] = (dur_acc * is_note.sum() / num_tokens).item()

    return pitch_acc, pitch_acc, pitch_acc, validation_loss, num_tokens, loss_dict



class OrchestraTrainer(Trainer):
  def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, device, save_dir, save_log=True, scheduler=None):
    super().__init__(model, optimizer, loss_fn, train_loader, valid_loader, device, save_dir, save_log, scheduler)
    self.midi_decoder = OrchestraDecoder(self.valid_loader.dataset.tokenizer)
    self.inferencer = Inferencer( model, 
                                  is_condition_shifted=True,  
                                  is_orch=True,
                                  is_sep=False,
                                  temperature=1.0, 
                                  top_p=0.9)
    if hasattr(self.valid_loader.dataset, 'era_dataset'):
      self.source_decoder = MidiDecoder(self.valid_loader.dataset.era_dataset.tokenizer)
    else:
      self.source_decoder = self.midi_decoder

    self.dynamic_template = get_dynamic_template_for_orch()
    # self.dynamic_templates = [self.dynamic_template for _ in range(8)]

    self.dynamic_templates = []
    for part_idx in range(8):
        frame = 4
        whole_measure_len = int(frame * 30)
        dynamic_list = []
        for i in range(whole_measure_len):
            dynamic = self.dynamic_template.get(float(i) / frame, 'none')
            dynamic_list.append(dynamic)
        self.dynamic_templates.append(dynamic_list)


  def make_inference_result(self, write_png=False, loader=None):
    # return 0, 0, 0
    if loader is None:
      loader = self.valid_loader
    # self.model.to('cpu')
    # for idx in range(self.valid_loader.batch_size):
    selected_idx_list = [i for i in range(len(loader.dataset))]
    measure_match = []
    similarity = [0]
    dynamic_corr = [0]
    for idx in selected_idx_list:
      sample, _, target = loader.dataset[idx]
      target = self.model.converter(target[:-1])
      input_part_idx = sample[0][0]
      target_part_idx = target[0][0]
      # if input_part_idx < 2: continue
      src, output, attn_map = self.inferencer.inference(sample.to(self.device), target_part_idx)
      try:
        is_match = sum([note[2] for note in target]) == sum([note[2] for note in output])
      except Exception as e:
        print(f"Error occured in inference result: {e}")
        print(src)
        print(output)
        print([note[2] for note in output])
        is_match = False
      measure_match += [is_match]
      if is_match:
        similarity += [get_similarity_with_inference(target, output, self.dynamic_templates, beat_sampling_num=4)]
        dynamic_corr += [get_correspondence_with_inference(target, output, self.dynamic_templates, beat_sampling_num=4)]
      
      src_midi, output_midi = self.source_decoder(src), self.midi_decoder(output)

      merged_midi = stream.Score()
      merged_midi.insert(0, src_midi)
      merged_midi.insert(0, output_midi)
      
      
      if write_png:
        merged_midi.write('musicxml.png', fp=str(self.save_dir / f'{input_part_idx}-{target_part_idx}.png'))
        if self.save_log:
          wandb.log({f'inference_result_{idx}_from{input_part_idx}to{target_part_idx}': wandb.Image(str(self.save_dir / f'{input_part_idx}-{target_part_idx}-1.png'))},
                    step=self.iteration)

    measure_match = sum(measure_match)/len(measure_match)
    if len(similarity) == 0:
      similarity = 0
      dynamic_corr = 0
    else:
      similarity = sum(similarity)/len(similarity)
      dynamic_corr = sum(dynamic_corr)/len(dynamic_corr)
    print(f"Measure Match: {measure_match:.4f}, Similarity: {similarity:.4f}, Dynamic Corr: {dynamic_corr:.4f}")
    self.measure_match.append(measure_match)
    self.pitch_similarity.append(similarity)
    self.dynamic_similarity.append(dynamic_corr)
    
    if self.save_log:
      wandb.log({'measure_match': measure_match}, step=self.iteration)
      wandb.log({'pitch_similarity': similarity}, step=self.iteration)
      wandb.log({'dynamic_similarity': dynamic_corr}, step=self.iteration)
    
    self.model.to(self.device)
    return measure_match, similarity, dynamic_corr


# ------------------ RollTrainer ------------------ # 


class RollTrainer(Trainer):
  def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, device, save_dir, save_log=True, scheduler=None):
    super().__init__(model, optimizer, loss_fn, train_loader, valid_loader, device, save_dir, save_log, scheduler)

  def get_loss_from_single_batch(self, batch):
    src, tgt, shifted_tgt = batch
    src = src.to(self.device)
    tgt = tgt.to(self.device)
    pred, attn_weight = self.model(src, tgt)
    pitch_loss = self.loss_fn(pred.data, shifted_tgt.data[:,1])
    total_loss = pitch_loss
    
    loss_dict = {'pitch_loss': pitch_loss.item(), 'dur_loss': pitch_loss.item(), 'total_loss': total_loss.item()}
    return total_loss, pred, loss_dict, attn_weight

  def get_valid_loss_from_single_batch(self, batch):
    src, _, shifted_tgt = batch
    loss, pred, loss_dict, attention_weight = self.get_loss_from_single_batch(batch)
    num_tokens = pred.data.shape[0]
    pitch_acc = float(torch.sum(torch.argmax(pred.data, dim=-1) == shifted_tgt.to(self.device).data[:,1])) / num_tokens
    main_acc = pitch_acc
    
    validation_loss = loss.item()

    loss_dict['pitch_acc'] = pitch_acc
    loss_dict['dur_acc'] = pitch_acc
    loss_dict['main_acc'] = main_acc
        
    return main_acc, pitch_acc, pitch_acc, validation_loss, num_tokens, loss_dict


  def visualize(self, num=1):
    batch = next(iter(self.valid_loader))
    src, dec_input, shifted_tgt = batch
    src = src.to(self.device)
    dec_input = dec_input.to(self.device)
    
    pred = self.model(src, dec_input)
    pred = torch.nn.utils.rnn.pad_packed_sequence(pred, batch_first=True)[0].detach().cpu().numpy()
    tgt = torch.nn.utils.rnn.pad_packed_sequence(shifted_tgt, batch_first=True)[0].detach().cpu().numpy()

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(pred[0].T, aspect='auto', interpolation='nearest')
    plt.subplot(2, 1, 2)
    plt.imshow(tgt[0].T, aspect='auto', interpolation='nearest')
    plt.show()
    plt.savefig(f'pred_{num}.png')
    plt.close()


  
class RollPitchTrainer(RollTrainer):
  def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, device, advanced_metric=True):
    super().__init__(model, optimizer, loss_fn, train_loader, valid_loader, device)
    self.advanced_metric = advanced_metric
  
  def get_loss_from_single_batch(self, batch):
    src, dec_input, tgt = batch
    src = src.to(self.device)
    dec_input = dec_input.to(self.device)
    tgt = tgt.to(self.device)
    pred = self.model(src, dec_input)
    
    pitch_loss = self.loss_fn(pred.data, tgt.data)
    onset_loss = None
    total_loss = pitch_loss#(pitch_loss + onset_loss) / 2
    
    loss_dict = {'pitch_loss': pitch_loss.item(), 'total_loss': total_loss.item()}
    return total_loss, pred, loss_dict, None

  def get_valid_loss_from_single_batch(self, batch):
    src, dec_input, tgt = batch
    loss, pred, loss_dict, _ = self.get_loss_from_single_batch(batch)
    num_tokens = pred.data.shape[0]

    if self.advanced_metric:
      predicted_pitch_result = torch.argmax(pred.data, dim=-1)
      predicted_pitch_result = PackedSequence(predicted_pitch_result, pred.batch_sizes, pred.sorted_indices, pred.unsorted_indices)
    #   predicted_is_onset_result = pred.data[:, -1] >0.5
    #   concated_result = torch.cat([predicted_pitch_result.unsqueeze(-1), predicted_is_onset_result.unsqueeze(-1)], dim=-1)
    #   pitch_result = torch.argmax(pred.data[:, :self.model.vocab_size[1]], dim=-1)
      filled_pitch_result = fill_pitches(predicted_pitch_result)
      filled_tgt = fill_pitches(tgt)
      filled_tgt = filled_tgt.to(filled_pitch_result.data.device)
    #   fillted_tgt = fill_pitches(shifted_tgt.to(self.device).data[:,0])
      pitch_acc = float(torch.sum(filled_pitch_result.data == filled_tgt.data)) / num_tokens

    #   is_onset = shifted_tgt.to(self.device).data[:,1] != 0
    #   onset_acc = float(torch.sum((pred.data[:, -1] >0.5)[is_onset] == shifted_tgt.to(self.device).data[:,1][is_onset]))  / is_onset.sum()

    # else:
    # pitch_acc = float(torch.sum(torch.argmax(pred.data, dim=-1) == shifted_tgt.to(self.device).data)) / num_tokens
    # onset_acc = float(torch.sum( (pred.data[:, -1] >0.5) == shifted_tgt.to(self.device).data[:,1]))  / num_tokens

    main_acc = pitch_acc
    
    validation_loss = loss.item() * num_tokens
    loss_dict['pitch_acc'] = pitch_acc
    # loss_dict['onset_acc'] = onset_acc
    loss_dict['main_acc'] = main_acc

    # loss_dict['main_acc'] = main_acc.item()
    # loss_dict['dur_acc'] = (dur_acc * is_note.sum() / num_tokens).item()

    return main_acc, pitch_acc, None, validation_loss, num_tokens, loss_dict