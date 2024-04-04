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