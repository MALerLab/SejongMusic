from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pack_padded_sequence

def previous_fill_pitches(datas): # [pitch, is_onset]
  padded_data, lengths = pad_packed_sequence(datas, batch_first=True, padding_value=-1)
  for data in padded_data:
    for note in data:
      if note[1] == 1:
        org_pitch = note[0]
      if note[0] != org_pitch:
        note[0] = org_pitch
  repacked_data = pack_padded_sequence(padded_data, lengths, batch_first=True, enforce_sorted=False)
  return repacked_data

def fill_pitches(datas, sus_idx=0):
  padded_data, lengths = pad_packed_sequence(datas, batch_first=True, padding_value=-1)
  for data in padded_data:
    # org_pitch = data[0]
    for i, pitch in enumerate(data[1:], start=1):
      if pitch == sus_idx:
        data[i] = data[i-1]
  repacked_data = pack_padded_sequence(padded_data, lengths, batch_first=True, enforce_sorted=False)

  return repacked_data