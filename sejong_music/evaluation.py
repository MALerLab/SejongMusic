from torch.nn.utils.rnn import pack_sequence, PackedSequence, pad_packed_sequence, pack_padded_sequence
from tqdm.auto import tqdm
from .eval_metrics import per_jg_note_acc, onset_f1
from .inference import ABCInferencer
from .jg_code import JeongganDataset, ABCDataset
from typing import Union

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



def get_test_result(model, inferencer, test_dataset:Union[JeongganDataset, ABCDataset], target_inst='daegeum', condition_insts=None):
  note_acc = []
  note_none_strict = []
  onset_f1_score = []
  onset_prec_score = []
  onset_recall_score = []
  jg_matched = []
  num_samples = 0
  for idx in tqdm(range(len(test_dataset))):
    if target_inst not in list(test_dataset.entire_segments_with_metadata[idx][0].keys()):
      # print(f"Target instrument {target_inst} not found in the piece {test_dataset.entire_segments_with_metadata[idx][2]}")
      continue
    num_samples += 1
    sample, tgt, shifted_tgt = test_dataset.get_item_with_target_inst(idx, target_inst, condition_insts)
    input_part_idx = sample[0][-1]
    target_part_idx = model.tokenizer.vocab[tgt[0][-1].item()]
    # if input_part_idx < 2: continue
    try:
      src, output, (attn, output_tensor, _) = inferencer.inference(sample.to(inferencer.device), target_part_idx)
    except Exception as e:
      print(f"Error occured in inference result: {e}")
      continue
      # print([note[2] for note in output])
    if isinstance(inferencer, ABCInferencer):
      num_gen_jg = sum([1 for note in output if note[0] in ('|', '\n')])
      gen_converted_tgt = inferencer.jg_decoder(inferencer.tokenizer.decode(shifted_tgt)).split(' ')
      num_tg_jg = sum([1 for note in gen_converted_tgt if note in ('|', '\n')])
    else:
      num_tg_jg = sum([1 for note in inferencer.tokenizer.decode(shifted_tgt) if note in ('|', '\n')])
      num_gen_jg = sum([1 for note in output if note[0] in ('|', '\n')])
    jg_matched.append(int(num_tg_jg == num_gen_jg))
    if num_tg_jg != num_gen_jg:
      print(f"Generated JG mismatch: num_tg_jg: {num_tg_jg}, num_gen_jg: {num_gen_jg}")
      continue
    try:
      if isinstance(inferencer, ABCInferencer):
        tgt_decoded = inferencer.tokenizer.decode(shifted_tgt)
        # tgt_decoded = [[token[0], token[-1]] for token in tgt_decoded]
        gen_str = inferencer.jg_decoder(tgt_decoded)
        tgt_converted = gen_str.split(' ')
        pred_converted = [x[0] for x in output]
      elif inferencer.use_offset:
        shifted_tgt = inferencer.beat2gen(inferencer.tokenizer.decode(shifted_tgt))
        tgt_converted = [x for x in shifted_tgt if x != '']
        pred_converted = [x[0] for x in output]
      else:
        pred_converted = output_tensor[:,0]
        tgt_converted = shifted_tgt        
      
      jg_note_acc = per_jg_note_acc(pred_converted, tgt_converted, inst=target_part_idx, tokenizer=inferencer.tokenizer, pitch_only=True)
      non_strict_acc = per_jg_note_acc(pred_converted, tgt_converted, inst=target_part_idx, tokenizer=inferencer.tokenizer, strict=False)
      f1, prec, recall = onset_f1(pred_converted, tgt_converted, inst=target_part_idx, tokenizer=inferencer.tokenizer)
    except Exception as e:
      print(f"Error occured in inference evaluation result: {e}")
      jg_note_acc, non_strict_acc = 0, 0
      f1, prec, recall = 0, 0, 0


    note_acc.append(jg_note_acc)
    note_none_strict.append(non_strict_acc)
    onset_f1_score.append(f1)
    onset_prec_score.append(prec)
    onset_recall_score.append(recall)
    
  jg_matched_score = sum(jg_matched)/num_samples
  note_acc = sum(note_acc)/num_samples
  note_none_strict = sum(note_none_strict)/num_samples
  onset_f1_score = sum(onset_f1_score)/num_samples
  onset_prec_score = sum(onset_prec_score)/num_samples
  onset_recall_score = sum(onset_recall_score)/num_samples

  
  return {'note_acc': note_acc, 
          'onset_f1': onset_f1_score, 
          'onset_prec': onset_prec_score, 
          'onset_recall': onset_recall_score, 
          'jg_matched': jg_matched_score, 
          'note_none_strict': note_none_strict}
