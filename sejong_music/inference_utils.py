import torch
import math

def get_measure_shifted_output(output: torch.Tensor):
  for i, token in enumerate(output):
    if token[-1] == 4: # new measure started:
      break
  for j in range(i, len(output)):
    if output[j][-1] == 6:
      break

  output_tokens_from_second_measure = output[i:j].clone()
  output_tokens_from_second_measure[0, 1] = 1
  output_tokens_from_second_measure[0, 2] = 1
  output_tokens_from_second_measure[:, -1] -= 1
  return output_tokens_from_second_measure


def prepare_input_for_next_part(outputs_tensor):
  pred, condition = outputs_tensor[:, :3], outputs_tensor[:, 3:]
  next_input = torch.cat([torch.cat([pred[0:1], torch.tensor([[3,3,3]])], dim=1) , torch.cat([pred[1:], condition[:-1]], dim=1) ], dim=0)
  return next_input


def round_number(number, decimals=3):
    scale = 10.0 ** decimals
    return math.floor(number * scale) / scale


def get_measure_specific_output(output:torch.LongTensor, measure_idx_in_token:int):
  corresp_ids = torch.where(output[:,-1] == measure_idx_in_token)[0] + 1
  if corresp_ids[-1] >= len(output):
    print(f"Warning: next measure of {measure_idx_in_token} is not in the output.")
    corresp_ids[-1] = len(output) - 1
  return output[corresp_ids]


def fix_measure_idx(sample:torch.LongTensor):
  new_sample = sample.clone()
  current_measure = 2
  ornament_on = False
  for note in new_sample:
    if note[1] in (1, 2): continue
    if note[3] == 3 and not ornament_on: # beat_start
      current_measure += 1
    note[5] = current_measure
    ornament_on = note[2] == 4 # note duration is zero, which means ornament
  return new_sample

def recover_beat(output:torch.LongTensor, tokenizer, recover_amount=5):
  new_output = output.clone()
  accum_dur = 0
  for i, token in enumerate(new_output):
    accum_dur += tokenizer.vocab['duration'][token[2]]
    if abs(accum_dur - recover_amount) < 0.1:
      break
    elif accum_dur > recover_amount+0.1:
      print('Exceed!', accum_dur, recover_amount)
      exceed_amount = accum_dur - recover_amount
      new_output[i+1, 2] = tokenizer.tok2idx['duration'][round(tokenizer.vocab['duration'][new_output[i+1, 2]] + exceed_amount)]
      break


  return new_output[i+1:]


def fill_in_source(src, num_measure):
  # src: list of token in strings
  outputs = []
  for note in src:
    if note[-1] in range(4-num_measure):
      continue
    else:
      outputs.append(note)
  return outputs

