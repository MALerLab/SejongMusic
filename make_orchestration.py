from pathlib import Path
from music21 import converter, stream, note as m21_note 
import torch

from tqdm.auto import tqdm

from yeominrak_processing import AlignedScore, SamplingScore, pack_collate, ShiftedAlignedScore, OrchestraScore
from model_zoo import Seq2seq, Converter, AttentionSeq2seq, QkvAttnSeq2seq, get_emb_total_size, QkvAttnSeq2seqOrch
import random as random
from decode import MidiDecoder, OrchestraDecoder
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from fractions import Fraction

import math
def round_number(number, decimals=3):
    scale = 10.0 ** decimals
    return math.floor(number * scale) / scale


def prepare_input_for_next_part(outputs_tensor):
  pred, condition = outputs_tensor[:, :3], outputs_tensor[:, 3:]
  next_input = torch.cat([torch.cat([pred[0:1], torch.tensor([[3,3,3]])], dim=1) , torch.cat([pred[1:], condition[:-1]], dim=1) ], dim=0)
  return next_input

def get_measure_specific_output(output:torch.LongTensor, measure_idx_in_token:int):
  corresp_ids = torch.where(output[:,-1] == measure_idx_in_token)[0]+1
  return output[corresp_ids]


def fix_measure_idx(sample:torch.LongTensor):
  new_sample = sample.clone()
  current_measure = 2
  for note in new_sample:
    if note[1] in (1, 2): continue
    if note[3] == 3: # beat_start
      current_measure += 1
    note[5] = current_measure
  return new_sample



def fill_in_source(src, num_measure):
  # src: list of token in strings
  outputs = []
  for note in src:
    if note[-1] in range(4-num_measure):
      continue
    else:
      outputs.append(note)
  return outputs

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
  output_tokens_from_second_measure[:, 5] -= 1
  return output_tokens_from_second_measure

save_name = 'vanilla_CHP'
state = torch.load('outputs/2023-12-20/16-02-36/epoch600_model.pt')
state_alt = torch.load('outputs/2023-12-20/16-02-36/best_loss_model.pt')

config = OmegaConf.load('yamls/orchestration.yaml')
config = get_emb_total_size(config)

output_dir = Path('gen_results/')
output_dir.mkdir(parents=True, exist_ok=True)

device = 'cpu'

val_dataset = OrchestraScore(is_valid= True,
                            # valid_measure_num = [i for i in range(93, 99)],
                            xml_path='yeominlak_omr5.musicxml', 
                            use_pitch_modification=False, 
                            slice_measure_num=config.data.max_meas,
                            min_meas=config.data.min_meas,
                            feature_types=config.model.features,
                              sampling_rate=config.data.sampling_rate)

# test_set = ShiftedAlignedScore(is_valid= True,
#                                    xml_path = 'gen_results/CHP_scoreCPH_seq2seq_grace2_errorfixed.musicxml',
#                                     valid_measure_num = 'entire',
#                                     slice_measure_num=config.data.max_meas,
#                                     min_meas=config.data.min_meas,
#                                     feature_types=config.model.features,
#                                       sampling_rate=config.data.sampling_rate)

test_set = ShiftedAlignedScore(is_valid= True,
                                   xml_path = 'gen_results/CHP_scoreseq2seq_grace2_errorfixed.musicxml',
                                    valid_measure_num = 'entire',
                                    slice_measure_num=config.data.max_meas,
                                    min_meas=config.data.min_meas,
                                    feature_types=config.model.features,
                                      sampling_rate=config.data.sampling_rate)


test_set.result_pairs = [x for x in test_set.result_pairs if x[0] == 0][:len(test_set.slice_info)]
for pair in test_set.result_pairs:
  pair[0] = 7
  pair[1] = 7
test_set.tokenizer = val_dataset.era_dataset.tokenizer


for i in range(30):
  offset = round_number(i/3, 3)
  if offset in test_set.tokenizer.tok2idx['offset']:
     continue
  if i < 6:
    other_value = i % 3 + 6
  else:
    other_value = i % 3 + 21
  test_set.tokenizer.tok2idx['offset'][offset] = test_set.tokenizer.tok2idx['offset'][round_number(other_value/3, 3)]

for i in range(30):
  offset = round_number(i/2, 3)
  if offset in test_set.tokenizer.tok2idx['offset']:
     continue
  other_value = i - 2
  test_set.tokenizer.tok2idx['offset'][offset] = test_set.tokenizer.tok2idx['offset'][round_number(other_value/2, 3)]


model = QkvAttnSeq2seqOrch(val_dataset.era_dataset.tokenizer, val_dataset.tokenizer, config.model).to(device)
# state = torch.load('best_model.pt')
model.is_condition_shifted = True
model.load_state_dict(state)
model.eval()

model_alt = QkvAttnSeq2seqOrch(val_dataset.era_dataset.tokenizer, val_dataset.tokenizer, config.model).to(device)
model_alt.is_condition_shifted = True
model_alt.load_state_dict(state_alt)
model_alt.eval()

decoder = OrchestraDecoder(val_dataset.tokenizer)
source_decoder = MidiDecoder(val_dataset.era_dataset.tokenizer)

if 'offset' in model.tokenizer.tok2idx:
  for i in range(120):
    if i % 4 == 0:
      continue
    if i / 4 not in model.tokenizer.tok2idx['offset']:
      model.tokenizer.tok2idx['offset'][i/4] = model.tokenizer.tok2idx['offset'][(i-2)/4]

model_alt.tokenizer = model.tokenizer

print("Start inference")
score = stream.Score(id='mainScore')

prev_generation = None
outputs = []
source_part = stream.Part()
merged_part = stream.Part() 
srcs = []

for target_idx in range(6):
  outputs = []
  prev_generation = None
  for i in tqdm(range(len(test_set))):
    sample, _, _ = test_set[i]
    src, output_decoded, (attention_map, output, new_out) = model.shifted_inference(sample, target_idx, prev_generation=prev_generation)
    if target_idx == 0 and i % test_set.slice_measure_number == 0:
      srcs.append(src)
    if i == 0:
      sel_out = torch.cat([output[0:1]] + [get_measure_specific_output(output, i) for i in range(3,6)], dim=0)
    else:
      sel_out = get_measure_specific_output(output, 5) 
      duration = sum([note[2] for note in model.converter(sel_out)])
      if abs(duration - 30) > 0.1:
        print(f"duration error: {duration}, inst: {target_idx}, measure: {i}")
        src, output_decoded, (attention_map, output, new_out) = model_alt.shifted_inference(sample, target_idx, prev_generation=prev_generation)
        sel_out = get_measure_specific_output(output, 5)
        duration = sum([note[2] for note in model.converter(sel_out)])
        assert abs(duration - 30) < 0.1, f"duration error: {duration}, inst: {target_idx}, measure: {i}"
    prev_generation = get_measure_shifted_output(output)
    outputs.append(sel_out)

  if target_idx == 0:
    srcs.append(fill_in_source(src, i % test_set.slice_measure_number))
    source_converted = [y for x in srcs for y in x]
    source_part = source_decoder(source_converted[1:])
    score.insert(0, source_part)

  outputs.append(get_measure_specific_output(output, 6)) # add last measure
  outputs_tensor = torch.cat(outputs, dim=0)
  final_midi = decoder(model.converter(outputs_tensor))
  score.insert(0, final_midi)


score.write('musicxml', output_dir/ f'CHP_Orchestration{save_name}.musicxml')
score.write('midi', output_dir/ f'CHP_Orchestration{save_name}.mid')