import argparse
import datetime
from pathlib import Path
from omegaconf import OmegaConf
from fractions import Fraction
from tqdm.auto import tqdm

import torch
from music21 import converter, stream, note as m21_note 

from sejong_music import model_zoo
from sejong_music.yeominrak_processing import OrchestraScore, ShiftedAlignedScore, TestScore, TestScoreCPH, Tokenizer
from sejong_music.model_zoo import QkvAttnSeq2seq, get_emb_total_size, QkvAttnSeq2seqMore
from sejong_music.decode import MidiDecoder, OrchestraDecoder
from sejong_music.inference_utils import prepare_input_for_next_part, get_measure_specific_output, fix_measure_idx, fill_in_source, get_measure_shifted_output, recover_beat, round_number


def get_argparse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_name', type=str, default='CPH_Orch')
  parser.add_argument('--state_dir', type=str, default='outputs/2024-03-08/16-17-44/wandb/run-20240308_161745-x9ao6gpf/files/checkpoints')
  parser.add_argument('--input_xml', type=str, default='gen_results/CHP_scoreCPH_from_1_0308-1209.musicxml')
  # parser.add_argument('--state_path', type=str, default='outputs/2023-12-20/14-46-58/best_pitch_sim_model.pt')
  # parser.add_argument('--state_alt_path', type=str, default='outputs/2023-12-20/14-46-58/best_loss_model.pt')
  parser.add_argument('--config_path', type=str, default='yamls/baseline.yaml')
  parser.add_argument('--output_dir', type=str, default='gen_results/')
  return parser.parse_args()

if __name__ == "__main__":
  args = get_argparse()
  save_name = f'{args.save_name}_{datetime.datetime.now().strftime("%m%d-%H%M")}'

  state = torch.load(f'{args.state_dir}/last_model.pt')
  state_alt = torch.load(f'{args.state_dir}/best_loss_model.pt')

  config = OmegaConf.load(f'{args.state_dir}/config.yaml')
  tokenizer = Tokenizer(parts=None, feature_types=None, json_fn=f'{args.state_dir}/tokenizer_vocab.json')
  era_tokenizer = Tokenizer(parts=None, feature_types=None, json_fn=f'{args.state_dir}/era_tokenizer_vocab.json')
  
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  device = 'cpu'

  test_set = ShiftedAlignedScore(is_valid= True,
                                 xml_path = args.input_xml,
                                 valid_measure_num = 'entire',
                                 slice_measure_num=config.data.max_meas,
                                 min_meas=config.data.min_meas,
                                 feature_types=config.model.features,
                                 sampling_rate=config.data.sampling_rate)

  # filter pairs to sweep the piece only once
  test_set.result_pairs = [x for x in test_set.result_pairs if x[1] == 1][:len(test_set.slice_info)]
  for pair in test_set.result_pairs:
    pair[0] = len(test_set.parts) - 1
    pair[1] = len(test_set.parts) - 1
  test_set.tokenizer = era_tokenizer


  
  for i in range(30):
    offset = round_number(i/3, 3)
    if offset in era_tokenizer.tok2idx['offset']:
      continue
    if i < 6:
      other_value = i % 3 + 6
    else:
      other_value = i % 3 + 21
    era_tokenizer.tok2idx['offset'][offset] = era_tokenizer.tok2idx['offset'][round_number(other_value/3, 3)]

  for i in range(30):
    offset = round_number(i/2, 3)
    if offset in era_tokenizer.tok2idx['offset']:
      continue
    other_value = i - 2
    era_tokenizer.tok2idx['offset'][offset] = era_tokenizer.tok2idx['offset'][round_number(other_value/2, 3)]
  
  if 'offset' in tokenizer.tok2idx:
    for i in range(120):
      if i % 4 == 0:
        continue
      if i / 4 not in tokenizer.tok2idx['offset']:
        tokenizer.tok2idx['offset'][i/4] = tokenizer.tok2idx['offset'][(i-2)/4]



  model = getattr(model_zoo, config.model_class)(era_tokenizer, tokenizer, config.model).to(device)
  model.is_condition_shifted = True
  model.load_state_dict(state)
  model.eval()

  model_alt = getattr(model_zoo, config.model_class)(era_tokenizer, tokenizer, config.model).to(device)
  model_alt.is_condition_shifted = True
  model_alt.load_state_dict(state_alt)
  model_alt.eval()

  decoder = OrchestraDecoder(tokenizer)
  source_decoder = MidiDecoder(era_tokenizer)

  print("Start inference")
  score = stream.Score(id='mainScore')

  prev_generation = None
  outputs = []
  source_part = stream.Part()
  merged_part = stream.Part() 
  srcs = []

  # for target_idx in range(6):
  for target_idx in range(5):
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
        while abs(duration - 30) > 0.1:
          print(f"duration error: {duration}, inst: {target_idx}, measure: {i}")
          src, output_decoded, (attention_map, output, new_out) = model_alt.shifted_inference(sample, target_idx, prev_generation=prev_generation)
          sel_out = get_measure_specific_output(output, 5)
          duration = sum([note[2] for note in model.converter(sel_out)])
          # assert abs(duration - 30) < 0.1, f"duration error: {duration}, inst: {target_idx}, measure: {i}"
      prev_generation = get_measure_shifted_output(output)
      outputs.append(sel_out)

    if target_idx == 0:
      srcs.append(fill_in_source(src, i % test_set.slice_measure_number))
      source_converted = [y for x in srcs for y in x]
      source_part = source_decoder(source_converted)
      score.insert(0, source_part)

    outputs.append(get_measure_specific_output(output, 6)) # add last measure
    outputs_tensor = torch.cat(outputs, dim=0)
    final_midi = decoder(model.converter(outputs_tensor))
    score.insert(0, final_midi)


  score.write('musicxml', output_dir/ f'{save_name}.musicxml')
  score.write('midi', output_dir/ f'{save_name}.mid')