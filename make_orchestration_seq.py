import argparse
import datetime
import copy
from pathlib import Path
from omegaconf import OmegaConf
from fractions import Fraction
from tqdm.auto import tqdm

import torch
from music21 import converter, stream, note as m21_note 

from sejong_music import model_zoo
from sejong_music.yeominrak_processing import OrchestraScoreSeq, ShiftedAlignedScore, TestScore, TestScoreCPH, Tokenizer
from sejong_music.model_zoo import QkvAttnSeq2seq, get_emb_total_size, QkvAttnSeq2seqMore
from sejong_music.decode import MidiDecoder, OrchestraDecoder
from sejong_music.inference_utils import prepare_input_for_next_part, get_measure_specific_output, fix_measure_idx, fill_in_source, get_measure_shifted_output, recover_beat, round_number
from sejong_music.inference import Inferencer

def get_argparse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_name', type=str, default='CPH_Orch')
  parser.add_argument('--gmg_state_dir', type=str, default='outputs/2024-03-08/23-02-48/wandb/run-20240308_230249-trfvg2ki/files/checkpoints')
  parser.add_argument('--gyg_state_dir', type=str, default='outputs/2024-03-08/23-13-35/wandb/run-20240308_231336-b5coyy91/files/checkpoints')
  parser.add_argument('--hg_state_dir', type=str, default='outputs/2024-03-08/20-14-57/wandb/run-20240308_201458-2hzc5z6j/files/checkpoints')
  parser.add_argument('--pr_state_dir', type=str, default='outputs/2024-03-08/19-54-14/wandb/run-20240308_195415-nsoza2ld/files/checkpoints')
  parser.add_argument('--dg_state_dir', type=str, default='outputs/2024-03-08/19-42-17/wandb/run-20240308_194218-kojvditr/files/checkpoints')
  parser.add_argument('--inst_state_dir', type=str, default='outputs/2024-03-23/22-46-33/wandb/run-20240323_224634-x589hybg/files/checkpoints')
  parser.add_argument('--input_xml', type=str, default='gen_results/CHP_scoreCPH_from_1_0308-1209.musicxml')
  # parser.add_argument('--state_path', type=str, default='outputs/2023-12-20/14-46-58/best_pitch_sim_model.pt')
  # parser.add_argument('--state_alt_path', type=str, default='outputs/2023-12-20/14-46-58/best_loss_model.pt')
  parser.add_argument('--config_path', type=str, default='yamls/baseline.yaml')
  parser.add_argument('--output_dir', type=str, default='gen_results/')
  parser.add_argument('--use_total_model', action='store_true')
  return parser

if __name__ == "__main__":
  args = get_argparse().parse_args()
  save_name = f'{args.save_name}_{datetime.datetime.now().strftime("%m%d-%H%M")}'
  inst_codes = ['gmg', 'gyg', 'hg', 'pr', 'dg']
  inst_codes_to_idx = {'dg': 0, 'pr': 1, 'hg': 2, 'gyg': 3, 'gmg': 4}
  states = {}
  configs = {}
  gmg_config = OmegaConf.load(f'{args.gmg_state_dir}/config.yaml')
  # gmg_config = '/home/danbi/userdata/DANBI/gugakwon/SejongMusic/yamls/orchestration.yaml'

  for inst_code in inst_codes:
    if inst_code == 'gmg':
      state_dir = getattr(args, f'{inst_code}_state_dir')
      configs[inst_code] = OmegaConf.load(f'{state_dir}/config.yaml')
    else:
      if args.use_total_model:
        state_dir = args.inst_state_dir + '/inst_0'
      else:
        state_dir = args.inst_state_dir + f'/inst_{inst_codes_to_idx[inst_code]}'
      configs[inst_code] = OmegaConf.load(f'{args.inst_state_dir}/config.yaml')

    states[inst_code] = torch.load(f'{state_dir}/epoch600_model.pt')
  tokenizer = Tokenizer(parts=None, feature_types=None, json_fn=f'{args.gmg_state_dir}/tokenizer_vocab.json')
  era_tokenizer = Tokenizer(parts=None, feature_types=None, json_fn=f'{args.gmg_state_dir}/era_tokenizer_vocab.json')
  
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  device = 'cuda'

  test_set = ShiftedAlignedScore(is_valid= True,
                                 xml_path = args.input_xml,
                                 valid_measure_num = 'entire',
                                 slice_measure_num=gmg_config.data.max_meas,
                                 min_meas=gmg_config.data.min_meas,
                                 feature_types=gmg_config.model.features,
                                 sampling_rate=gmg_config.data.sampling_rate)

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
        if i >= 6:
          tokenizer.tok2idx['offset'][i/4] = tokenizer.tok2idx['offset'][(i-6)/4]
        else:
          tokenizer.tok2idx['offset'][i/4] = tokenizer.tok2idx['offset'][(i-2)/4]



  gmg_model = getattr(model_zoo, gmg_config.model_class)(era_tokenizer, tokenizer, gmg_config.model).to(device)
  gmg_model.is_condition_shifted = True
  gmg_model.load_state_dict(states['gmg'])
  gmg_model.eval()
  gmg_model = Inferencer(gmg_model, True, True, False, temperature=1.0, top_p=0.8)
  

  decoder = OrchestraDecoder(tokenizer)
  source_decoder = MidiDecoder(era_tokenizer)

  print("Start inference")

  prev_generation = None
  outputs = []
  source_part = stream.Part()
  merged_part = stream.Part()
  empty_part = stream.Part()
  srcs = []

  #convert era to Geomungo score
  score = stream.Score(id='mainScore')
  gen_by_insts = {}
  outputs = []
  prev_generation = None
  model = gmg_model
  target_idx = 4
  for i in tqdm(range(len(test_set))):
    sample, _, _ = test_set[i]
    src, output_decoded, (attention_map, output, new_out) = model.inference(sample, target_idx, prev_generation=prev_generation)
    if i % test_set.slice_measure_number == 0:
      srcs.append(src)
    if i == 0:
      sel_out = torch.cat([output[0:1]] + [get_measure_specific_output(output, i) for i in range(3,6)], dim=0)
    else:
      sel_out = get_measure_specific_output(output, 5) 
      duration = sum([note[2] for note in model.converter(sel_out)])
      while abs(duration - 30) > 0.1:
        print(f"duration error: {duration}, inst: {target_idx}, measure: {i}")
        src, output_decoded, (attention_map, output, new_out) = model.inference(sample, target_idx, prev_generation=prev_generation)
        sel_out = get_measure_specific_output(output, 5)
        duration = sum([note[2] for note in model.converter(sel_out)])
        # assert abs(duration - 30) < 0.1, f"duration error: {duration}, inst: {target_idx}, measure: {i}"
    prev_generation = get_measure_shifted_output(output)
    outputs.append(sel_out)
  outputs.append(get_measure_specific_output(output, 6)) # add last measure
  outputs_tensor = torch.cat(outputs, dim=0)
  gen_by_insts['gmg'] = decoder(model.converter(outputs_tensor))
  for _ in range(len(inst_codes)):
    score.insert(0, copy.copy(gen_by_insts['gmg']))

  prev_xml_path = output_dir/ f'{save_name}_seq_gmg.musicxml'
  score.write('musicxml', prev_xml_path)

  # Make Sequential
  
  for tg_idx, inst in enumerate(inst_codes[1:], start=1):
    target_idx = len(inst_codes) - tg_idx - 1
    print(f"target_idx: {target_idx}, inst: {inst}")


  
    next_dataset = OrchestraScoreSeq(is_valid=True,
                                    xml_path=prev_xml_path,
                                    valid_measure_num = 'entire',
                                    slice_measure_num=configs[inst].data.max_meas,
                                    min_meas=configs[inst].data.min_meas,
                                    feature_types=configs[inst].model.features,
                                    sampling_rate=configs[inst].data.sampling_rate,
                                    target_instrument=target_idx)
    # next_dataset.tokenizer = tokenizer
    # state_dir = getattr(args, f'{inst}_state_dir')
    state_dir = args.inst_state_dir
    tokenizer = Tokenizer(parts=next_dataset.parts, feature_types=configs[inst].model.features, json_fn=f'{state_dir}/tokenizer_vocab.json')
    next_dataset.tokenizer = tokenizer
    model = getattr(model_zoo, configs[inst].model_class)(tokenizer, configs[inst].model).to(device)
    model.is_condition_shifted = True
    model.load_state_dict(states[inst])
    model.eval()
    model = Inferencer(model, is_condition_shifted=True, is_orch=True, is_sep=True, temperature=1.0, top_p=0.9)

    next_dataset.result_pairs = [x for x in next_dataset.result_pairs if x[1] == 1]
    score = stream.Score(id='mainScore')
    outputs = []
    prev_generation = None
    for i in tqdm(range(len(next_dataset))):
      sample, _, _ = next_dataset[i]
      _, output_decoded, (attention_map, output, new_out) = model.inference(sample, target_idx, prev_generation=prev_generation)
      if i == 0:
        sel_out = torch.cat([output[0:1]] + [get_measure_specific_output(output, i) for i in range(3,6)], dim=0)
      else:
        sel_out = get_measure_specific_output(output, 5) 
        duration = sum([note[2] for note in model.converter(sel_out)])
        num_note_on_beat = sum([1 for note in model.converter(sel_out) if note[3] % 1.5 == 0]) / len(sel_out)
        while abs(duration - 30) > 0.1 or num_note_on_beat < 0.3:
          print(f"duration error: {duration}, inst: {target_idx}, measure: {i}, num_note_on_beat: {num_note_on_beat}")
          _, output_decoded, (attention_map, output, new_out) = model.inference(sample, target_idx, prev_generation=prev_generation)
          sel_out = get_measure_specific_output(output, 5)
          duration = sum([note[2] for note in model.converter(sel_out)])
          num_note_on_beat = sum([1 for note in model.converter(sel_out) if note[3] % 1.5 == 0]) / len(sel_out)
          # assert abs(duration - 30) < 0.1, f"duration error: {duration}, inst: {target_idx}, measure: {i}"
      prev_generation = get_measure_shifted_output(output)
      outputs.append(sel_out)
    outputs.append(get_measure_specific_output(output, 6)) # add last measure
    outputs_tensor = torch.cat(outputs, dim=0)
    gen_by_insts[inst] = decoder(model.converter(outputs_tensor))
    for _ in range(target_idx+1):
      score.insert(0, copy.copy(gen_by_insts[inst]))
    
    for prev_inst in reversed(inst_codes[:tg_idx]):
      score.insert(0, gen_by_insts[prev_inst])
        
    prev_xml_path = output_dir/ f'{save_name}_seq_{inst}.musicxml'
    print(f"write {prev_xml_path}")
    score.write('musicxml', prev_xml_path)
  
  score.write('midi', output_dir/ f'{save_name}_seq.midi')
    