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
from sejong_music.yeominrak_processing import OrchestraScoreSeq, ShiftedAlignedScore, Tokenizer
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
  # inst_codes = ['gmg', 'gyg', 'hg', 'pr', 'dg']
  inst_codes = ['pr','gmg', 'gyg', 'hg',  'dg']
  inst_codes_to_idx = {'dg': 0, 'pr': 1, 'hg': 2, 'gyg': 3, 'gmg': 4}
  states = {}
  configs = {}
  gmg_config = OmegaConf.load(f'{args.gmg_state_dir}/config.yaml')
  # gmg_config = '/home/danbi/userdata/DANBI/gugakwon/SejongMusic/yamls/orchestration.yaml'

  for inst_code in inst_codes:
    if inst_code == 'gmg':
      state_dir = getattr(args, f'{inst_code}_state_dir')
      configs[inst_code] = OmegaConf.load(f'{state_dir}/config.yaml')
      states[inst_code] = torch.load(f'{state_dir}/inst_0/epoch1000_model.pt')
    else:
      if args.use_total_model:
        state_dir = args.inst_state_dir + '/inst_0'
      else:
        state_dir = args.inst_state_dir + f'/inst_{inst_codes_to_idx[inst_code]}'
      configs[inst_code] = OmegaConf.load(f'{args.inst_state_dir}/config.yaml')

      states[inst_code] = torch.load(f'{state_dir}/epoch1000_model.pt')
  tokenizer = Tokenizer(parts=None, feature_types=None, json_fn=f'{args.inst_state_dir}/tokenizer_vocab.json')
  era_tokenizer = Tokenizer(parts=None, feature_types=None, json_fn=f'{args.inst_state_dir}/era_tokenizer_vocab.json')
  
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  device = 'cuda'

  test_set = OrchestraScoreSeq(is_valid= True,
                              xml_path = args.input_xml,
                              valid_measure_num = 'entire',
                              slice_measure_num=gmg_config.data.max_meas,
                              min_meas=gmg_config.data.min_meas,
                              feature_types=gmg_config.model.features,
                              sampling_rate=gmg_config.data.sampling_rate,
                              target_instrument=4)

  # filter pairs to sweep the piece only once
  inst = 'gyg'
  model = getattr(model_zoo, configs[inst].model_class)(tokenizer, configs[inst].model).to(device)
  model.is_condition_shifted = True
  model.load_state_dict(states[inst])
  model.eval()
  model = Inferencer(model, is_condition_shifted=True, is_orch=True, is_sep=True, temperature=1.5, top_p=0.75)


  decoder = OrchestraDecoder(tokenizer)
  source_decoder = OrchestraDecoder(tokenizer)

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

  srcs.append(fill_in_source(src, i % test_set.slice_measure_number))
  source_converted = [y for x in srcs for y in x]
  source_part = source_decoder(source_converted)
  gen_by_insts['pr'] = source_part

  outputs.append(get_measure_specific_output(output, 6)) # add last measure
  outputs_tensor = torch.cat(outputs, dim=0)
  gen_by_insts['gmg'] = decoder(model.converter(outputs_tensor))
  for _ in range(2):
    score.insert(0, copy.copy(gen_by_insts['pr']))
    
  for _ in range(len(inst_codes)-2):
    score.insert(0, copy.copy(gen_by_insts['gmg']))

  prev_xml_path = output_dir/ f'{save_name}_seq_gmg.musicxml'
  score.write('musicxml', prev_xml_path)

  # Make Sequential
  
  for tg_idx, inst in enumerate(inst_codes[2:], start=2):
    # target_idx = len(inst_codes) - tg_idx - 1
    target_idx = inst_codes_to_idx[inst]
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
    if target_idx == 0:
      score.insert(0, copy.copy(gen_by_insts['dg']))
      score.insert(0, copy.copy(gen_by_insts['pr']))
      score.insert(0, copy.copy(gen_by_insts['hg']))
      score.insert(0, copy.copy(gen_by_insts['gyg']))
      score.insert(0, copy.copy(gen_by_insts['gmg']))
    elif target_idx == 2:
      score.insert(0, copy.copy(gen_by_insts['pr']))
      score.insert(0, copy.copy(gen_by_insts['pr']))
      score.insert(0, copy.copy(gen_by_insts['hg']))
      score.insert(0, copy.copy(gen_by_insts['gyg']))
      score.insert(0, copy.copy(gen_by_insts['gmg']))
    elif target_idx == 3:
      score.insert(0, copy.copy(gen_by_insts['pr']))
      score.insert(0, copy.copy(gen_by_insts['pr']))
      score.insert(0, copy.copy(gen_by_insts['pr']))
      score.insert(0, copy.copy(gen_by_insts['gyg']))
      score.insert(0, copy.copy(gen_by_insts['gmg']))
        
    prev_xml_path = output_dir/ f'{save_name}_seq_{inst}.musicxml'
    print(f"write {prev_xml_path}")
    score.write('musicxml', prev_xml_path)
  
  score.write('midi', output_dir/ f'{save_name}_seq.midi')
    