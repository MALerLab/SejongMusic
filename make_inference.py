import argparse
import datetime
from pathlib import Path
from omegaconf import OmegaConf
from fractions import Fraction
from tqdm.auto import tqdm

import torch
from music21 import converter, stream, note as m21_note 

from sejong_music import model_zoo
from sejong_music.yeominrak_processing import AlignedScore, SamplingScore, pack_collate, ShiftedAlignedScore, TestScore, TestScoreCPH, Tokenizer
from sejong_music.model_zoo import QkvAttnSeq2seq, get_emb_total_size, QkvAttnSeq2seqMore
from sejong_music.decode import MidiDecoder
from sejong_music.inference_utils import prepare_input_for_next_part, get_measure_specific_output, fix_measure_idx, fill_in_source, get_measure_shifted_output, recover_beat




def get_argparse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_name', type=str, default='CPH_from_1')
  parser.add_argument('--state_dir', type=str, default='outputs/2024-03-08/00-27-49/wandb/run-20240308_002750-ui4ko24t/files/checkpoints')
  parser.add_argument('--input_xml', type=str, default='music_score/chwipunghyeong.musicxml')
  # parser.add_argument('--state_path', type=str, default='outputs/2023-12-20/14-46-58/best_pitch_sim_model.pt')
  # parser.add_argument('--state_alt_path', type=str, default='outputs/2023-12-20/14-46-58/best_loss_model.pt')
  parser.add_argument('--config_path', type=str, default='yamls/baseline.yaml')
  parser.add_argument('--output_dir', type=str, default='gen_results/')
  return parser

if __name__ == '__main__':
  args = get_argparse().parse_args()
  # save_name = 'dropout0.5_bestacc'
  # state = torch.load('outputs/2023-12-12/04-08-05/best_model.pt')

  # save_name = 'seq2seq_bestacc_token_fixed'
  # # state = torch.load('outputs/2023-12-12/04-46-10/best_model.pt')
  # state = torch.load('outputs/2023-12-13/11-19-42/best_pitch_sim_model.pt') # best pitch sim model

  save_name = f'{args.save_name}_{datetime.datetime.now().strftime("%m%d-%H%M")}'
  # state = torch.load(f'{args.state_dir}/best_pitch_sim_model.pt')
  # state_alt = torch.load(f'{args.state_dir}/best_loss_model.pt')
  state = torch.load(f'{args.state_dir}/last_model.pt')
  state_alt = torch.load(f'{args.state_dir}/best_pitch_sim_model.pt')

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  # config = OmegaConf.load('yamls/baseline.yaml')
  config = OmegaConf.load(f'{args.state_dir}/config.yaml')
  tokenizer = Tokenizer(parts=None, feature_types=None, json_fn=f'{args.state_dir}/tokenizer_vocab.json')
  # config = get_emb_total_size(config)
  device = 'cpu'
  # val_dataset = ShiftedAlignedScore(is_valid= True, slice_measure_num=4, feature_types=config.model.features)
  # valid_loader = DataLoader(val_dataset, batch_size=4, collate_fn=pack_collate, shuffle=False)
  model = getattr(model_zoo, config.model_class)(tokenizer, config.model).to(device)
  model.is_condition_shifted = True
  model.load_state_dict(state)
  model.eval()


  decoder = MidiDecoder(tokenizer)

  if 'chihwapyung' in args.input_xml:
    test_set = TestScore(xml_path='chihwapyung.musicxml', is_valid=True, 
                        valid_measure_num='entire', slice_measure_num=4, transpose=+3, feature_types=config.model.features)
    beat_recover_amount = [5, 5, 6, 6, 6, 6, 7, 7]
    start_idx = 0
    target_idx = start_idx + 1
  elif 'chwipunghyeong' in args.input_xml:
    test_set = TestScoreCPH(xml_path=args.input_xml, is_valid=True, 
                        #  valid_measure_num=[i for i in range(133)], 
                        valid_measure_num='entire',
                        slice_measure_num=4, transpose=+8, feature_types=config.model.features)
    beat_recover_amount = [3, 3, 3, 3, 3, 3, 3, 3]
    start_idx = 1
    target_idx = start_idx + 1
  else:
    raise ValueError(f'Invalid input xml path: {args.input_xml}')
    

  test_set.tokenizer = tokenizer

  for i in range(30):
    if i % 3 == 0:
      continue
    if Fraction(i, 3) in model.tokenizer.tok2idx['offset_fraction']:
      continue
      print("yes")
    else:
      if i < 6:
        other_value = i % 3 + 6
      else:
        other_value = i % 3 + 21
      model.tokenizer.tok2idx['offset_fraction'][Fraction(i, 3)] = model.tokenizer.tok2idx['offset_fraction'][Fraction(other_value, 3)]


  model_alt = getattr(model_zoo, config.model_class)(model.tokenizer, config.model).to(device)
  model_alt.is_condition_shifted = True
  model_alt.load_state_dict(state_alt)
  model_alt.eval()



  print("Start inference")
  score = stream.Score(id='mainScore')

  prev_generation = None
  outputs = []
  source_part = stream.Part()
  merged_part = stream.Part() 
  srcs = []

  for i in tqdm(range(len(test_set))):
    sample = test_set[i]
    sample[:,0] = start_idx # regard as era 1
    src, output_decoded, (attention_map, output, new_out) = model.shifted_inference(sample, 
                                                                                    target_idx, 
                                                                                    prev_generation=prev_generation, 
                                                                                    fix_first_beat=False, 
                                                                                    compensate_beat=(beat_recover_amount[sample[0,0]], beat_recover_amount[target_idx]))
    if i % test_set.slice_measure_number == 0:
      srcs.append(src)
    if i == 0:
      sel_out = torch.cat([output[0:1]] + [get_measure_specific_output(output, i) for i in range(3,6)], dim=0)
    else:
      sel_out = get_measure_specific_output(output, 5)
      sel_duration = sum([x[2] for x in model.converter(sel_out)])
      if abs(sel_duration - model.tokenizer.measure_duration[target_idx]) > 0.001:
        print(f'Generated duration: {sel_duration}, target idx: {target_idx}, Running Alt model inference')
        src, output_decoded, (attention_map, output, new_out) = model_alt.shifted_inference(sample, target_idx, prev_generation=prev_generation, fix_first_beat=False, compensate_beat=(beat_recover_amount[sample[0,0]], beat_recover_amount[target_idx]))
        sel_out = get_measure_specific_output(output, 5) 
        sel_duration = sum([x[2] for x in model.converter(sel_out)])
        assert abs(sel_duration - model.tokenizer.measure_duration[target_idx]) < 0.001
    prev_generation = get_measure_shifted_output(output)
    outputs.append(sel_out)

  srcs.append(fill_in_source(src, i % test_set.slice_measure_number))
  source_converted = [y for x in srcs for y in x]
  source_part = decoder(source_converted[1:])
  outputs.append(get_measure_specific_output(output, 6)) # add last measure
  outputs_tensor = torch.cat(outputs, dim=0)
  final_midi = decoder(model.converter(recover_beat(outputs_tensor, model.tokenizer, beat_recover_amount[target_idx])))
  score.insert(0, source_part)
  score.insert(0, final_midi)

  # final_midi.write('midi', f'sequential_inference_CHP_{2}.mid')

  next_input = prepare_input_for_next_part(outputs_tensor)
  measure_boundary = [0] + torch.where(torch.diff(next_input[:,-1]) > 0)[0].tolist() + [len(next_input)]
  num_measures = len(measure_boundary)



  output_by_part = []
  for target_idx in range(start_idx+2, 8):
    start_token = torch.tensor([[outputs_tensor[0,0], 1, 1, 1, 1, 1]])
    end_token = torch.tensor([[outputs_tensor[0,0], 2, 2, 2 , 2, 2]])
    prev_generation = None
    outputs =[]
    for i in tqdm(range(num_measures-4)):
      measure_start = measure_boundary[i]
      measure_end = measure_boundary[i+4]
      sample = next_input[measure_start:measure_end]
      sample = torch.cat([start_token, sample, end_token], dim=0)
      sample = fix_measure_idx(sample)
      assert max(sample[:,-1]) == 6, f"Last measure should be idx6, 4th measure {max(sample[:,-1])} \n{sample}"
      src, output_decoded, (attention_map, output, new_out) = model.shifted_inference(sample, target_idx, prev_generation=prev_generation, fix_first_beat=False, compensate_beat=(beat_recover_amount[sample[0,0]], beat_recover_amount[target_idx]))

      if i == 0:
        sel_out = torch.cat([output[0:1]] + [get_measure_specific_output(output, i) for i in range(3,6)], dim=0)
      else:
        sel_out = get_measure_specific_output(output, 5) 
        sel_duration = sum([x[2] for x in model.converter(sel_out)])
        while abs(sel_duration - model.tokenizer.measure_duration[target_idx]) > 0.001:
          print(f'Generated duration: {sel_duration}, target idx: {target_idx}, Running Alt model inference')
          src, output_decoded, (attention_map, output, new_out) = model_alt.shifted_inference(sample, target_idx, prev_generation=prev_generation, fix_first_beat=False, compensate_beat=(beat_recover_amount[sample[0,0]], beat_recover_amount[target_idx]))
          sel_out = get_measure_specific_output(output, 5) 
          sel_duration = sum([x[2] for x in model.converter(sel_out)])
      
      prev_generation = get_measure_shifted_output(output)
      outputs.append(sel_out)
    outputs.append(get_measure_specific_output(output, 6)) # add last measure
    outputs_tensor = torch.cat(outputs, dim=0)
    next_input = prepare_input_for_next_part(outputs_tensor)
    measure_boundary = [0] + torch.where(torch.diff(next_input[:,-1]) > 0)[0].tolist() + [len(next_input)]
    num_measures = len(measure_boundary)
    output_by_part.append(outputs_tensor)
    final_midi = decoder(model.converter(recover_beat(outputs_tensor, model.tokenizer, beat_recover_amount[target_idx])))
    merged_part = stream.Part()
    for element in final_midi:
      merged_part.append(element)
    score.insert(0, merged_part)
    # final_midi.write('midi', output_dir / f'sequential_inference_CHP_{target_idx}.mid')

  score.write('musicxml', output_dir/ f'CHP_score{save_name}.musicxml')
  score.write('midi', output_dir/ f'CHP_score{save_name}.mid')