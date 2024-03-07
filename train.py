import random
import datetime
from pathlib import Path
from fractions import Fraction
import wandb 
import hydra

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from sejong_music import yeominrak_processing, model_zoo, trainer
from sejong_music.yeominrak_processing import pack_collate
from sejong_music.model_zoo import get_emb_total_size
from sejong_music.loss import nll_loss, focal_loss
from sejong_music.trainer import Trainer, RollTrainer

def make_experiment_name_with_date(config):
  current_time_in_str = datetime.datetime.now().strftime("%m%d-%H%M")
  return f'{current_time_in_str}_{config.general.exp_name}_{config.dataset_class}_{config.model_class}'

@hydra.main(config_path='./yamls/', config_name='baseline')
def main(config: DictConfig):
  config = get_emb_total_size(config)
  if 'duration' not in config.model.features:
    config.model.features.append('duration')
  if 'sampling_rate' in config.data and 'samlping_rate' not in config.model:
    config.model.sampling_rate = config.data.sampling_rate

  if config.general.make_log:
    wandb.init(
      project="yeominrak", 
      entity="dasaem", 
      name = make_experiment_name_with_date(config), 
      config = OmegaConf.to_container(config)
    )
    save_dir = Path(wandb.run.dir) / 'checkpoints'
  else:
    save_dir = Path('wandb/debug/checkpoints')

  original_wd = Path(hydra.utils.get_original_cwd())
  if not save_dir.is_absolute():
    save_dir = original_wd / save_dir
  
  dataset_class = getattr(yeominrak_processing, config.dataset_class)
  model_class = getattr(model_zoo, config.model_class)

  train_dataset = dataset_class(is_valid= False, 
                                xml_path=original_wd/'music_score/yeominlak.musicxml',
                                use_pitch_modification = config.data.use_pitch_modification, 
                                pitch_modification_ratio=config.data.modification_ratio,
                                min_meas=config.data.min_meas, 
                                max_meas=config.data.max_meas,
                                feature_types=config.model.features,
                                sampling_rate=config.data.sampling_rate
                                )
  val_dataset = dataset_class(is_valid= True,
                              valid_measure_num = [i for i in range(93, 99)],
                              xml_path=original_wd/'music_score/yeominlak.musicxml', 
                              use_pitch_modification=False, 
                              slice_measure_num=config.data.max_meas,
                              min_meas=config.data.min_meas,
                              feature_types=config.model.features,
                                sampling_rate=config.data.sampling_rate)
  test_dataset = dataset_class(is_valid= True,
                              valid_measure_num = [i for i in range(99, 104)],
                              xml_path=original_wd/'music_score/yeominlak.musicxml', 
                              use_pitch_modification=False, 
                              slice_measure_num=config.data.max_meas,
                              min_meas=config.data.min_meas,
                              feature_types=config.model.features,
                                sampling_rate=config.data.sampling_rate)
  train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size , shuffle=True, collate_fn=pack_collate)
  valid_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=pack_collate, drop_last=True)
  test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=pack_collate, drop_last=True)

  tokenizer_vocab_path = save_dir / 'tokenizer_vocab.json'
  
  
  device = 'cuda'

  measure_match_exp = []
  similarity_exp = []
  beat_pitch_match_exp = []
  total_iteration = 0

  for seed_idx in range(config.general.num_exp): # run experiment 5 times with different seed
    random.seed(seed_idx)
    torch.manual_seed(seed_idx)
    torch.cuda.manual_seed(seed_idx)

    model = model_class(train_dataset.tokenizer, config.model).to(device)

    if 'offset_fraction' in model.tokenizer.tok2idx:
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


    model.is_condition_shifted = (config.dataset_class == "ShiftedAlignedScore")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.99)
    scheduler = None
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'config.yaml', 'w') as f:
      OmegaConf.save(config, f)
    train_dataset.tokenizer.save_to_json(tokenizer_vocab_path)

    loss_fn = nll_loss

    if config.dataset_class == "SamplingScore":
      trainer_class = "RollTrainer"
    else:
      trainer_class = "Trainer"
    
    atrainer:Trainer = getattr(trainer, trainer_class)(model=model, optimizer=optimizer, 
                      loss_fn=loss_fn, train_loader=train_loader, 
                      valid_loader=valid_loader, device = device, save_log=config.general.make_log, 
                      save_dir=save_dir, 
                      scheduler=scheduler)
    atrainer.iteration = total_iteration
    
    atrainer.train_by_num_epoch(config.train.num_epoch)

    atrainer.load_best_model()

    atrainer.make_inference_result(write_png=True)
    atrainer.make_sequential_inference_result(write_png=True)
    measure_match, similarity, beat_pitch_match = atrainer.make_inference_result(loader=test_loader)
    measure_match_exp.append(measure_match)
    similarity_exp.append(similarity)
    beat_pitch_match_exp.append(beat_pitch_match)
    total_iteration = atrainer.iteration

  if config.general.make_log:
    # wandb.summary['test_measure_match'] = measure_match
    # wandb.summary['test_similarity'] = similarity
    # wandb.summary['test_beat_pitch_match'] = beat_pitch_match
    wandb.log({'test_measure_match_best': max(measure_match_exp), 
                'test_similarity_best': max(similarity_exp),
                'test_beat_pitch_match_best': max(beat_pitch_match_exp),
                'test_measure_match_mean': sum(measure_match_exp)/len(measure_match_exp),
                'test_similarity_mean': sum(similarity_exp)/len(similarity_exp),
                'test_beat_pitch_match_mean': sum(beat_pitch_match_exp)/len(beat_pitch_match_exp),
                'test_measure_match_exp': measure_match_exp,
                'test_similarity_exp': similarity_exp,
                'test_beat_pitch_match_exp': beat_pitch_match_exp})




if __name__ == '__main__':
  # config = OmegaConf.load('./yamls/baseline.yaml')
  main()
  