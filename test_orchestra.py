
import datetime
import random
import wandb
import hydra
from pathlib import Path
import copy

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from sejong_music import yeominrak_processing, model_zoo
from sejong_music.yeominrak_processing import pack_collate
from sejong_music.model_zoo import get_emb_total_size
from sejong_music.loss import nll_loss, focal_loss
from sejong_music.trainer import OrchestraTrainer

def make_experiment_name_with_date(config):
  current_time_in_str = datetime.datetime.now().strftime("%m%d-%H%M")
  return f'{current_time_in_str}_{config.general.exp_name}_{config.dataset_class}_{config.model_class}'

@hydra.main(config_path='/home/danbi/userdata/DANBI/gugakwon/SejongMusic/yamls', config_name='orchestration')
def main(config: DictConfig):
  config = get_emb_total_size(config)
  
  if 'duration' not in config.model.features:
    config.model.features.append('duration')
  if 'sampling_rate' in config.data and 'samlping_rate' not in config.model:
    config.model.sampling_rate = config.data.sampling_rate

  if config.general.make_log:
    wandb.init(
      project="yeominrak", 
      entity="maler", 
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
                                xml_path=original_wd/'music_score/FullScore_edit5.musicxml',
                                use_pitch_modification = config.data.use_pitch_modification, 
                                pitch_modification_ratio=config.data.modification_ratio,
                                min_meas=config.data.min_meas, 
                                max_meas=config.data.max_meas,
                                feature_types=copy.copy(config.model.features),
                                sampling_rate=config.data.sampling_rate,
                                target_instrument=config.data.target_instrument,
                                is_sep = config.data.is_sep
                                )
  val_dataset = dataset_class(is_valid= True,
                              # valid_measure_num = [i for i in range(93, 99)],
                              xml_path=original_wd/'music_score/FullScore_edit5.musicxml', 
                              use_pitch_modification=False, 
                              slice_measure_num=config.data.max_meas,
                              min_meas=config.data.min_meas,
                              feature_types=copy.copy(config.model.features),
                              sampling_rate=config.data.sampling_rate,
                              target_instrument=config.data.target_instrument)
    

  train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size , shuffle=True, collate_fn=pack_collate)
  valid_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=pack_collate, drop_last=True)

  device = 'cuda'
  total_iteration = 0
  if isinstance(train_dataset, yeominrak_processing.OrchestraScoreSeq):
    encoder_tokenizer = train_dataset.tokenizer
  else:
    encoder_tokenizer = train_dataset.era_dataset.tokenizer
  model = model_class(encoder_tokenizer, config.model).to(device)
  if 'offset_fraction' in model.tokenizer.tok2idx:
    model.tokenizer.tok2idx.pop('offset_fraction')
  if 'offset' in model.tokenizer.tok2idx:
    for i in range(120):
      if i % 4 == 0:
        continue
      if i / 4 not in model.tokenizer.tok2idx['offset']:
        model.tokenizer.tok2idx['offset'][i/4] = model.tokenizer.tok2idx['offset'][(i-2)/4]

  model.is_condition_shifted = isinstance(train_dataset, yeominrak_processing.ShiftedAlignedScore)
  model.load_state_dict(torch.load('/home/danbi/userdata/DANBI/gugakwon/SejongMusic/outputs/2024-03-22/01-29-11/wandb/run-20240322_012912-3a7w91xw/files/checkpoints/best_model.pt'))
  selected_idx_list = [i for i in range(len(valid_loader.dataset))]
  
  for idx in selected_idx_list:
    sample, _, target = valid_loader.dataset[idx]
    # if isinstance(self.train_loader.dataset, SamplingScore):
    #   target = self.model.converter(target)
    # else:
    target = model.converter(target[:-1])
    input_part_idx = sample[0][0]
    target_part_idx = target[0][0]
    # if input_part_idx < 2: continue
    if model.is_condition_shifted:
      src, output, attn_map = model.shifted_inference(sample.to(device), target_part_idx) 
      
  # optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
  # # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.99)
  # scheduler = None
  # save_dir.mkdir(parents=True, exist_ok=True)
  # loss_fn = nll_loss
  
  # with open(save_dir / 'config.yaml', 'w') as f:
  #   OmegaConf.save(config, f)
  # tokenizer_vocab_path = save_dir / 'tokenizer_vocab.json'
  # train_dataset.tokenizer.save_to_json(tokenizer_vocab_path)
  # if isinstance(train_dataset, yeominrak_processing.OrchestraScoreSeq):
  #   train_dataset.tokenizer.save_to_json(save_dir/'era_tokenizer_vocab.json')
  # else:
  #   train_dataset.era_dataset.tokenizer.save_to_json(save_dir/'era_tokenizer_vocab.json')

  # atrainer = OrchestraTrainer(model=model, optimizer=optimizer, 
  #                   loss_fn=loss_fn, train_loader=train_loader, 
  #                   valid_loader=valid_loader, device = device, save_log=config.general.make_log, 
  #                   save_dir=save_dir, 
  #                   scheduler=scheduler)

  # atrainer.iteration = total_iteration
  # atrainer.train_by_num_epoch(config.train.num_epoch)




if __name__ == '__main__':
  # config = OmegaConf.load('./yamls/baseline.yaml')
  main()
  