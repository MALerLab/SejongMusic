import datetime
import random
import wandb
import hydra
from pathlib import Path
import copy
from typing import List, Union

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from sejong_music import yeominrak_processing, model_zoo, loss, utils, jg_code, trainer as trainer_zoo
from sejong_music.yeominrak_processing import pack_collate
from sejong_music.model_zoo import get_emb_total_size
from sejong_music.loss import nll_loss, focal_loss
from sejong_music.trainer import JeongganTrainer, BertTrainer
from sejong_music.train_utils import CosineLRScheduler
# from sejong_music.constants import PART, POSITION, PITCH
from sejong_music.jg_code import JeongganDataset, JGMaskedDataset, ABCDataset
from sejong_music.full_inference import Generator

def make_experiment_name_with_date(config):
  current_time_in_str = datetime.datetime.now().strftime("%m%d-%H%M")
  return f'{current_time_in_str}_data={config.dataset_class}_depth={config.model.depth}_head={config.model.num_heads}_drop={config.model.dropout}_is_beat={config.data.use_offset}_pos_count={config.data.is_pos_counter}_{config.general.exp_name}_{config.model_class}'

@hydra.main(config_path='yamls/', config_name='transformer_jeonggan')
def main(config: DictConfig):
  config = get_emb_total_size(config)
  
  if config.general.make_log:
    wandb.init(
      project=config.general.project, 
      entity=config.general.entity, 
      name = make_experiment_name_with_date(config), 
      config = OmegaConf.to_container(config),
      dir=Path(hydra.utils.get_original_cwd())
    )
    save_dir = Path(wandb.run.dir) / 'checkpoints'
  else:
    save_dir = Path('wandb/debug/checkpoints')

  original_wd = Path(hydra.utils.get_original_cwd())
  if not save_dir.is_absolute():
    save_dir = original_wd / save_dir
  save_dir.mkdir(parents=True, exist_ok=True)
  dataset_class = getattr(jg_code, config.dataset_class)
  model_class = getattr(model_zoo, config.model_class)
  dataset_class:Union[JeongganDataset, JGMaskedDataset] = getattr(jg_code, config.dataset_class)
  trainer_class:Union[JeongganTrainer, BertTrainer] = getattr(trainer_zoo, config.trainer_class)
  
  feature_types = ['token', 'in_jg_position', 'jg_offset', 'gak_offset', 'jangdan', 'genre', 'inst']
  if not config.data.use_jangdan:
    feature_types.pop(feature_types.index('jangdan'))
  if not config.data.use_genre:
    feature_types.pop(feature_types.index('genre'))
  
  train_dataset = dataset_class(data_path= original_wd / 'music_score/jg_cleaned',
                  slice_measure_num = config.data.slice_measure_num,
                  split='train',
                  augment_param = config.aug,
                  num_max_inst = config.data.num_max_inst,
                  feature_types = feature_types,
                  use_offset=config.data.use_offset
                  )
  
  val_dataset = dataset_class(data_path= original_wd / 'music_score/jg_cleaned', 
                  split='valid',
                  augment_param = config.aug,
                  num_max_inst = config.data.num_max_inst,
                  feature_types = feature_types,
                  use_offset=config.data.use_offset
                  )
  
  test_dataset = dataset_class(data_path= original_wd / 'music_score/jg_cleaned', 
                  split='test',
                  augment_param = config.aug,
                  num_max_inst = config.data.num_max_inst,
                  feature_types = feature_types,
                  use_offset=config.data.use_offset
                  )
    
  collate_fn = getattr(utils, config.collate_fn)
  loss_fn = getattr(loss, config.loss_fn)

  train_loader = DataLoader(train_dataset, 
                            batch_size=config.train.batch_size , 
                            shuffle=True, 
                            collate_fn=collate_fn,
                            num_workers=4)
  valid_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=collate_fn, drop_last=False)
  test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_fn, drop_last=False)
  device = 'cuda'

  
  # --- Save config and tokenizer --- #
  with open(save_dir / 'config.yaml', 'w') as f:
    OmegaConf.save(config, f)
  tokenizer_vocab_path = save_dir / 'tokenizer_vocab.json'
  train_dataset.tokenizer.save_to_json(tokenizer_vocab_path)
  
  model = model_class(train_dataset.tokenizer, config.model).to(device)
  num_model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f'Number of model parameters: {num_model_parameters}')
  if config.general.make_log: 
    wandb.run.summary['num_model_parameters'] = num_model_parameters
  
  optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
  scheduler = CosineLRScheduler(optimizer, total_steps=config.train.num_epoch * len(train_loader), warmup_steps=1000, lr_min_ratio=0.001, cycle_length=1.0)
  

  # --- Training --- #
  atrainer:JeongganTrainer = trainer_class(model=model, 
                                optimizer=optimizer, 
                                loss_fn=loss_fn, 
                                train_loader=train_loader, 
                                valid_loader=valid_loader, 
                                device = device, 
                                save_log=config.general.make_log, 
                                save_dir=save_dir, 
                                scheduler=scheduler,
                                is_pos_counter='gak:0' in train_dataset.tokenizer.vocab,
                                min_epoch_for_infer=100)
  # generator = Generator(config=None,
  #                       model=model,
  #                       output_dir=save_dir,
  #                       inferencer=atrainer.inferencer, 
  #                       is_abc = dataset_class==ABCDataset,
  #                       )

  atrainer.train_by_num_iteration(config.train.num_epoch * len(train_loader))
  atrainer.load_best_model()
  # ckpt = torch.load(model_path / 'best_model.pt', map_location='cpu')
  # model.load_state_dict(ckpt)
  # model.eval()

  print(atrainer.make_inference_result(loader=test_loader))


if __name__ == '__main__':
  main()