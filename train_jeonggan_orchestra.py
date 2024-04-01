import datetime
import random
import wandb
import hydra
from pathlib import Path
import copy

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from sejong_music import yeominrak_processing, model_zoo, loss, utils
from sejong_music.yeominrak_processing import pack_collate
from sejong_music.model_zoo import get_emb_total_size
from sejong_music.loss import nll_loss, focal_loss
from sejong_music.trainer import JeongganTrainer
from sejong_music.train_utils import CosineLRScheduler
# from sejong_music.constants import PART, POSITION, PITCH
from sejong_music.jg_code import JeongganDataset

def make_experiment_name_with_date(config):
  current_time_in_str = datetime.datetime.now().strftime("%m%d-%H%M")
  return f'{current_time_in_str}_{config.general.exp_name}_{config.dataset_class}_{config.model_class}'

@hydra.main(config_path='yamls/', config_name='transformer_jeonggan')
def main(config: DictConfig):
  config = get_emb_total_size(config)
  
  if config.general.make_log:
    wandb.init(
      project="yeominrak", 
      entity="danbinaerin", 
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
  # dataset_class = JeongganDataset
  model_class = getattr(model_zoo, config.model_class)
  
  train_dataset = JeongganDataset(data_path= Path('/home/danbi/userdata/DANBI/gugakwon/SejongMusic/music_score/gen_code'), 
                  # slice_measure_num = 2,
                  is_valid=False,
                  # use_pitch_modification=False,
                  # pitch_modification_ratio=0.3,
                  # min_meas=3,
                  # max_meas=6,
                  # feature_types=['index', 'token', 'position'],
                  # target_instrument='daegeum'
                  )
  
  val_dataset = JeongganDataset(data_path= Path('/home/danbi/userdata/DANBI/gugakwon/SejongMusic/music_score/gen_code'), 
                  # slice_measure_num = 2,
                  is_valid=True,
                  # use_pitch_modification=False,
                  # pitch_modification_ratio=0.3,
                  # min_meas=3,
                  # max_meas=6,
                  # feature_types=['index', 'token', 'position'],
                  # part_list = PART, position_token = POSITION, pitch_token = PITCH, 
                  # target_instrument=0
                  )
    
  collate_fn = getattr(utils, config.collate_fn)
  loss_fn = getattr(loss, config.loss_fn)

  train_loader = DataLoader(train_dataset, 
                            batch_size=config.train.batch_size , 
                            shuffle=True, 
                            collate_fn=collate_fn,
                            num_workers=4)
  valid_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=collate_fn, drop_last=True)

  device = 'cuda'
  total_iteration = 0

  
  # --- Save config and tokenizer --- #
  with open(save_dir / 'config.yaml', 'w') as f:
    OmegaConf.save(config, f)
  tokenizer_vocab_path = save_dir / 'tokenizer_vocab.json'
  train_dataset.tokenizer.save_to_json(tokenizer_vocab_path)

  
  repeat = 4
  for target_inst_idx in range(repeat):
    if not isinstance(train_dataset, yeominrak_processing.OrchestraScoreSeqTotal) and isinstance(train_dataset, yeominrak_processing.OrchestraScoreSeq):
      train_dataset.target_instrument = target_inst_idx
      val_dataset.target_instrument = target_inst_idx
    
    model = model_class(train_dataset.tokenizer, config.model).to(device)
    num_model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of model parameters: {num_model_parameters}')
    if config.general.make_log: 
      wandb.run.summary['num_model_parameters'] = num_model_parameters
    
    model.is_condition_shifted = isinstance(train_dataset, yeominrak_processing.ShiftedAlignedScore)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    scheduler = CosineLRScheduler(optimizer, total_steps=config.train.num_epoch * len(train_loader), warmup_steps=500, lr_min_ratio=0.0001, cycle_length=1.0)
    # scheduler = None
    

    # --- Training --- #
    inst_save_dir = save_dir / f'inst_{target_inst_idx}'
    atrainer = JeongganTrainer(model=model, 
                                optimizer=optimizer, 
                                loss_fn=loss_fn, 
                                train_loader=train_loader, 
                                valid_loader=valid_loader, 
                                device = device, 
                                save_log=config.general.make_log, 
                                save_dir=inst_save_dir, 
                                scheduler=scheduler)

    atrainer.iteration = total_iteration
    atrainer.train_by_num_iteration(config.train.num_epoch * len(train_loader))
    atrainer.load_best_model()

    atrainer.make_inference_result(write_png=True)
    total_iteration = atrainer.iteration


if __name__ == '__main__':
  main()