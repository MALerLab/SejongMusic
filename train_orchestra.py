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
from sejong_music.trainer import OrchestraTrainer
from sejong_music.train_utils import CosineLRScheduler

def make_experiment_name_with_date(config):
  current_time_in_str = datetime.datetime.now().strftime("%m%d-%H%M")
  return f'{current_time_in_str}_{config.general.exp_name}_{config.dataset_class}_{config.model_class}'

@hydra.main(config_path='yamls/', config_name='orchestration')
def main(config: DictConfig):
  config = get_emb_total_size(config)
  
  assert 'duration' in config.model.features, 'Duration should be included in the feature list.'

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
  save_dir.mkdir(parents=True, exist_ok=True)
  dataset_class = getattr(yeominrak_processing, config.dataset_class)
  valid_dataset_class = yeominrak_processing.OrchestraScoreSeq if config.dataset_class=='OrchestraScoreSeqTotal' else dataset_class
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
  val_dataset = valid_dataset_class(is_valid= True,
                              # valid_measure_num = [i for i in range(93, 99)],
                              xml_path=original_wd/'music_score/FullScore_edit5.musicxml', 
                              use_pitch_modification=False, 
                              slice_measure_num=config.data.max_meas,
                              min_meas=config.data.min_meas,
                              feature_types=copy.copy(config.model.features),
                              sampling_rate=config.data.sampling_rate,
                              target_instrument=config.data.target_instrument,
                              is_sep=config.data.is_sep)
    
  collate_fn = getattr(utils, config.collate_fn)
  loss_fn = getattr(loss, config.loss_fn)

  train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size , shuffle=True, collate_fn=collate_fn)
  valid_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=collate_fn, drop_last=True)

  device = 'cuda'
  total_iteration = 0
  if isinstance(train_dataset, yeominrak_processing.OrchestraScoreSeq):
    encoder_tokenizer = train_dataset.tokenizer
  else:
    encoder_tokenizer = train_dataset.era_dataset.tokenizer
    
  if 'offset_fraction' in train_dataset.tokenizer.tok2idx: # Orchestra does not uses offset_fraction
    train_dataset.tokenizer.tok2idx.pop('offset_fraction')
  if 'offset' in train_dataset.tokenizer.tok2idx: # Fill in the missing offset
    for i in range(120):
      if i % 4 == 0:
        continue
      if i / 4 not in train_dataset.tokenizer.tok2idx['offset']:
        train_dataset.tokenizer.tok2idx['offset'][i/4] = train_dataset.tokenizer.tok2idx['offset'][(i-2)/4]
  
  # --- Save config and tokenizer --- #
  with open(save_dir / 'config.yaml', 'w') as f:
    OmegaConf.save(config, f)
  tokenizer_vocab_path = save_dir / 'tokenizer_vocab.json'
  train_dataset.tokenizer.save_to_json(tokenizer_vocab_path)
  if isinstance(train_dataset, yeominrak_processing.OrchestraScoreSeq):
    train_dataset.tokenizer.save_to_json(save_dir/'era_tokenizer_vocab.json')
  else:
    train_dataset.era_dataset.tokenizer.save_to_json(save_dir/'era_tokenizer_vocab.json')

  if isinstance(train_dataset, yeominrak_processing.OrchestraScoreSeqTotal):
    repeat = 1
    val_dataset.target_instrument = 1
  elif isinstance(train_dataset, yeominrak_processing.OrchestraScoreSeq):
    repeat = 4
  else:
    raise NotImplementedError
  
  
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
    atrainer = OrchestraTrainer(model=model, 
                                optimizer=optimizer, 
                                loss_fn=loss_fn, 
                                train_loader=train_loader, 
                                valid_loader=valid_loader, 
                                device = device, 
                                save_log=config.general.make_log, 
                                save_dir=inst_save_dir, 
                                scheduler=scheduler)

    atrainer.iteration = total_iteration
    atrainer.train_by_num_epoch(config.train.num_epoch)
    atrainer.load_best_model()

    atrainer.make_inference_result(write_png=True)
    total_iteration = atrainer.iteration


if __name__ == '__main__':
  main()