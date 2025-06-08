import argparse
from pathlib import Path
from omegaconf import OmegaConf

import torch

from sejong_music import model_zoo, jg_code
from sejong_music.jg_code import JeongganTokenizer, ABCTokenizer
from sejong_music.inference import JGInferencer, ABCInferencer
from sejong_music.evaluation import get_test_result


def parse_args():
  parser = argparse.ArgumentParser(description='Get test results from trained model')
  parser.add_argument('--model_path', type=str, required=True,
                    help='Path to the model checkpoint directory containing best_model.pt and config.yaml')
  
  return parser.parse_args()


def main():
  args = parse_args()
  model_path = Path(args.model_path)
  # ckpt = torch.load(model_path / 'iter34800_model.pt', map_location='cpu')
  # ckpt = torch.load(model_path / 'last_model.pt', map_location='cpu')
  config = OmegaConf.load(model_path / 'config.yaml')
  
  is_abc = config.dataset_class == 'ABCDataset'

  tokenizer_path = model_path / 'tokenizer_vocab.json'
  if not tokenizer_path.exists():
    raise FileNotFoundError(f"Tokenizer vocab file not found at {tokenizer_path}")
  
  if is_abc:
    tokenizer = ABCTokenizer(None, json_fn=str(tokenizer_path))
  else:
    tokenizer = JeongganTokenizer(None, json_fn=str(tokenizer_path))

  dataset_class = getattr(jg_code, config.dataset_class)
  model_class = getattr(model_zoo, config.model_class)

  model = model_class(tokenizer, config.model)

  from torch.utils.data import DataLoader

  # Replicating dataset creation from train.py to ensure consistency
  feature_types = ['token', 'in_jg_position', 'jg_offset', 'gak_offset', 'jangdan', 'genre', 'inst']
  if not hasattr(config.data, 'use_jangdan') or not config.data.use_jangdan:
    feature_types.pop(feature_types.index('jangdan'))
  if not hasattr(config.data, 'use_genre') or not config.data.use_genre:
    feature_types.pop(feature_types.index('genre'))
  if not hasattr(config.data, 'use_offset'):
    use_offset = False
  else:
    use_offset = config.data.use_offset
  if config.dataset_class == "JGMaskedDataset":
    feature_types.pop(feature_types.index('token'))
    feature_types = ['pitch', 'sigimsae'] + feature_types
    
  slice_measure_num = config.data.slice_measure_num if hasattr(config.data, 'slice_measure_num') else 4

  test_dataset = dataset_class(data_path='music_score/jg_cleaned',
                               split='test',
                               slice_measure_num=slice_measure_num,
                               augment_param=config.aug,
                               num_max_inst=config.data.num_max_inst,
                               feature_types=feature_types,
                               use_offset=use_offset
                               )
  test_dataset.tokenizer = tokenizer  # ensure dataset uses the loaded tokenizer


  if is_abc:
    inferencer = ABCInferencer(model, use_offset, True, 1.0, 0.9)
  else:
    inferencer = JGInferencer(model, use_offset, True, 1.0, 0.9)


  ckpt = torch.load(model_path / 'best_model.pt', map_location='cpu')
  model.load_state_dict(ckpt)
  model.eval()
  model.to('cuda')
  
  print("Loaded Best Val Acc Model")
  output = get_test_result(model, inferencer, test_dataset, 'daegeum', None)
  print("Result for target instrument: daegeum / Condition instruments: All")
  print(output)
  
  output = get_test_result(model, inferencer, test_dataset, 'geomungo', ['piri'])
  print("Result for target instrument: geomungo / Condition instruments: [piri]")
  print(output)
  
  ckpt = torch.load(model_path / 'best_note_acc_model.pt', map_location='cpu')
  model.load_state_dict(ckpt)
  model.eval()
  model.to('cuda')
  
  print("="*40)
  print("Loaded Best Note Acc Model")
  output = get_test_result(model, inferencer, test_dataset, 'daegeum', None)
  print("Result for target instrument: daegeum / Condition instruments: All")
  print(output)
  
  output = get_test_result(model, inferencer, test_dataset, 'geomungo', ['piri'])
  print("Result for target instrument: geomungo / Condition instruments: [piri]")
  print(output)
  
  print("="*40)
  ckpt = torch.load(model_path / 'last_model.pt', map_location='cpu')
  model.load_state_dict(ckpt)
  model.eval()
  model.to('cuda')
  
  print("Loaded Last Model")
  output = get_test_result(model, inferencer, test_dataset, 'daegeum', None)
  print("Result for target instrument: daegeum / Condition instruments: All")
  print(output)
  
  output = get_test_result(model, inferencer, test_dataset, 'geomungo', ['piri'])
  print("Result for target instrument: geomungo / Condition instruments: [piri]")
  print(output)
  

if __name__ == '__main__':
  main()