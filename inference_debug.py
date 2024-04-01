from omegaconf import OmegaConf
import torch

from sejong_music.jg_code import JeongganDataset, JeongganTokenizer
from sejong_music.model_zoo import JeongganTransSeq2seq
from sejong_music.inference import JGInferencer

config = OmegaConf.load('outputs/2024-03-30/00-53-31/wandb/run-20240330_005332-1un8d0dv/files/checkpoints/config.yaml')
dataset = JeongganDataset(is_valid=True)
tokenizer = JeongganTokenizer(None, None, json_fn='outputs/2024-03-30/00-53-31/wandb/run-20240330_005332-1un8d0dv/files/checkpoints/tokenizer_vocab.json')
dataset.tokenizer = tokenizer
model = JeongganTransSeq2seq(tokenizer, config.model)
inferencer = JGInferencer(model, is_condition_shifted=True, is_orch=True)

state_dict = torch.load('outputs/2024-03-30/01-21-13/wandb/run-20240330_012114-3va5pb74/files/checkpoints/inst_0/iter21275_model.pt')
model.load_state_dict(state_dict)
model.cuda()
model.eval()

src, tgt, shifted_tgt = dataset[0]
source, output, _ = inferencer.inference(src, inst_name='daegeum')


source, output, _ = inferencer.inference(src, inst_name='daegeum')