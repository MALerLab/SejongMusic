CUDA_VISIBLE_DEVICES=1 python3 train_jeonggan_orchestra.py model.dim=64 model.depth=6 model.dropout=0.2 general.entity=dasaem
CUDA_VISIBLE_DEVICES=1 python3 train_jeonggan_orchestra.py model.dim=128 model.depth=6 model.dropout=0.2 general.entity=dasaem
CUDA_VISIBLE_DEVICES=1 python3 train_jeonggan_orchestra.py model.dim=128 model.depth=8 model.dropout=0.2 general.entity=dasaem
CUDA_VISIBLE_DEVICES=1 python3 train_jeonggan_orchestra.py model.dim=64 model.depth=10 model.dropout=0.2 general.entity=dasaem
