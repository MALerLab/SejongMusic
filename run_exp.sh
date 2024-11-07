CUDA_VISIBLE_DEVICES=1 python3 --config-name=bert train_jeonggan_orchestra.py model.dim=64 model.depth=4 data.num_max_inst=4 general.entity=dasaem train.num_epoch=300 
CUDA_VISIBLE_DEVICES=1 python3 --config-name=bert train_jeonggan_orchestra.py model.dim=64 model.depth=6 data.num_max_inst=4 general.entity=dasaem train.num_epoch=300 
CUDA_VISIBLE_DEVICES=1 python3 --config-name=bert train_jeonggan_orchestra.py model.dim=64 model.depth=10 data.num_max_inst=3 general.entity=dasaem train.num_epoch=300 
