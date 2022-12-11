from config import selected_config as conf
import torch


conf.N_BATCH= 2
conf.LOAD_PATH= None# RUNS/FIT/20221120-103911/MODELS,
conf.LOAD_IDX= 0
conf.DEVICE = torch.device("cuda")
conf.set_derivate_parameters()
conf.ROOT_DATASET='dataset'
conf.ROOT_RUNS='./'


from train import Trainer

trainer = Trainer()
trainer.train()