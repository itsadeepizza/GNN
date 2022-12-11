from config import selected_config as conf
import torch


conf.N_BATCH= 2
conf.LOAD_PATH= "runs/fit/20221211-150225/models"
conf.LOAD_IDX= 220000
conf.DEVICE = torch.device("cuda")
conf.set_derivate_parameters()
conf.ROOT_DATASET='dataset'
conf.ROOT_RUNS='./'


from train import Trainer

trainer = Trainer()
trainer.train()