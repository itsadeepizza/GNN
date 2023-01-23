from config import selected_config as conf
import torch


conf.N_BATCH = 2
conf.LOAD_PATH = None
conf.LOAD_IDX = 0
conf.DEVICE = torch.device("cuda")
conf.set_derivate_parameters()
conf.ROOT_DATASET = 'dataset'
conf.ROOT_RUNS = './'
conf.INTERVAL_SAVE_MODEL = 20000
conf.LR_INIT = 1E-4
conf.LR_DECAY = 1E-1
conf.LR_STEP = 1e6
conf.STD_NOISE = 0

from train import Trainer

trainer = Trainer()
trainer.train()