from config import Config
import torch

Config(
    ROOT_DATASET='dataset',
    ROOT_RUNS='./',

    N_BATCH= 2,
    LR= 1E-4,
    N_EPOCHS= 20,
    INTERVAL_TENSORBOARD= 100,
    N_FEATURES= 128,  # 128
    M= 10,  # 10
    R= 0.015,  # 0.015
    STD_NOISE= 1E-5,
    LOAD_PATH= None,# RUNS/FIT/20221120-103911/MODELS,
    LOAD_IDX= 0,
    DEVICE = torch.device("cuda")

).set_config()

from train import Trainer

trainer = Trainer()
trainer.train()