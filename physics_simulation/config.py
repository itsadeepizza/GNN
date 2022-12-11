from pathlib import Path
import os
import torch

class Config():


    root = Path(__file__).resolve().parents[0]
    N_BATCH=2
    # LR is multiplied by LR_DECAY every LR_STEP
    LR_INIT=1E-4
    LR_DECAY=1E-1
    LR_STEP = 5e5
    N_EPOCHS=20
    INTERVAL_TENSORBOARD=100
    N_FEATURES=128  # 128
    M=10 # 10
    R=0.015 # 0.015
    STD_NOISE=1E-5
    LOAD_PATH=None  # RUNS/FIT/20221120-103911/MODELS,
    LOAD_IDX=0
    SEED=99
    INTERVAL_SAVE_UPDATE=2000
    DEVICE=torch.device('cpu')

    def set_derivate_parameters(config):
        """Set parameters which are derivate from other parameters"""
        config.ROOT_DATASET = config.root / 'dataset'
        config.ROOT_RUNS = config.root



selected_config = Config()