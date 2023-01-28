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
    INTERVAL_SAVE_MODEL=2000
    INTERVAL_UPDATE_LR=2000
    INTERVAL_TEST=2000
    N_TEST = 100
    DEVICE=torch.device('cpu')
    MAX_NEIGH = 6 # Max number of neighbours for each node
    DATASET_NAME = 'WaterDrop'

    def set_derivate_parameters(config):
        """Set parameters which are derivate from other parameters"""
        config.ROOT_DATASET = str(config.root / 'dataset')
        config.ROOT_RUNS = str(config.root)
        config.TRAIN_DATASET = config.ROOT_DATASET + f"/{config.DATASET_NAME}/train.tfrecord"
        config.TEST_DATASET = config.ROOT_DATASET + f"/{config.DATASET_NAME}/test.tfrecord"
        config.VALIDATION_DATASET = config.ROOT_DATASET + f"/{config.DATASET_NAME}/validation.tfrecord"
        config.METADATA = config.ROOT_DATASET + f"/{config.DATASET_NAME}/metadata.json"

    def get(self, key, default_return_value=None):
        """Safe metod to get an attribute. If the attribute does not exist it returns
        None or a specified default value"""
        if hasattr(self, key):
            return self.__getattribute__(key)
        else:
            return default_return_value



selected_config = Config()