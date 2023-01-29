from config import selected_config as conf
import torch


conf.N_BATCH = 2
conf.LOAD_PATH = "runs/fit/local"
conf.LOAD_IDX = 1980000
conf.DEVICE = torch.device("cuda")
conf.set_derivate_parameters()
conf.ROOT_DATASET = 'dataset'
conf.ROOT_RUNS = './'
conf.INTERVAL_SAVE_MODEL = 20000
conf.LR_INIT = 1E-4
conf.LR_DECAY = 1E-1
conf.LR_STEP = 1e6
conf.STD_NOISE = 0
conf.TEST_DATASET = conf.ROOT_DATASET + f"/{conf.DATASET_NAME}/valid.tfrecord"

conf.N_STEP = 300

from train import Trainer

trainer = Trainer(init_logger=False)
trainer.simulate()