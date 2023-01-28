#Just a little script to download the dataset
# {DATASET_NAME} one of the datasets following the naming used in the paper:
#
# WaterDrop
# Water
# Sand
# Goop
# MultiMaterial
# RandomFloor
# WaterRamps
# SandRamps
# FluidShake
# FluidShakeBox
# Continuous
# WaterDrop-XL
# Water-3D
# Sand-3D
# Goop-3D

import os
import wget
from config import selected_config as conf
conf.set_derivate_parameters()

def get_dataset(DATASET_NAME="WaterDrop"):
    BASE_URL=f"https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{DATASET_NAME}/"

    dataset_dir = conf.ROOT_DATASET / DATASET_NAME
    os.makedirs(str(dataset_dir), exist_ok=True)

    for file in ["metadata.json", "train.tfrecord", "valid.tfrecord", "test.tfrecord"]:
        wget.download(f"{BASE_URL}{file}", "dataset/" + DATASET_NAME)