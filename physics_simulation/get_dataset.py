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


import wget

DATASET_NAME = "WaterDrop"
BASE_URL=f"https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{DATASET_NAME}/"



for file in ["metadata.json", "train.tfrecord", "valid.tfrecord", "test.tfrecord"]:
    wget.download(f"{BASE_URL}{file}", "dataset/water_drop")