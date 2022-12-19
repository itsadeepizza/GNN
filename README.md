# Simulate physics using GNN

## Introduction
A [2020 paper](https://www.deepmind.com/publications/learning-to-simulate-complex-physics-with-graph-networks)
from deepmind showed how to simulate complex interaction between physical
particles using Graph Neural Network.

We have decided to try to recreate the model used for this task "from scratch", based solely on the paper, and 
occasionally using an [unofficial pytorch implementation](https://github.com/wu375/simple-physics-simulator-pytorch-geometry) 
to help us.
As we are limited in hardware (4GB Quadro GPU on a laptop) and our aim was not to produce a model
so accurate as the one of Deepmind, we trained only on a small portion of dataset, the "Water Drop".

This is the result achieved after 616k steps:

![GIF simulation](animation/simulation_616k_20221216-221417.gif)

The simulation is far from perfect, but probably it could be improved training
the model on the whole dataset. However, it shows the effectiveness of the model, 
especially if we consider the fact that starting only from the initial positions, a 
sequence of 200 frames is calculated, so errors cumulates very quickly.


## Repository Structure
 
We report most important files related to the project:
```
GNN
├── README.md
├── env.yml -> Conda environment 
└── physics_simulation
    ├── animation   -> Contains gif of simulation from test.py 
    ├── dataset     -> Contains dataset
    │   └── waterdrop
    │       ├── metadata.json
    │       ├── train.tfrecord
    │       └── test.tfrecord
    ├── frame       -> Contains single frame of simulation from test.py
    ├── lib         -> Some inspiring code
    ├── pretrained  -> Trained models 
    ├── runs        -> Tensorboard logs and saved models
    ├── __main__.py -> Use this to run train
    ├── base_trainer.py -> Implement base trainer functionalities 
    ├── config.py       -> Hyperparameters and paths
    ├── encoder.py      -> Encoder model
    ├── processor.py    -> Processor model
    ├── decoder.py      -> Decoder model
    ├── euler_integrator.py -> Calculate position from acceleration
    ├── get_dataset.py  -> Download dataset
    ├── loader.py       -> Load data from .tfrecord 
    ├── test.py         -> Test trained model
    ├── colab_train.ipynp   -> Train the model on Google Colab
    └── train.py        -> Train model
```

- `animation`, `frame` Are used to generate simulation gif during test of the model.
    You will need to have `ffmpeg` installed for generating the gif file;
- `lib` Code from https://github.com/wu375/simple-physics-simulator-pytorch-geometry, we used it
 for loading tfrecord dataset;
- `pretrained` you can directly load pretrained model from here;
- `runs` Folder generated during train. A subfolder for each run is generated, containing
 tensorboard logs and saved models;
- `base_trainer.py` a module we used also in other projects, it implements basic fonctionnality
 as logging and saving models;
- `encoder.py`, `processor.py`, `decoder.py` constitues the three steps of the model;
- `colab_train.ipynp` Load the file on Colab to train the model on the cloud platform from Google


## Remarks

### About `wu375` github repository

Dataloader is taken from `wu375`, the remaining part of the code is developed independently.
We remark that test loss of our model outperform the `wu375` repository during training.
The reason is plausibly in some errors in `wu375` code about residual layer implementation and MessagePassing forward.


### Processor

### Noise

### Normalisation

### Learning Rate

## Conclusion
    
### Training time

### Performance

