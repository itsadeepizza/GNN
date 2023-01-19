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


### Model

- **Encoder**: Data about positions are used to generate a graph, each particle 
is a node, close particles are linked by an arc. Past positions of particles are 
used to obtain features for each node and arc.
There are two strategies in making features:
  - *absolute features*: Using absolute positions of the particles. It is more straightforward
  and also easier for identify bounds, but the model will have some difficults in generating results
  for other positions
  - *relative features*: Using velocities for particle features and relative positions as arc features.
  We used a mix of the two approaches, which is probably almost the same that using absolute features.
  It would be interesting to test the model with only relative features, but the presence of bounds make it pretty
  complicated, as giving the distance from the bound is the same that giving the absoluto position of the particle.


- **Processor**: It is composed by a sequence of block. Each block is composed mainly by
a convolutional GNN, but, convolution is performed on both node and edge features in
order to obtain new node and edge features. Such a convolutional GNN was not present in
pytorch library, so we implemented it using MessagePassing, which is a base class in pytorch_geometric
for all GNN. However, we remark that in the paper (Appendix C.1) authors explains they detected
no significant variation in disabling edges state update, which seems quite reasonable, as
at the end there is probably a strong redondance with node state update.


- **Decoder** : Just a MLP to distillate two values for each node (acceleration of the particle)


### Noise

In the paper authors suggested to use random walk noise in position particles.
We implemented a simpler gaussian noise 

### Normalisation

### Learning Rate

## Conclusion
    
### Training time

### Performance

