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

While the simulation is far from perfect, it still showcases the effectiveness of the model. It's worth noting that the model calculates 200 frames starting only from the initial positions, which can result in a significant accumulation of errors over time.


## Repository Structure
 
Here are the most important files related to the project:
```
GNN
├── README.md
├── env.yml (Conda environment)
└── physics_simulation
    ├── animation (contains the gif of the simulation from test.py)
    ├── dataset (contains the dataset)
    │   └── waterdrop
    │       ├── metadata.json
    │       ├── train.tfrecord
    │       └── test.tfrecord
    ├── lib (contains code that hlped in the project, including reading_utils used in the dataloader)
    ├── pretrained (contains trained models)
    ├── runs (contains TensorBoard logs and saved models)
    ├── __main__.py (used to run the training)
    ├── base_trainer.py (implements base trainer functionality)
    ├── config.py (contains hyperparameters and paths)
    ├── encoder.py (Encoder model)
    ├── processor.py (Processor model)
    ├── decoder.py (Decoder model)
    ├── euler_integrator.py (calculates positions from acceleration)
    ├── get_dataset.py (downloads the dataset)
    ├── loader.py (loads data from .tfrecord)
    ├── make_animation.py (generates a simulation from a saved model)
    ├── colab_train.ipynp (can be loaded on Colab to train the model on the cloud)
    └── train.py (training script)

```

- `animation`, is used to generate simulation gif during test of the model.

- `lib` Code from https://github.com/wu375/simple-physics-simulator-pytorch-geometry, we used it
 for loading tfrecord dataset;
- Pretrained models can be found in the `pretrained` directory.
- TensorBoard logs and saved models are stored in the `runs` directory, with a subfolder for each run;
- `base_trainer.py` is a module used in other projects and implements basic logging and model-saving functionality;
- `encoder.py`, `processor.py`, `decoder.py` make up the three components of the model.;
- `colab_train.ipynp` Load the file on Colab to train the model on the cloud platform


## Remarks

### About `wu375` github repository

The data loader in this repository was obtained from the wu375 repository, while the rest of the code was developed independently. Our model outperformed the wu375 repository in terms of test loss during training, likely due to errors in the residual layer implementation and message passing forward in the wu375 code.


### Model

- **Encoder**: The encoder uses particle position data to generate a graph, where each particle is represented as a node and close particles are connected by arcs. Features for each node and arc are generated using past particle positions. There are two strategies for generating features:
  - *absolute features*:  Using absolute particle positions. This approach is straightforward and makes it easier to identify bounds, but the model may struggle to generate results for other positions.
  - *relative features*: Using velocities for particle features and relative positions as arc features.
  This repository uses a mix of the two approaches, which is likely similar to using absolute features. It would be interesting to test the model using only relative features, but the presence of bounds makes this complicated as providing the distance from the bound is equivalent to providing the absolute particle position.


- **Processor**: The processor is a sequence of blocks, each of which mainly consists of a convolutional GNN that operates on both node and edge features to generate new node and edge features. This type of convolutional GNN was not available in the PyTorch library, so it was implemented using the MessagePassing class in the PyTorch Geometric library. It is noted that in the paper's appendix (C.1), the authors explain that they detected no significant difference in disabling edge state updates, which seems reasonable given the potential for strong redundancy with node state updates.


- **Decoder** : The decoder is a simple MLP that distills two values for each node (particle acceleration).


### Noise

A random walk noise in particle positions was implemented, as suggested by the authors in the paper. 

The effect of training with and without noise was tested. TODO.

### Normalisation

Input velocities were normalized using the mean and standard deviation of the dataset, provided by the metadata file. The output acceleration was multiplied by the standard deviation to improve control over the learning rate.

### Learning Rate

The initial learning rate was set to 1e-4 and was decreased by a factor of 10 every 500k steps.

## Conclusion
    
### Training time

Training the model on a laptop was challenging due to the heavy computational demands and insufficient GPU memory. As a result, training was conducted on Google Colab (the relevant notebook is provided in the repository)

### Performance

TODO