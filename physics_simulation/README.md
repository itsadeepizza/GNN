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


## Files Structure

## Remarks