import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import torch
from processor import Processor
from decoder import Decoder
from encoder import Encoder
from euler_integrator import integrator
from base_trainer import BaseTrainer
from loader import prepare_data_from_tfds
from noise import add_noise
import numpy as np


class Trainer(BaseTrainer):

    def __init__(self, hyperparams: dict, device=None, seed=None):
        super().__init__(hyperparams, device=device, seed=seed)

        self.init_models()
        self.init_dataloader()
        self.init_logger()

        self.loss_list = []
        self.idx = 0

    def init_dataloader(self):
        self.ds = prepare_data_from_tfds()

    def init_models(self):

        # INITIALISING MODELS
        self.encoder = Encoder(device=self.device)
        self.proc = Processor(128, 128, 128, 128, M=10, device=self.device)
        self.decoder = Decoder().to(self.device)

        self.models = [self.encoder, self.proc, self.decoder]

        # OPTIMIZER
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.opt_proc = optim.Adam(self.proc.all_parameters(), lr=self.lr)
        self.opt_decoder = optim.Adam(self.decoder.parameters(), lr=self.lr)

        self.optimizers = [self.opt_encoder, self.opt_proc, self.opt_decoder]



    def init_logger(self):
        # TENSORBOARD AND LOGGING
        super().init_logger()
        self.mean_ratio_board = torch.zeros([1], device=self.device)
        # variable to store the mean number of invalid moves (this value need to reduce)
        self.mean_error_game = torch.zeros([1], device=self.device)



    def train(self):
        for epoch in range(self.n_epochs):
            print("Epoch ", epoch)
            self.train_epoch()
            self.save_models(self.idx)


    def train_epoch(self):
        for features, labels in self.ds:
            self.idx += 1
            # ███████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗    ██╗███╗   ██╗███████╗ ██████╗
            # ██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝    ██║████╗  ██║██╔════╝██╔═══██╗
            # █████╗   ╚███╔╝    ██║   ██████╔╝███████║██║        ██║       ██║██╔██╗ ██║█████╗  ██║   ██║
            # ██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══██║██║        ██║       ██║██║╚██╗██║██╔══╝  ██║   ██║
            # ███████╗██╔╝ ██╗   ██║   ██║  ██║██║  ██║╚██████╗   ██║       ██║██║ ╚████║██║     ╚██████╔╝
            # ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝       ╚═╝╚═╝  ╚═══╝╚═╝      ╚═════╝
            features['position'] = torch.tensor(features['position']).to(self.device)
            features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(self.device)
            features['particle_type'] = torch.tensor(features['particle_type']).to(self.device)
            labels = torch.tensor(labels).to(self.device)
            position = features["position"]
            # add noise
            # TODO: implement noise for position
            position = add_noise(position)

            """
            n is the nuber of particles
            `partycle_type`: Integer values tensor of size n. Each value represent the material of the ith particle
            `position`: Float values tensor of size n x 6 x 2. It represents the last six positions (x, y) of the particles
            `n_particles_per_example`: Integer values Tensor of size 2 = [n1, n2] with n1 + n2 = n ????????? 

            `labels`: Float values tensor of size n x 2. It represents future positions to predict        
            """

            # █████╗ ██████╗ ██████╗ ██╗  ██╗   ██╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗
            # ██╔══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║
            # ███████║██████╔╝██████╔╝██║   ╚████╔╝     ██╔████╔██║██║   ██║██║  ██║█████╗  ██║
            # ██╔══██║██╔═══╝ ██╔═══╝ ██║    ╚██╔╝      ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║
            # ██║  ██║██║     ██║     ███████╗██║       ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
            # ╚═╝  ╚═╝╚═╝     ╚═╝     ╚══════╝╚═╝       ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝

            # Create graph with features
            data = self.encoder(position)
            # Process graph
            data = self.proc(data)
            # print("Processed Data: ", data)
            # extract acceleration using decoder + euler integrator
            acc = self.decoder(data)
            labels_est = integrator(position, acc)
            # print("Acceleration:", acc)

            # ██╗   ██╗██████╗ ██████╗  █████╗ ████████╗███████╗
            # ██║   ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
            # ██║   ██║██████╔╝██║  ██║███████║   ██║   █████╗
            # ██║   ██║██╔═══╝ ██║  ██║██╔══██║   ██║   ██╔══╝
            # ╚██████╔╝██║     ██████╔╝██║  ██║   ██║   ███████╗
            # ╚═════╝ ╚═╝     ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝

            # reset gradients
            for opt in self.optimizers:
                opt.zero_grad()
            # calculate loss
            loss = torch.abs(labels_est - labels).sum()
            # backpropagation
            loss.backward()
            # update parameters
            for opt in self.optimizers:
                opt.step()
            loss_logged = loss.item()
            self.loss_list.append(loss_logged)
            self.mean_loss += loss_logged
            print("Loss is :", loss_logged)
            if self.idx % self.interval_tensorboard == 0:
                self.report(self.idx)


    def save_models(self, i):
        for model in self.models:
            self.save_model(model, model.__class__.__name__, i)


    def report(self, i):
        self.loss_list = []
        super().report(i)
        self.mean_loss = 0





if __name__ == "__main__":

    hyperparams = {
        "lr": 0.01,
        "n_epochs": 20,
        "interval_tensorboard": 3
    }
    device = torch.device("cpu")

    trainer = Trainer(hyperparams=hyperparams, seed=99, device=device)
    trainer.train()


#type "tensorboard --logdir=runs" in terminal
