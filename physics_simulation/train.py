import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import torch
from processor import Processor
from decoder import Decoder
from encoder import Encoder
from euler_integrator import integrator, get_acc
from base_trainer import BaseTrainer
from loader import prepare_data_from_tfds
import numpy as np
from benchmark import benchmark_nomove_acc, benchmark_noacc_acc, benchmark_nojerk_acc
import torch
from torch.optim.lr_scheduler import StepLR


def add_noise(position: torch.Tensor):
    std = 0.00015
    noise = torch.randn(position.shape, device=position.device) * std
    return position + noise



class Trainer(BaseTrainer):

    def __init__(self, hyperparams: dict, device=None, seed=None, load_path=None, load_idx=0):
        super().__init__(hyperparams, device=device, seed=seed)

        self.init_dataloader()
        self.init_models()
        self.init_logger()

        self.mean_loss_nomove = 0
        self.mean_loss_noacc = 0
        self.mean_loss_nojerk = 0

        self.loss_list = []
        self.idx = 0
        self.load_path = load_path
        self.load_idx  = load_idx

    def init_dataloader(self):
        self.ds = prepare_data_from_tfds(batch_size=1)
        metadata_path = "dataset/water_drop/metadata.json"
        with open(metadata_path, 'rt') as f:
            metadata = json.loads(f.read())
        # num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
        self.normalization_stats = {
            'acceleration': {
                'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
                'std': torch.FloatTensor(metadata['acc_std']).to(device),
                },
            'velocity': {
                'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
                'std': torch.FloatTensor(metadata['vel_std']).to(device),
                },
            }

    def init_models(self):

        # INITIALISING MODELS
        self.encoder = Encoder(self.normalization_stats, device=self.device, edge_features_dim=self.n_features, R=self.R)
        self.proc = Processor(self.n_features, self.n_features, self.n_features, self.n_features, M=self.M, device=self.device)
        self.decoder = Decoder(self.normalization_stats, node_features_dim=self.n_features).to(self.device)

        self.models = [self.encoder, self.proc, self.decoder]

        if self.load_path is not None:
            load_path = self.load_path
            load_idx = self.load_idx
            encoder_w = torch.load(os.path.join(load_path, f"Encoder/encoder_{load_idx}.pth"))
            self.encoder.load_state_dict(encoder_w)
            # self.encoder.eval()

            processor_w = torch.load(os.path.join(load_path, f"Processor/processor_{load_idx}.pth"))
            self.proc.load_state_dict(processor_w)
            # self.processor.eval()

            decoder_w = torch.load(os.path.join(load_path, f"Decoder/decoder_{load_idx}.pth"))
            self.decoder.load_state_dict(decoder_w)
            # self.decoder.eval()

        # OPTIMIZER
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.opt_proc = optim.Adam(self.proc.all_parameters(), lr=self.lr)
        self.opt_decoder = optim.Adam(self.decoder.parameters(), lr=self.lr)

        self.optimizers = [self.opt_encoder, self.opt_proc, self.opt_decoder]
        self.schedulers = [StepLR(optimizer, step_size=int(5e6), gamma=0.1) for optimizer in self.optimizers]



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
            # position_with_noise = add_noise(position)
            position_with_noise = position

            """
            n is the number of particles
            `partycle_type`: Integer values tensor of size n. Each value represent the material of the ith particle
            `position`: Float values tensor of size n x 6 x 2. It represents the last six positions (x, y) of the particles
            `n_particles_per_example`: Integer values Tensor of size 2 = [n1, n2] with n1 + n2 = n ????????? 

            `labels`: Float values tensor of size n x 2. It represents future positions to predict        
            """

            #  █████╗ ██████╗ ██████╗ ██╗  ██╗   ██╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗
            # ██╔══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║
            # ███████║██████╔╝██████╔╝██║   ╚████╔╝     ██╔████╔██║██║   ██║██║  ██║█████╗  ██║
            # ██╔══██║██╔═══╝ ██╔═══╝ ██║    ╚██╔╝      ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║
            # ██║  ██║██║     ██║     ███████╗██║       ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
            # ╚═╝  ╚═╝╚═╝     ╚═╝     ╚══════╝╚═╝       ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝

            # Create graph with features
            data = self.encoder(position_with_noise)
            # Process graph
            data = self.proc(data)
            # print("Processed Data: ", data)
            # extract acceleration using decoder + euler integrator
            acc_pred = self.decoder(data)
            # labels_est = integrator(position, acc)
            # print("Acceleration:", acc)

            # ██╗   ██╗██████╗ ██████╗  █████╗ ████████╗███████╗
            # ██║   ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
            # ██║   ██║██████╔╝██║  ██║███████║   ██║   █████╗
            # ██║   ██║██╔═══╝ ██║  ██║██╔══██║   ██║   ██╔══╝
            # ╚██████╔╝██║     ██████╔╝██║  ██║   ██║   ███████╗
            #  ╚═════╝ ╚═╝     ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝

            # Ground truth acceleration
            acc = get_acc(position, labels)
            # reset gradients
            for opt in self.optimizers:
                opt.zero_grad()
            # calculate loss
            loss = nn.MSELoss()(acc_pred, acc)
            # backpropagation
            loss.backward()
            # update parameters
            for opt in self.optimizers:
                opt.step()
            loss_logged = loss.item()

            nomove_loss = benchmark_nomove_acc(position_with_noise, acc)
            noacc_loss = benchmark_noacc_acc(position_with_noise, acc)
            nojerk_loss = benchmark_nojerk_acc(position_with_noise, acc)

            self.loss_list.append(loss_logged)
            self.mean_loss += loss_logged
            self.mean_loss_nomove += nomove_loss
            self.mean_loss_noacc += noacc_loss
            self.mean_loss_nojerk += nojerk_loss
            print("Loss is :", loss_logged)
            if self.idx % self.interval_tensorboard == 0:
                self.report(self.idx)
            if self.idx % 500 == 0:
                self.save_models(self.idx)

            [schedule.step() for schedule in self.schedulers]

    def save_models(self, i):
        for model in self.models:
            self.save_model(model, model.__class__.__name__, i)


    def report(self, i):
        self.loss_list = []
        super().report(i)
        self.writer.add_scalar("loss_plot/nomove", self.mean_loss_nomove / self.interval_tensorboard, i)
        self.writer.add_scalar("loss_plot/noacc", self.mean_loss_noacc / self.interval_tensorboard, i)
        self.writer.add_scalar("loss_plot/nojerk", self.mean_loss_nojerk / self.interval_tensorboard, i)
        self.mean_loss = 0
        self.mean_loss_nomove = 0
        self.mean_loss_noacc = 0
        self.mean_loss_nojerk = 0


if __name__ == "__main__":

    hyperparams = {
        "lr": 1e-4,
        "n_epochs": 20,
        "interval_tensorboard": 3,
        "n_features": 128, #  128
        "M": 5, # 10
        "R": 0.015, #0.015
        "load_path": "runs/fit/20220902-000721/models",
        "load_idx": 5000
    }
    device = torch.device("cpu")

    trainer = Trainer(hyperparams=hyperparams, seed=99, device=device)
    trainer.train()


#type "tensorboard --logdir=runs" in terminal
