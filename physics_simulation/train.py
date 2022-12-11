import torch.nn as nn
import torch.optim as optim
import json
from config import selected_config as conf
from euler_integrator import integrator, get_acc
from base_trainer import BaseTrainer
from loader import prepare_data_from_tfds
from benchmark import benchmark_nomove_acc, benchmark_noacc_acc, benchmark_nojerk_acc
import torch
from torch.optim.lr_scheduler import StepLR
import os
import matplotlib.pyplot as plt

def add_noise(position: torch.Tensor, std=0):

    noise = torch.randn(position.shape, device=position.device) * std
    noise[:, -1, :] = 0
    return position + noise



class Trainer(BaseTrainer):


    def __init__(self, load_path=None, load_idx=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_dataset = conf.ROOT_DATASET + '/water_drop/valid.tfrecord'
        self.test_dataset = conf.ROOT_DATASET + '/water_drop/valid.tfrecord'

        self.init_dataloader()
        self.init_models()

        self.mean_loss_nomove = 0
        self.mean_loss_noacc = 0
        self.mean_loss_nojerk = 0

        self.loss_list = []
        self.idx = load_idx
        self.load_path = conf.LOAD_PATH
        self.load_idx = conf.LOAD_IDX



    def init_dataloader(self):
        self.ds = prepare_data_from_tfds(data_path=self.train_dataset, batch_size=conf.N_BATCH)
        self.test_ds = prepare_data_from_tfds(data_path=self.test_dataset,
                                              shuffle=False, batch_size=conf.N_BATCH)
        metadata_path = conf.ROOT_DATASET + "/water_drop/metadata.json"
        with open(metadata_path, 'rt') as f:
            metadata = json.loads(f.read())
        # num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
        self.normalization_stats = {
            'acceleration': {
                'mean': torch.FloatTensor(metadata['acc_mean']).to(self.device),
                'std': torch.FloatTensor(metadata['acc_std']).to(self.device),
                },
            'velocity': {
                'mean': torch.FloatTensor(metadata['vel_mean']).to(self.device),
                'std': torch.FloatTensor(metadata['vel_std']).to(self.device),
                },
            }
        self.bounds = torch.tensor(metadata["bounds"], device=self.device)

    def init_models(self):
        from processor import Processor
        from decoder import Decoder
        from encoder import Encoder

        # INITIALISING MODELS
        self.encoder = Encoder(self.normalization_stats, self.bounds, device=self.device,
                               edge_features_dim=conf.N_FEATURES, R=conf.R)
        self.proc = Processor(conf.N_FEATURES, conf.N_FEATURES, conf.N_FEATURES, conf.N_FEATURES,
                              M=conf.M, device=self.device)
        self.decoder = Decoder(self.normalization_stats, node_features_dim=conf.N_FEATURES).to(self.device)

        self.models = [self.encoder, self.proc, self.decoder]
        for model in self.models:
            model.to(self.device)

        if conf.LOAD_PATH is not None:
            load_path = self.load_path
            load_idx = self.load_idx
            encoder_w = torch.load(os.path.join(load_path, f"Encoder/Encoder_{load_idx}.pth"))
            self.encoder.load_state_dict(encoder_w)
            # self.encoder.eval()

            processor_w = torch.load(os.path.join(load_path, f"Processor/Processor_{load_idx}.pth"))
            self.proc.load_state_dict(processor_w)
            # self.processor.eval()

            decoder_w = torch.load(os.path.join(load_path, f"Decoder/Decoder_{load_idx}.pth"))
            self.decoder.load_state_dict(decoder_w)
            # self.decoder.eval()

        # OPTIMIZER
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=conf.LR_INIT)
        self.opt_proc = optim.Adam(self.proc.parameters(), lr=conf.LR_INIT)
        self.opt_decoder = optim.Adam(self.decoder.parameters(), lr=conf.LR_INIT)

        self.optimizers = [self.opt_encoder, self.opt_proc, self.opt_decoder]
        # self.schedulers = [StepLR(optimizer, step_size=int(5e6), gamma=0.1) for optimizer in self.optimizers]



    def init_logger(self):
        # TENSORBOARD AND LOGGING
        super().init_logger()
        self.mean_ratio_board = torch.zeros([1], device=self.device)
        # variable to store the mean number of invalid moves (this value need to reduce)
        self.mean_error_game = torch.zeros([1], device=self.device)


    def test(self):
        n_test = 100
        loss_test = 0
        for model in self.models:
            model.eval()
        for i, (features, labels) in zip(range(n_test), self.test_ds):
            positions = torch.tensor(features['position']).to(self.device)
            # Create batch index tensor (which batch each particle is assigned)
            batch_pos = features["n_particles_per_example"].cumsum(0)[:-1]
            batch_index = torch.zeros([len(positions)])
            batch_index[batch_pos] = 1
            batch_index = batch_index.cumsum(0).to(self.device)
            labels = torch.tensor(labels).to(self.device)
            with torch.no_grad():
                acc_pred = self.apply_model(positions, batch_index)
            acc_norm = get_acc(positions, labels, self.normalization_stats)  # normalised
            loss = nn.MSELoss()(acc_pred, acc_norm)
            loss_test += loss.item() / n_test
        self.writer.add_scalar("loss_test", loss_test, self.idx)

    def apply_model(self, positions, batch_index, normalise=True):

        # Create graph with features
        data = self.encoder(positions, batch_index)
        # Process graph
        data = self.proc(data)
        # print("Processed Data: ", data)
        # extract acceleration using decoder
        acc_pred = self.decoder(data)
        return acc_pred


    def train(self):
        for epoch in range(conf.N_EPOCHS):
            print("Epoch ", epoch)
            self.train_epoch()
            self.save_models(self.idx)


    def train_epoch(self):
        for features, labels in self.ds:
            for model in self.models:
                model.train()
            self.idx += 1
            features['position'] = torch.tensor(features['position']).to(self.device)
            features['n_particles_per_example'] = torch.tensor(
                features['n_particles_per_example']).to(self.device)
            features['particle_type'] = torch.tensor(features['particle_type']).to(self.device)
            labels = torch.tensor(labels).to(self.device)
            positions = features["position"]
            # Create batch index tensor (which batch each particle is assigned)
            batch_pos = features["n_particles_per_example"].cumsum(0)[:-1]
            batch_index = torch.zeros([len(positions)])
            batch_index[batch_pos] = 1
            batch_index = batch_index.cumsum(0).to(self.device)

            print(positions.shape)
            # if positions.shape[0] < 1000:
            #     continue

            if any(features['particle_type'] != 5):
                print(features['particle_type'].unique())
                raise RuntimeError("Ma allora particle type puo davvero essere diverso da 5 !")

            # add noise
            positions = add_noise(positions, std=conf.STD_NOISE)

            acc_pred = self.apply_model(positions, batch_index, normalise=True)

            # ██╗   ██╗██████╗ ██████╗  █████╗ ████████╗███████╗
            # ██║   ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
            # ██║   ██║██████╔╝██║  ██║███████║   ██║   █████╗
            # ██║   ██║██╔═══╝ ██║  ██║██╔══██║   ██║   ██╔══╝
            # ╚██████╔╝██║     ██████╔╝██║  ██║   ██║   ███████╗
            #  ╚═════╝ ╚═╝     ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝

            # Ground truth normalised acceleration
            acc_norm = get_acc(positions, labels, self.normalization_stats) #normalised
            acc = get_acc(positions, labels, None) # original
            # reset gradients
            for opt in self.optimizers:
                opt.zero_grad()
            # calculate loss
            loss = nn.MSELoss()(acc_pred, acc_norm) # use normalised acc for calculating loss
            # backpropagation
            loss.backward()
            # update parameters
            for opt in self.optimizers:
                opt.step()
            loss_logged = loss.item()

            # Normalize outpout for better compairing with model predicted result
            nomove_loss = benchmark_nomove_acc(positions, acc, self.normalization_stats[
                'acceleration'])
            noacc_loss = (acc_norm.detach() ** 2).mean() #benchmark_noacc_acc(position_with_noise, acc, self.normalization_stats['acceleration'])
            nojerk_loss = benchmark_nojerk_acc(positions, acc, self.normalization_stats['acceleration'])

            self.loss_list.append(loss_logged)
            self.mean_loss += loss_logged
            self.mean_loss_nomove += nomove_loss
            self.mean_loss_noacc += noacc_loss
            self.mean_loss_nojerk += nojerk_loss
            print("Loss is :", loss_logged)
            if self.idx % conf.INTERVAL_TENSORBOARD == 0:
                self.report(self.idx)
            if self.idx % conf.INTERVAL_SAVE_UPDATE == 0:
                self.save_models(self.idx)
                self.test()
                # UPDATE LR
                self.lr = conf.LR_INIT * (conf.LR_DECAY ** (self.idx/conf.LR_STEP))
                for optimizer in self.optimizers:
                    for g in optimizer.param_groups:
                        g['lr'] = self.lr
            # [schedule.step() for schedule in self.schedulers]

    def save_models(self, i):
        for model in self.models:
            self.save_model(model, model.__class__.__name__, i)
        # Plot weights
        # self.writer.add_image("Encoder", self.plot_small_module(self.encoder, i), i)
        # self.writer.add_image("Proc - GN0", self.plot_small_module(self.proc.GNs[0], i), i)
        # self.writer.add_image("Proc - GN5", self.plot_small_module(self.proc.GNs[5], i), i)
        # # self.writer.add_image("Proc - GN9", self.plot_small_module(self.proc.GNs[9], i), i)
        # self.writer.add_image("Decoder", self.plot_small_module(self.decoder, i), i)



    def report(self, i):
        self.loss_list = []
        super().report(i)
        # self.writer.add_scalar("loss_plot/nomove", self.mean_loss_nomove / self.interval_tensorboard, i)
        self.writer.add_scalar("loss_plot/noacc", self.mean_loss_noacc /
                               conf.INTERVAL_TENSORBOARD, i)
        self.writer.add_scalar("loss_plot/nojerk", self.mean_loss_nojerk /
                               conf.INTERVAL_TENSORBOARD, i)
        self.writer.add_scalar("loss_over_benchmark", self.mean_loss / self.mean_loss_noacc, i)
        self.mean_loss = 0
        self.mean_loss_nomove = 0
        self.mean_loss_noacc = 0
        self.mean_loss_nojerk = 0





#type "tensorboard --logdir=runs" in terminal
