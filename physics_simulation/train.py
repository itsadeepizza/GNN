import torch.nn as nn
import torch.optim as optim
import json
from config import selected_config as conf
from euler_integrator import integrator, get_acc
from base_trainer import BaseTrainer
from loader import prepare_data_from_tfds
import torch
import os
import matplotlib.pyplot as plt
import tempfile
# For creating a gif
import imageio
import datetime


def add_noise(position: torch.Tensor, std=0):
    noise = torch.randn(position.shape, device=position.device) * std
    noise[:, -1, :] = 0
    return position + noise



class Trainer(BaseTrainer):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.init_dataloader()
        self.init_models()


    def init_dataloader(self):
        self.train_ds = prepare_data_from_tfds(data_path=conf.TRAIN_DATASET, batch_size=conf.N_BATCH)
        self.validation_ds = prepare_data_from_tfds(data_path=conf.VALIDATION_DATASET,
                                                    shuffle=False, batch_size=conf.N_BATCH)
        self.test_ds = prepare_data_from_tfds(data_path=conf.TEST_DATASET,
                                                    shuffle=False, batch_size=1)

        with open(conf.METADATA, 'rt') as f:
            self.metadata = json.loads(f.read())
        # num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
        self.normalization_stats = {
            'acceleration': {
                'mean': torch.FloatTensor(self.metadata['acc_mean']).to(self.device),
                'std': torch.FloatTensor(self.metadata['acc_std']).to(self.device),
                },
            'velocity': {
                'mean': torch.FloatTensor(self.metadata['vel_mean']).to(self.device),
                'std': torch.FloatTensor(self.metadata['vel_std']).to(self.device),
                },
            }
        self.bounds = torch.tensor(self.metadata["bounds"], device=self.device)

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
            # I added "map_location=conf.DEVICE" to deal with error in google colab when gpu is
            # not available

            encoder_w = torch.load(os.path.join(conf.LOAD_PATH, f"Encoder/Encoder_{conf.LOAD_IDX}.pth"), map_location=conf.DEVICE)
            self.encoder.load_state_dict(encoder_w)
            # self.encoder.eval()

            processor_w = torch.load(os.path.join(conf.LOAD_PATH, f"Processor/Processor_{conf.LOAD_IDX}.pth"), map_location=conf.DEVICE)
            self.proc.load_state_dict(processor_w)
            # self.processor.eval()

            decoder_w = torch.load(os.path.join(conf.LOAD_PATH, f"Decoder/Decoder_{conf.LOAD_IDX}.pth"), map_location=conf.DEVICE)
            self.decoder.load_state_dict(decoder_w)
            # self.decoder.eval()

        # OPTIMIZER
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.opt_proc = optim.Adam(self.proc.parameters(), lr=self.lr)
        self.opt_decoder = optim.Adam(self.decoder.parameters(), lr=self.lr)

        self.optimizers = [self.opt_encoder, self.opt_proc, self.opt_decoder]


    def test(self):
        n_test = conf.N_TEST
        mean_test_loss = 0
        for model in self.models:
            model.eval()
        for i, (features, labels) in zip(range(n_test), self.validation_ds):
            positions = torch.tensor(features['position'])
            # Create batch index tensor (which batch each particle is assigned)
            batch_pos = features["n_particles_per_example"].cumsum(0)[:-1]
            batch_index = torch.zeros([len(positions)])
            batch_index[batch_pos] = 1
            batch_index = batch_index.cumsum(0)
            labels = torch.tensor(labels).to(self.device)
            with torch.no_grad():
                # model returns normalised predicted accelerations
                acc_pred = self.apply_model(positions, batch_index)
            # Calculate normalised ground truth accelerations
            acc_norm = get_acc(positions, labels, self.normalization_stats)  # normalised
            loss = nn.MSELoss()(acc_pred, acc_norm)
            mean_test_loss += loss.item() / n_test
        print(f"Loss on test set is {mean_test_loss:.3f}")
        self.writer.add_scalar("loss_test", mean_test_loss, self.idx)

    def simulate(self):
        """Run a simulation where initial sates are the outputs of the previous iteration"""
        def roll_position(gnn_position, labels_est):
            # Roll the position tensor to the left and insert the estimated last position
            rolled_position = gnn_position[:, 1:, :]
            return torch.cat((rolled_position, labels_est.unsqueeze(1)), dim=1)

        def plot_particles(dataset_positions, predicted_positions, step, frame_dir,
                           bounds=self.metadata["bounds"]):
            """Plot particles position with matplotlib"""
            fig, ax = plt.subplots(ncols=2)
            ax[0].scatter(x=dataset_positions[:, 0].to('cpu').numpy(), y=dataset_positions[:,
                                                               1].to('cpu').numpy(),
                          color='b', s=2)
            ax[1].scatter(x=predicted_positions[:, 0].to('cpu').numpy(), y=predicted_positions[:, 1].to('cpu').numpy(), color='r', s=2)
            ax[0].set_title("Ground trouth")
            ax[1].set_title("Simulation")
            ax[0].set_xlim([bounds[0][0] - 0.1, bounds[0][1] + 0.1])
            ax[1].set_xlim([bounds[0][0] - 0.1, bounds[0][1] + 0.1])
            ax[0].set_ylim([bounds[1][0] - 0.1, bounds[1][1] + 0.1])
            ax[1].set_ylim([bounds[1][0] - 0.1, bounds[1][1] + 0.1])
            fig.savefig(os.path.join(frame_dir, f"{step:04d}"))

        # Create a temporary folder, deleted at the execution of the function
        with tempfile.TemporaryDirectory() as tmpdirname:

            n_step = conf.N_STEP
            for model in self.models:
                model.eval()
            bounds = self.metadata["bounds"]
            positions_pred = None
            for i, (features, labels) in zip(range(n_step), self.test_ds):
                print(f"Step {i+1}/{n_step}")
                dataset_positions = torch.tensor(features['position'])
                if positions_pred is None:
                    simulation_positions = dataset_positions
                else:
                    simulation_positions = positions_pred
                # Create batch index tensor (which batch each particle is assigned)
                batch_index = torch.zeros([len(simulation_positions)])
                labels = torch.tensor(labels).to(self.device)
                with torch.no_grad():
                    # model returns normalised predicted accelerations
                    acc_pred = self.apply_model(simulation_positions, batch_index, denormalize=True)
                # Move all tensors to the same device
                simulation_positions = simulation_positions.to(conf.DEVICE)
                last_position_pred = integrator(simulation_positions, acc_pred)
                # Clip predicted positions to the bounds
                last_position_pred[:,0] = last_position_pred[:,0].clamp(bounds[0][0], bounds[0][1])
                last_position_pred[:,1] = last_position_pred[:, 1].clamp(bounds[1][0], bounds[1][1])

                positions_pred = roll_position(simulation_positions, last_position_pred)
                # Make plot
                plot_particles(labels, last_position_pred, i, tmpdirname)
            # Create gif
            os.makedirs(conf.ANIMATION_DIR, exist_ok=True)
            images = []
            now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            idx_as_str = f"{int(conf.LOAD_IDX / 1000)}k"
            print("Creating gif...", end="")
            for filename in os.listdir(tmpdirname):
                images.append(imageio.imread(os.path.join(tmpdirname, filename)))
            imageio.mimsave(os.path.join(conf.ANIMATION_DIR,
                                         f"simulation_{idx_as_str}_{now_str}.gif"),
                            images,
                            fps=conf.FPS)
            print("Done")



    def apply_model(self, positions, batch_index, denormalize=False):
        """Run model on positions"""
        # Create graph with features
        data = self.encoder(positions, batch_index)
        # Process graph
        data = self.proc(data)
        # extract acceleration using decoder
        acc_pred = self.decoder(data, denormalize=denormalize)
        return acc_pred


    def train(self):
        """Whole procedure for train"""
        for features, labels in self.train_ds:
            # predict, loss and backpropagation
            self.train_sample(features, labels)
            # Perform different opeartions at regular intervals
            if self.idx % conf.INTERVAL_TENSORBOARD == 0:
                # Write results on tensorboard
                self.log_tensorboard()
            if self.idx % conf.INTERVAL_SAVE_MODEL == 0:
                # Save models as pth
                self.save_models()
            if self.idx % conf.INTERVAL_UPDATE_LR == 0:
                # UPDATE LR
                self.update_lr()
                for optimizer in self.optimizers:
                    for g in optimizer.param_groups:
                        g['lr'] = self.lr
            if self.idx % conf.INTERVAL_TEST == 0:
                # Test model
                self.test()
            if self.idx % conf.INTERVAL_SIMULATION == 0:
                # Create simulation
                self.simulate()


    def train_sample(self, features, labels):
        self.idx += 1
        positions, predictions = self.make_prediction(features)
        loss = self.loss_calculation(positions, labels, predictions).item()
        self.mean_train_loss += loss
        print(f"Step {self.idx} - Loss = {loss:.5f}")

    def update_lr(self):
        self.lr = conf.LR_INIT * (conf.LR_DECAY ** (self.idx / conf.LR_STEP))



    def make_prediction(self, features):
        for model in self.models:
            model.train()

        features['position'] = torch.tensor(features['position']).to(self.device)
        features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(
            self.device)
        # Type of particle (water, sand) We use only water, so no need for it
        # features['particle_type'] = torch.tensor(features['particle_type']).to(self.device)

        positions = features["position"]
        # Create batch index tensor (which batch each particle is assigned, we need it for
        # building graph)
        batch_pos = features["n_particles_per_example"].cumsum(0)[:-1]
        batch_index = torch.zeros([len(positions)])
        batch_index[batch_pos] = 1
        batch_index = batch_index.cumsum(0).to(self.device)

        # add noise
        positions = add_noise(positions, std=conf.STD_NOISE)
        # apply model, it returns a normalised acceleration
        acc_pred = self.apply_model(positions, batch_index)
        return positions, acc_pred


    def loss_calculation(self, positions, labels, predictions):
        # Ground truth normalised acceleration
        labels = torch.tensor(labels).to(self.device)
        acc_norm = get_acc(positions, labels, self.normalization_stats)  # normalised
        # reset gradients
        for opt in self.optimizers:
            opt.zero_grad()
        # calculate loss
        loss = nn.MSELoss()(predictions, acc_norm)  # use normalised acc for
        # calculating loss
        # backpropagation
        loss.backward()
        # update parameters
        for opt in self.optimizers:
            opt.step()
        return loss


    def save_models(self):
        for model in self.models:
            self.save_model(model, model.__class__.__name__)
        # Plot weights as matrix - comment/uncomment below
        # self.writer.add_image("Encoder", self.plot_small_module(self.encoder, i), i)
        # self.writer.add_image("Proc - GN0", self.plot_small_module(self.proc.GNs[0], i), i)
        # self.writer.add_image("Proc - GN5", self.plot_small_module(self.proc.GNs[5], i), i)
        # # self.writer.add_image("Proc - GN9", self.plot_small_module(self.proc.GNs[9], i), i)
        # self.writer.add_image("Decoder", self.plot_small_module(self.decoder, i), i)


#type "tensorboard --logdir=runs --bind_all" in terminal
