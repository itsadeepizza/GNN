import os
from loader import prepare_data_from_tfds, prepare_data_from_tfds_test
from builtins import config as conf
from processor import Processor
from decoder import Decoder
from encoder import Encoder
from euler_integrator import integrator, get_acc
import matplotlib.pyplot as plt
import torch
import datetime
import json

import builtins.config as conf

# model is represented by the (encoder, processor, decoder) tuple

device = torch.device("cpu")
n_features = 128 #  128
M = 10 # 10
now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
frame_dir = "frame/" + now_str
os.makedirs(frame_dir, exist_ok=True)
step = 0

metadata_path = "dataset/water_drop/metadata.json"
with open(metadata_path, 'rt') as f:
    metadata = json.loads(f.read())
normalization_stats = {
        'acceleration': {
            'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
            'std': torch.FloatTensor(metadata['acc_std']).to(device),
            },
        'velocity': {
            'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
            'std': torch.FloatTensor(metadata['vel_std']).to(device),
            },
        }
bounds = torch.tensor(metadata["bounds"], device=device)

def loadmodel(path, idx):

    encoder = Encoder(normalization_stats, bounds, device=device, edge_features_dim=n_features)
    processor = Processor(n_features, n_features, n_features, n_features, M=M,
                          device=device)


    decoder = Decoder(normalization_stats, node_features_dim=n_features).to(device)

    encoder_w = torch.load(os.path.join(path,f"Encoder/Encoder_{idx}.pth"))
    encoder.load_state_dict(encoder_w)
    encoder.eval()

    processor_w = torch.load(os.path.join(path,f"Processor/Processor_{idx}.pth"))
    processor.load_state_dict(processor_w)
    processor.eval()

    decoder_w = torch.load(os.path.join(path,f"Decoder/Decoder_{idx}.pth"))
    decoder.load_state_dict(decoder_w)
    decoder.eval()

    return encoder, processor, decoder

# (enc, proc, dec) = loadmodel()

def predict(model, position, batch_index):
    (encoder, proc, decoder) = model
    # Create graph with features + process graph
    with torch.no_grad():
        data = encoder(position, batch_index)
        data = proc(data)
        # extract acceleration using decoder + euler integrator
        acc = decoder(data, denormalize=False)

    return acc


def plot_particles(labels, labels_est):
    """Plot particles position with matplotlib"""
    fig, ax = plt.subplots(ncols=2)
    ax[0].scatter(x=labels[:, 0].numpy(), y=labels[:, 1].numpy(), color='b', s=2)
    ax[1].scatter(x=labels_est[:, 0].numpy(), y=labels_est[:, 1].numpy(), color='r', s=2)
    ax[0].set_title("Ground trouth")
    ax[1].set_title("Simulation")
    ax[0].set_xlim([bounds[0][0] - 0.1, bounds[0][1] + 0.1])
    ax[1].set_xlim([bounds[0][0] - 0.1, bounds[0][1] + 0.1])
    ax[0].set_ylim([bounds[1][0] - 0.1, bounds[1][1] + 0.1])
    ax[1].set_ylim([bounds[1][0] - 0.1, bounds[1][1] + 0.1])
    fig.savefig(os.path.join(frame_dir, f"{step:04d}"))
    fig.show()

def integrate_position(positions, acc):
    """Return calculated new positions applying acceleration on old positions"""
    last_velocity = positions[:, -1, :] - positions[:, -2, :]
    new_velocity = last_velocity + acc
    new_position = positions[:, -1, :] + new_velocity
    return new_position

def roll_position(gnn_position, labels_est):
    rolled_position = gnn_position[:,1:,:]
    return torch.cat((rolled_position, labels_est.unsqueeze(1)), dim=1)
if __name__ == "__main__":
    conf.ROOT_DATASET = 'dataset'
    gnn_position = None

    model = loadmodel("runs/fit/20221123-003344/models", 302000)
    # test_ds = prepare_data_from_tfds(data_path='dataset/water_drop/train.tfrecord', shuffle=False, batch_size=1)
    # test_ds = prepare_data_from_tfds_test(data_path='dataset/water_drop/valid.tfrecord', is_rollout=True, shuffle=False, batch_size=1)
    test_ds = prepare_data_from_tfds(data_path='dataset/water_drop/valid.tfrecord', shuffle=False, batch_size=1)

    for features, labels in test_ds:
        step += 1
        print(step)
        # forward
        # ███████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗    ██╗███╗   ██╗███████╗ ██████╗
        # ██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝    ██║████╗  ██║██╔════╝██╔═══██╗
        # █████╗   ╚███╔╝    ██║   ██████╔╝███████║██║        ██║       ██║██╔██╗ ██║█████╗  ██║   ██║
        # ██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══██║██║        ██║       ██║██║╚██╗██║██╔══╝  ██║   ██║
        # ███████╗██╔╝ ██╗   ██║   ██║  ██║██║  ██║╚██████╗   ██║       ██║██║ ╚████║██║     ╚██████╔╝
        # ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝       ╚═╝╚═╝  ╚═══╝╚═╝      ╚═════╝
        features['position'] = torch.tensor(features['position']).to(device)
        features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(device)
        features['particle_type'] = torch.tensor(features['particle_type']).to(device)
        labels = torch.tensor(labels).to(device)
        position = features["position"]
        batch_index = torch.zeros(len(position))
        if gnn_position is None:
            print("Init position for GNN")
            gnn_position = position

        #  █████╗ ██████╗ ██████╗ ██╗  ██╗   ██╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗
        # ██╔══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║
        # ███████║██████╔╝██████╔╝██║   ╚████╔╝     ██╔████╔██║██║   ██║██║  ██║█████╗  ██║
        # ██╔══██║██╔═══╝ ██╔═══╝ ██║    ╚██╔╝      ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║
        # ██║  ██║██║     ██║     ███████╗██║       ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
        # ╚═╝  ╚═╝╚═╝     ╚═╝     ╚══════╝╚═╝       ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝

        acc_est_norm = predict(model, gnn_position, batch_index)
        acc_est = acc_est_norm * normalization_stats['acceleration']['std']
        # acc_est = acc_est_norm * normalization_stats['acceleration']['std'] + normalization_stats['acceleration']['mean']
        acc = get_acc(gnn_position, labels)
        acc_norm = get_acc(position, labels, normalization_stats)
        print(acc_norm)
        print(acc_est_norm)
        position_est = integrator(gnn_position, acc_est)
        gnn_position = roll_position(gnn_position, position_est)

        plot_particles(labels, position_est)

#ffmpeg -f image2  -framerate 50 -i %004d.png animation.gif
