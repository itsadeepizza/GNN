from loader import prepare_data_from_tfds
from processor import Processor
from decoder import Decoder
from encoder import Encoder
from euler_integrator import integrator
import matplotlib.pyplot as plt
import torch
import os
import datetime

# model is represented by the (encoder, processor, decoder) tuple

device = torch.device("cpu")
n_features = 128 #  128
M = 5 # 10
now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
frame_dir = "frame/" + now_str
os.makedirs(frame_dir, exist_ok=True)
step = 0

def loadmodel(path, idx):
    encoder = Encoder(device=device, edge_features_dim=n_features)
    processor = Processor(n_features, n_features, n_features, n_features, M=M,
                          device=device)
    decoder = Decoder(node_features_dim=n_features).to(device)

    encoder_w = torch.load(os.path.join(path,f"Encoder/encoder_{idx}.pth"))
    encoder.load_state_dict(encoder_w)
    encoder.eval()

    processor_w = torch.load(os.path.join(path,f"Processor/processor_{idx}.pth"))
    processor.load_state_dict(processor_w)
    processor.eval()

    decoder_w = torch.load(os.path.join(path,f"Decoder/decoder_{idx}.pth"))
    decoder.load_state_dict(decoder_w)
    decoder.eval()

    return encoder, processor, decoder

# (enc, proc, dec) = loadmodel()

def predict(model, position):
    (encoder, proc, decoder) = model
    # Create graph with features + process graph
    with torch.no_grad():
        data = encoder(position)
        data = proc(data)
        # extract acceleration using decoder + euler integrator
        acc = decoder(data)
    return integrator(position, acc)


def plot_particles(labels, labels_est):
    """Plot particles position with matplotlib"""
    fig, ax = plt.subplots(ncols=2)
    ax[0].scatter(x=labels[:, 0].numpy(), y=labels[:, 1].numpy(), color='b', s=2)
    ax[1].scatter(x=labels_est[:, 0].numpy(), y=labels_est[:, 1].numpy(), color='r', s=2)
    ax[0].set_title("Ground trouth")
    ax[1].set_title("Simulation")
    fig.savefig(os.path.join(frame_dir, f"{step:04d}"))
    fig.show()


def roll_position(gnn_position, labels_est):
    rolled_position = gnn_position[:,1:,:]
    return torch.cat((rolled_position, labels_est.unsqueeze(1)), dim=1)
if __name__ == "__main__":
    gnn_position = None
    model = loadmodel("runs/fit/20220819-003931/models", 17000)
    test_ds = prepare_data_from_tfds(data_path='dataset/water_drop/train.tfrecord', shuffle=False)
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
        if gnn_position is None:
            print("Init position for GNN")
            gnn_position = position

        #  █████╗ ██████╗ ██████╗ ██╗  ██╗   ██╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗
        # ██╔══██╗██╔══██╗██╔══██╗██║  ╚██╗ ██╔╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║
        # ███████║██████╔╝██████╔╝██║   ╚████╔╝     ██╔████╔██║██║   ██║██║  ██║█████╗  ██║
        # ██╔══██║██╔═══╝ ██╔═══╝ ██║    ╚██╔╝      ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║
        # ██║  ██║██║     ██║     ███████╗██║       ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
        # ╚═╝  ╚═╝╚═╝     ╚═╝     ╚══════╝╚═╝       ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝

        labels_est = predict(model, gnn_position)
        gnn_position = roll_position(gnn_position, labels_est)

        plot_particles(labels, labels_est)
