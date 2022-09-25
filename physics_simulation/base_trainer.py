import torch
import random
import math
import os, datetime
from torch.utils.tensorboard import SummaryWriter
# import torchsummary
import time
# import inspect
# import tabulate
import numpy.random
import numpy as np

class BaseTrainer():

    def __init__(self, hyperparams: dict, device=None, seed=None):
        if seed is None:
            seed = random.randint(0, 9999999)
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        numpy.random.seed(self.seed)

        # if gpu is to be used

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # SETTING PARAMETERS
        self.hparams = hyperparams
        for key, value in self.hparams.items():
            setattr(self, key, value)

        self.init_logger()

    def do_each_n(self, i, n):
        """Execute the event each n moves"""
        return abs(i % n - 0.5 * n) < 0.5 * self.batch_size

    def init_logger(self):
        # TENSORBOARD AND LOGGING
        # Create directories for logs
        layout = {
            "Loss": {
                "loss_plot": ["Multiline", ["loss_plot/train", 
                                            "loss_plot/nomove",
                                            "loss_plot/noacc",
                                            "loss_plot/nojerk",
                                            ]]
            },
        }
        now = datetime.datetime.now()
        now_str = now.strftime("%Y%m%d-%H%M%S")
        self.log_dir = "runs/fit/" + now_str
        self.summary_dir = self.log_dir + "/summary"
        self.models_dir = self.log_dir + "/models"
        self.test_dir = self.log_dir + "/test"
        os.makedirs(self.log_dir, exist_ok=True)
        # os.mkdir(summary_dir)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        self.writer = SummaryWriter(self.summary_dir)
        # Custom scalar for overlapping plots
        self.writer.add_custom_scalars(layout)
        self.mean_loss = 0
        self.timer = 0
        # LOG hyperparams

        # model_stat = f"```{str(torchsummary.summary(self.model()))}```"
        # self.writer.add_text("Torchsummary", model_stat)
        # self.writer.add_text("Time", now.strftime("%a %d %b %y - %H:%M"))
        # self.writer.add_text("Model name", str(self.model.__name__))
        # self.writer.add_text("Model code", "```  \n" + inspect.getsource(self.model) + "  \n```")
        # # log_hparams = tabulate.tabulate([[param, value] for param, value in self.hparams.items()],
        # #                                 headers=["NAME", "VALUE"], tablefmt="pipe")
        # self.writer.add_text("Hyperparameters", log_hparams)

    def report(self, i):
        self.writer.add_scalar("loss_plot/train", self.mean_loss / self.interval_tensorboard, i)
        tot_time = time.time() - self.timer
        self.timer = time.time()
        self.writer.add_scalar("steps_for_second", self.interval_tensorboard / tot_time, i)

    def save_model(self, model, name: str, i: int):
        path = os.path.join(self.models_dir, name)
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(model.state_dict(), f"{path}/{name}_{i}.pth")

    @staticmethod
    def plot_to_tensorboard(fig):
        """
        Takes a matplotlib figure handle and converts it using
        canvas and string-casts to a numpy array that can be
        visualized in TensorBoard using the add_image function

        Parameters:
            writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
            fig (matplotlib.pyplot.fig): Matplotlib figure handle.
            step (int): counter usually specifying steps/epochs/time.
        """


        # Draw figure on canvas
        fig.canvas.draw()

        # Convert the figure to numpy array, read the pixel values and reshape the array
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
        img = img / 255.0
        img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8
        img = np.swapaxes(img, 1, 2) # elsewhere image is inverted
        return img


    def plot_small_module(self, module: torch.nn.Module):
        """Plot weights for small modules"""
        import math
        import matplotlib.pyplot as plt
        import numpy as np
        import io

        def normalize(l):
            """extract detached weights from layer"""
            w = l.weight.detach().to("cpu").abs()
            if w.dim() == 1:
                w = w.unsqueeze(1)
            return w

        layers = [l for l in module.named_modules()][1:]  # first one is the module itself
        # parameters = [x[0] for x in self.encoder.named_parameters()] # for bias and ALL parameters in the module
        n_layers = len(layers)
        n_rows = math.ceil(n_layers / 2)
        fig, ax = plt.subplots(n_rows, 2, figsize=(14, n_rows * 7))
        for i, (name, layer) in enumerate(layers):
            ax_ = ax[i // 2][i % 2]
            ax_.imshow(normalize(layer), cmap="gray", vmin=0, vmax=0.3)
            ax_.set_title(name, fontsize=20)
        fig.suptitle(module.__class__.__name__, fontsize=26)
        # save the image in memory buffer
        # buf = io.BytesIO()
        # fig.savefig(buf, format='png')
        # buf.seek(0)
        # return buf
        fig.canvas.draw()
        img = self.plot_to_tensorboard(fig)
        plt.close(fig)
        return img

