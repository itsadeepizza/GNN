import torch
import random
import math
import os, datetime
from torch.utils.tensorboard import SummaryWriter
# import torchsummary
import time
import numpy.random
import numpy as np
from config import selected_config as conf
import inspect

class BaseTrainer():

    def __init__(self):
        # Init random seed
        seed = conf.get('SEED')
        if seed is None:
            seed = random.randint(0, 9999999)
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        numpy.random.seed(self.seed)

        # set device
        device = conf.get('DEVICE', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = device

        # Get idx
        self.idx = conf.get('LOAD_IDX', 0)

        self.init_logger()


    def init_logger(self):
        # TENSORBOARD AND LOGGING

        def generate_docker_style_name():
            """Generate name of the log folder"""
            import random

            # Select a random adjective from a list of common adjectives
            adjectives = ['adorable', 'beautiful', 'clean', 'drab', 'elegant', 'fancy',
                          'glamorous', 'handsome', 'long', 'magnificent', 'old',
                          'plain', 'quaint', 'sparkling', 'ugliest', 'unsightly', 'wide',
                          'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'gray', 'black',
                          'white', 'pink', 'brown']
            adjective = random.choice(adjectives)

            # Select a random noun from a list of common nouns
            nouns = ['person', 'year', 'way', 'day', 'thing', 'man', 'world', 'life', 'hand',
                     'part', 'child', 'eye', 'woman', 'place', 'work', 'week', 'case', 'point',
                     'government', 'company', 'number', 'group', 'problem', 'fact']
            noun = random.choice(nouns)

            # Add current datetime
            now_str = datetime.datetime.now().strftime("%HH%M-%d-%m")
            # Combine the adjective, noun and date to form the name
            name = adjective + '_' + noun + '_' + now_str
            return name

        # Create directories for logs
        self.log_dir = conf.ROOT_RUNS + "runs/fit/" + generate_docker_style_name()
        self.summary_dir = self.log_dir + "/summary"
        self.models_dir = self.log_dir + "/models"
        self.test_dir = self.log_dir + "/test"
        self.img_dir = self.log_dir + "/img"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)


        # Init writer
        self.writer = SummaryWriter(self.summary_dir)

        self.mean_train_loss = 0
        self.timer = 0

        # Log model info
        # model_stat = f"```{str(torchsummary.summary(self.model()))}```"
        # self.writer.add_text("Torchsummary", model_stat)
        # self.writer.add_text("Model name", str(self.model.__name__))
        # self.writer.add_text("Model code", "```  \n" + inspect.getsource(self.model) + "  \n```")

        # Log time
        self.writer.add_text("Time", datetime.datetime.now().strftime("%a %d %b %y - %H:%M"))
        # Log a formatted configuration on tensorboard
        config_as_table = "\n".join( [f"{param:>26} = {value}" for param, value in
                                 conf.__dict__.items()])
        self.writer.add_text("Configuration", config_as_table)

    def log_tensorboard(self):
        # Log always loss on train, steps per second and current learning rate
        self.writer.add_scalar("train_loss", self.mean_train_loss / conf.INTERVAL_TENSORBOARD, self.idx)
        tot_time = time.time() - self.timer
        self.timer = time.time()
        self.writer.add_scalar("steps_for_second", conf.INTERVAL_TENSORBOARD / tot_time, self.idx)
        self.writer.add_scalar("lr", self.lr, self.idx)
        self.mean_train_loss = 0


    def save_model(self, model, name: str):
        path = os.path.join(self.models_dir, name)
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(model.state_dict(), f"{path}/{name}_{self.idx}.pth")

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


    def plot_small_module(self, module: torch.nn.Module, step:int=0):
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
        module_name = module.__class__.__name__
        fig.suptitle(module_name, fontsize=26)
        fig.canvas.draw()
        fig.savefig(os.path.join(self.img_dir,f"{module_name}_{step:06}.png"))
        img = self.plot_to_tensorboard(fig)
        plt.close(fig)
        return img

