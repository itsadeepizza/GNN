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
        self.writer.add_scalar("loss", self.mean_loss / self.interval_tensorboard, i)
        tot_time = time.time() - self.timer
        self.timer = time.time()
        self.writer.add_scalar("steps_for_second", self.interval_tensorboard / tot_time, i)

    def save_model(self, model, name: str, i: int):
        path = os.path.join(self.models_dir, name)
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(model.state_dict(), f"{path}/{name}_{i}.pth")