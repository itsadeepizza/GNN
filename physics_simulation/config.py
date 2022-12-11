from pathlib import Path
import os
import torch

class Config():
    def __init__(config, **kwargs):
        config_variables_primitive = dict(

        root = "Path(__file__).resolve().parents[0]",
        N_BATCH=2,
        # LR is multiplied by LR_DECAY every LR_STEP
        LR_INIT=1E-4,
        LR_DECAY=1E-1,
        LR_STEP = 5e5,
        N_EPOCHS=20,
        INTERVAL_TENSORBOARD=100,
        N_FEATURES=128,  # 128
        M=10,  # 10
        R=0.015,  # 0.015
        STD_NOISE=1E-5,
        LOAD_PATH=None,  # RUNS/FIT/20221120-103911/MODELS,
        LOAD_IDX=0,
        SEED=99,
        INTERVAL_SAVE_UPDATE=2000,
        DEVICE=torch.device('cpu'),
        )

        # This config variables are calculated using other config variables, so
        # definition is as string
        config_variables_derivated = dict(
        ROOT_DATASET = "config.root / 'dataset'",
        ROOT_RUNS = "config.root",
        )



        # Set variable using dict above if no environment variables has not been defined
        for name_var, value in config_variables_primitive.items():
            if name_var not in kwargs:
                config.__setattr__(name_var, value)
            else:
                print(f'Setting {name_var} = {kwargs[name_var]}')
                config.__setattr__(name_var, kwargs[name_var])

        for name_var, expression in config_variables_derivated.items():
            if name_var not in kwargs:
                config.__setattr__(name_var, eval(expression))
            else:
                print(f'Setting {name_var} = {kwargs[name_var]}')
                config.__setattr__(name_var, kwargs[name_var])

    def set_config(self):
        import builtins
        builtins.config = self