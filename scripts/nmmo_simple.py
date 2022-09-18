from pdb import set_trace as T

import nmmo

from pdb import set_trace as T

import functools

import os
import sys
import random
import time

import wandb
import gym
import numpy as np
import supersuit as ss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from ray.air.config import RunConfig
from ray.air.config import ScalingConfig  
from ray.tune.registry import register_env
from ray.tune.tuner import Tuner
from ray.train.rl.rl_trainer import RLTrainer
from ray import rllib
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.env import ParallelPettingZooEnv 

import nmmo
import config

from neural import policy, io, subnets
from config.cleanrl import Train

class Config(Train):
    RESPAWN = False

env_creator = lambda: nmmo.integrations.CleanRLEnv(Config())
register_env('nmmo', lambda config: ParallelPettingZooEnv(env_creator()))


test_env = env_creator()
obs = test_env.reset()

trainer = RLTrainer(
    run_config=RunConfig(stop={"training_iteration": 1}),
    scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
    algorithm="PPO",
    config={
        "env": "nmmo",
        "framework": "torch",
        "num_sgd_iter": 1,
        "model": {
            "fcnet_hiddens": [16],
            "fcnet_activation": "relu",
        },
    }
)

tuner = Tuner(
    trainer,
    _tuner_kwargs={"checkpoint_at_end": True},
)

result = tuner.fit()[0]
print('Saved ', result.checkpoint)

