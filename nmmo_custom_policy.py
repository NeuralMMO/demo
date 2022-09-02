from pdb import set_trace as T

import nmmo

import utils

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


class Policy(TorchModelV2, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.Module.__init__(self)
        self.config = config
        nmmo.io.action.Action.hook(config)

        self.input  = io.Input(config,
                embeddings=io.MixedEmbedding,
                attributes=subnets.SelfAttention)
        self.output = io.Output(config)
        self.value  = nn.Linear(config.HIDDEN, 1)
        self.policy = policy.Simple(config)

    def forward(self, input_dict, state, seq_lens):
        x         = nmmo.emulation.unpack_obs(self.config, input_dict['obs'])
        lookup    = self.input(x)
        hidden, _ = self.policy(lookup)

        #hidden, lstm_state = self.lstm(hidden, lstm_state)
        self.val = self.value(hidden).squeeze(-1)
        logits = self.output(hidden, lookup)

        flat_logits = []
        for atn in nmmo.Action.edges(self.config):
            for arg in atn.edges:
                flat_logits.append(logits[atn][arg])

        flat_logits = torch.cat(flat_logits, 1)
        return flat_logits, state

    def value_function(self):
        return self.val.view(-1)

class Config(Train):
    RESPAWN = False
    HIDDEN = 2


config = Config()
ModelCatalog.register_custom_model('custom', Policy) 
env_creator = lambda: nmmo.integrations.CleanRLEnv(Config())
register_env('custom', lambda config: ParallelPettingZooEnv(env_creator()))


test_env = env_creator()
obs = test_env.reset()

trainer = RLTrainer(
    run_config=RunConfig(stop={"training_iteration": 1}),
    scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
    algorithm="PPO",
    config={
        "env": "custom",
        "framework": "torch",
        "num_sgd_iter": 1,
        "model": {
            "custom_model": "custom",
        },
    }
)

tuner = Tuner(
    trainer,
    _tuner_kwargs={"checkpoint_at_end": True},
)

result = tuner.fit()[0]
print('Saved ', result.checkpoint)

