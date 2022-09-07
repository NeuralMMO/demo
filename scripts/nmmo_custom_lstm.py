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
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

import nmmo
import config

from neural import policy, io, subnets
from config.cleanrl import Train


class BatchFirstLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batch_first=True, **kwargs)

    def forward(self, input, hx):
        h, c       = hx
        h          = h.transpose(0, 1)
        c          = c.transpose(0, 1)
        hidden, hx = super().forward(input, [h, c])
        h, c       = hx
        h          = h.transpose(0, 1)
        c          = c.transpose(0, 1)
        return hidden, [h, c]

class Policy(RecurrentNetwork, nn.Module):
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
        self.lstm = BatchFirstLSTM(config.HIDDEN, config.HIDDEN)

    def get_initial_state(self):
        return [self.value.weight.new(1, self.config.HIDDEN).zero_(),
                self.value.weight.new(1, self.config.HIDDEN).zero_()]

    def forward_rnn(self, x, state, seq_lens):
        B, TT, _  = x.shape
        x         = x.reshape(B*TT, 10939)

        x         = nmmo.emulation.unpack_obs(self.config, x)
        lookup    = self.input(x)
        hidden, _ = self.policy(lookup)

        hidden        = hidden.view(B, TT, self.config.HIDDEN)
        hidden, state = self.lstm(hidden, state)
        hidden        = hidden.reshape(B*TT, self.config.HIDDEN)

        self.val = self.value(hidden).squeeze(-1)
        logits   = self.output(hidden, lookup)

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
        "rollout_fragment_length": 32,
        "train_batch_size": 128,
        "multiagent": {
            "count_steps_by": "agent_steps"
        },
        "model": {
            "custom_model": "custom",
            "max_seq_len": 16
        },
    }
)

tuner = Tuner(
    trainer,
    _tuner_kwargs={"checkpoint_at_end": True},
)

result = tuner.fit()[0]
print('Saved ', result.checkpoint)

