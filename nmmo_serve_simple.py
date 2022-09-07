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

        config = Config()
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


def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    return 0
    return (agent_id-1) // 8

class Config(Train):
    RESPAWN = False
    HIDDEN = 2
    RENDER = True


env_creator = lambda: nmmo.integrations.CleanRLEnv(Config())
register_env('custom', lambda config: ParallelPettingZooEnv(env_creator()))

test_env = env_creator()
obs = test_env.reset()

#ModelCatalog.register_custom_model('custom', Policy) 

num_policies = 1
#checkpoint_path = '/home/jsuarez/ray_results/AIRPPO_2022-09-07_11-59-55/AIRPPO_26821_00000_0_2022-09-07_12-00-11/checkpoint_000001'
checkpoint_path = '/home/jsuarez/ray_results/AIRPPO_2022-09-07_10-44-08/AIRPPO_90459_00000_0_2022-09-07_10-44-24/checkpoint_000001'
utils.make_demo(None, env_creator, policy_mapping_fn, num_policies, 2, False, checkpoint_path)
