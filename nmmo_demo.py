from gc import callbacks
from pdb import set_trace as T

import nmmo

import numpy as np

import torch
import torch.nn as nn

import ray
from ray.air.config import RunConfig
from ray.air.config import ScalingConfig  
from ray.tune.registry import register_env
from ray.tune.tuner import Tuner
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.train.rl.rl_trainer import RLTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.env import ParallelPettingZooEnv 
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.algorithms.callbacks import DefaultCallbacks


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
        x         = x.reshape(B*TT, -1)

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

class NMMOLogger(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        assert len(base_env.envs) == 1, 'One env per worker'
        env = base_env.envs[0]

        for key, vals in stats.items():
            policy_stat = defaultdict(list)

            # Per-population metrics
            for policy_id, v in zip(policy_ids, vals):
                policy_stat[policy_id].append(v)

            for policy_id, vals in policy_stat.items():
                policy = inv_map[policy_id].__name__

                k = f'{policy}_{policy_id}_{key}'
                episode.custom_metrics[k] = np.mean(vals)

class Config(Train):
    RESPAWN = False
    HIDDEN = 2


# Dashboard fails on WSL
ray.init(include_dashboard=False, num_gpus=1)

config = Config()
ModelCatalog.register_custom_model('custom', Policy) 
env_creator = lambda: nmmo.integrations.CleanRLEnv(Config())
register_env('nmmo', lambda config: ParallelPettingZooEnv(env_creator()))

test_env = env_creator()
obs = test_env.reset()

trainer = RLTrainer(
    run_config=RunConfig(
        stop={"training_iteration": 3},
        callbacks=[
            WandbLoggerCallback(
                project='NeuralMMO',
                api_key_file='wandb_api_key',
                log_config=False,
            )
        ]
    ),
    callbacks=NMMOLogger,
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
    algorithm="PPO",
    config={
        "num_gpus": 1,
        "num_workers": 4,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 32,
        "train_batch_size": 2**10,
        "sgd_minibatch_size": 128,
        #"train_batch_size": 2**19,
        "num_sgd_iter": 1,
        "framework": "torch",
        "env": "nmmo",
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
    param_space={
        'callbacks': NMMOLogger,
    }
)

result = tuner.fit()[0]
print('Saved ', result.checkpoint)
#Checkpoint(local_path=/home/jsuarez/ray_results/AIRPPO_2022-09-02_19-01-08/AIRPPO_2aed2_00000_0_2022-09-02_19-01-25/checkpoint_000001)
