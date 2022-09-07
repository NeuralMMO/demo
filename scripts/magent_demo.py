from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

from pettingzoo.magent import battle_v3
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

from ray.air.config import RunConfig
from ray.air.config import ScalingConfig
from ray.train.rl.rl_trainer import RLTrainer
from ray.tune.tuner import Tuner
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2 
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models import ModelCatalog

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

        self.HIDDEN = 16
        self.input = nn.Linear(13*13*5, self.HIDDEN)
        self.output = nn.Linear(self.HIDDEN, 21)
        self.lstm = BatchFirstLSTM(self.HIDDEN, self.HIDDEN)
        self.value = nn.Linear(self.HIDDEN, 1)

    def get_initial_state(self):
        return [self.value.weight.new(1, self.HIDDEN).zero_(),
                self.value.weight.new(1, self.HIDDEN).zero_()]

    def forward_rnn(self, x, state, seq_lens): 
        self._last_batch_size = x.shape[0]
        x = self.input(x)
        x, state = self.lstm(x, state)
        self.val = self.value(x)
        x = self.output(x)
        return x, state

    def value_function(self):
        return self.val.view(-1)

def env_creator():
    return aec_to_parallel_wrapper(
        battle_v3.env( 
            map_size=45, minimap_mode=False, step_reward=-0.005,
            dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
            max_cycles=1000, extra_features=False))

def make_policies(n):
    return {
        str(i): PolicySpec(
            policy_class=None,
            observation_space=None,
            action_space=None,
            config={"gamma": 0.85},
        )
        for i in range(n)
    }

def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    team = 'red' if agent_id.startswith('red') else 'blue'
    return str(hash(team + str(episode)) % num_policies)


ModelCatalog.register_custom_model('custom', Policy) 
register_env('custom', lambda config: ParallelPettingZooEnv(env_creator()))
num_policies = 5

trainer = RLTrainer(
    run_config=RunConfig(stop={"training_iteration": 1}),
    scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
    algorithm="PPO",
    config={
        "env": "custom",
        "framework": "torch",
        "num_sgd_iter": 1,
        "multiagent": {
            "policies": make_policies(num_policies),
            "policy_mapping_fn": policy_mapping_fn,
        },
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
