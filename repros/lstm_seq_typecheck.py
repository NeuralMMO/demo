'''Uncaught exception in RecurrentNet forward
Severity: Medium
Description: Defining custom recurrent networks is tricky because of a lack of shape checking. Incorrect usage throws internal errors.
Suggested fix: Add some shape checks
'''     

from pdb import set_trace as T

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
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2 
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models import ModelCatalog


class Policy(RecurrentNetwork, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.Module.__init__(self)

        self.input = nn.Linear(13*13*5, 32)
        self.output = nn.Linear(32, 4)
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.value = nn.Linear(32, 1)

    def forward_rnn(self, input_dict, state, seq_lens): 
        batch, time, _, _, _ = input_dict['obs']
        x = input_dict['obs'].reshape(-1, 13*13*5)
        #x = self.input(x).unsqueeze(1)
        x, state = self.lstm(x, state)
        self.val = self.value(x)
        x = self.output(x)
        return x, state

    def value_function(self):
        return self.val.squeeze(-1)

def env_creator():
    return aec_to_parallel_wrapper(
        battle_v3.env( 
            map_size=45, minimap_mode=False, step_reward=-0.005,
            dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
            max_cycles=1000, extra_features=False))


ModelCatalog.register_custom_model('custom', Policy) 
register_env('custom', lambda config: ParallelPettingZooEnv(env_creator()))

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
