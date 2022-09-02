'''Uncaught error when value function returns wrong data type
Severity: Medium
Description: Missing type and shape check on custom value functions
Suggested fix: Add these checks with the expected shape
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
from ray.rllib.models import ModelCatalog


class Policy(TorchModelV2, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.Module.__init__(self)

        self.fc = nn.Linear(13*13*5, 3)

    def forward(self, input_dict, state, seq_lens): 
        obs = input_dict['obs'].reshape(-1, 13*13*5)
        return self.fc(obs), state

    def value_function(self):
        return 

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
