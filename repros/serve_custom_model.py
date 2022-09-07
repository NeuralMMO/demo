'''Cannot serve custom model without definition
Severity: High
Description: Cannot serve models without the original model definition.
Suggestion: Save model def as part of checkpoint
''' 

from pdb import set_trace as T
import numpy as np

import torch
from torch import nn

from pettingzoo.magent import battle_v3
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv

import utils


def env_creator():
    return aec_to_parallel_wrapper(
        battle_v3.env( 
            map_size=45, minimap_mode=False, step_reward=-0.005,
            dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
            max_cycles=1000, extra_features=False))

def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    team = 'red' if agent_id.startswith('red') else 'blue'
    return hash(team + str(episode)) % num_policies


register_env('custom', lambda config: ParallelPettingZooEnv(env_creator()))
num_policies = 5

# Checkpoint path obtained by running magent_demo.py
checkpoint_path='/home/jsuarez/ray_results/AIRPPO_2022-09-06_17-08-24/AIRPPO_13f85_00000_0_2022-09-06_17-08-39/checkpoint_000001'
utils.make_demo(None, env_creator, policy_mapping_fn, num_policies, 2, False, checkpoint_path)
