from pdb import set_trace as T

from ray.air.checkpoint import Checkpoint
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

import nmmo
import pufferlib

from config.cleanrl import Train
from policy import make_policy


def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    return 0

class Config(Train):
    RESPAWN = False
    HIDDEN = 2
    RENDER = True

config = Config()
#ModelCatalog.register_custom_model('custom', make_policy(config)) 

env_creator = lambda: nmmo.integrations.CleanRLEnv(Config())
pufferlib.rllib.register_env('nmmo', env_creator)

num_policies = 2
num_teams = 2
checkpoint_path = '/home/jsuarez/ray_results/AIRPPO_2022-09-18_01-52-11/AIRPPO_0b88d_00000_0_2022-09-18_01-52-14/checkpoint_000001'

tournament = pufferlib.evaluation.Tournament(num_policies, env_creator, policy_mapping_fn)

for i in range(num_policies):
    checkpoint = Checkpoint(checkpoint_path)
    tournament.add(f'policy_{i}', checkpoint, anchor=i==0)

for episode in range(3):
    ratings = tournament.run_match(episode)
    print(ratings)
        
from ray import serve
serve.shutdown()