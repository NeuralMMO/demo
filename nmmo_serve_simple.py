from pdb import set_trace as T

from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv 

import nmmo

from config.cleanrl import Train
import utils


def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    return 0

class Config(Train):
    RESPAWN = False
    HIDDEN = 2
    RENDER = True


env_creator = lambda: nmmo.integrations.CleanRLEnv(Config())
register_env('custom', lambda config: ParallelPettingZooEnv(env_creator()))

num_policies = 1
checkpoint_path = '/home/jsuarez/ray_results/AIRPPO_2022-09-07_10-44-08/AIRPPO_90459_00000_0_2022-09-07_10-44-24/checkpoint_000001'
utils.make_demo(None, env_creator, policy_mapping_fn, num_policies, 2, False, checkpoint_path)
