from pdb import set_trace as T

from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv 
from ray.air.checkpoint import Checkpoint

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

num_policies = 2
num_teams = 2
checkpoint_path = '/home/jsuarez/ray_results/AIRPPO_2022-09-12_22-02-51/AIRPPO_2dee5_00000_0_2022-09-12_22-02-54/checkpoint_000001'

tournament = utils.Tournament()

for i in range(num_policies):
    checkpoint = Checkpoint(checkpoint_path)
    tournament.add(f'policy_{i}', checkpoint, anchor=i==0)

for i in range(3):
    ratings = tournament.run_match(num_policies, env_creator, policy_mapping_fn)
    print(ratings)
        
from ray import serve
serve.shutdown()