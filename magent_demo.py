from pdb import set_trace as T

from pettingzoo.magent import battle_v3
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

import utils


def env_creator():
    return aec_to_parallel_wrapper(
        battle_v3.env( 
            map_size=45, minimap_mode=False, step_reward=-0.005,
            dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
            max_cycles=1000, extra_features=False))

def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    team = 'red' if agent_id.startswith('red') else 'blue'
    val = hash(team + str(episode)) % num_policies
    return val


num_policies = 2

checkpoint = local_path='/home/jsuarez/ray_results/AIRPPO_2022-08-28_18-56-59/AIRPPO_c1d28_00000_0_2022-08-28_18-57-15/checkpoint_000001'
checkpoint = None

utils.register_env(env_creator, 'custom')
utils.make_demo(env_creator, policy_mapping_fn, num_policies, num_workers=2, use_gpu=False, checkpoint_path=checkpoint)
