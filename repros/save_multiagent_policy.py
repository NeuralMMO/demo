from pdb import set_trace as T

from pettingzoo.magent import battle_v3
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

from ray.air.config import RunConfig
from ray.air.config import ScalingConfig
from ray.train.rl.rl_trainer import RLTrainer
from ray.tune.tuner import Tuner
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec


def env_creator():
    return aec_to_parallel_wrapper(
        battle_v3.env( 
            map_size=45, minimap_mode=False, step_reward=-0.005,
            dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
            max_cycles=1000, extra_features=False))

def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    return 'red' if agent_id.startswith('red') else 'blue'

register_env('custom', lambda config: ParallelPettingZooEnv(env_creator()))

trainer = RLTrainer(
    run_config=RunConfig(stop={"training_iteration": 1}),
    scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
    algorithm="PPO",
    config={
        "env": "custom",
        "framework": "torch",
        "num_sgd_iter": 1,
        "multiagent": {
            "policies": {
                "red": PolicySpec(
                    policy_class=None,
                    observation_space=None,
                    action_space=None,
                    config={"gamma": 0.85},
                ),
                "blue": PolicySpec(
                    policy_class=None,
                    observation_space=None,
                    action_space=None,
                    config={"gamma": 0.85},
                ),
            },
            "policy_mapping_fn": policy_mapping_fn,
        },
        "model": {
            "conv_filters": [
                [1, [13, 13], 1],
             ],
             "fcnet_hiddens": [1],
             "fcnet_activation": "relu",
        },
    }
)

tuner = Tuner(
    trainer,
    _tuner_kwargs={"checkpoint_at_end": True},
)

result = tuner.fit()[0] 

from ray.train.rl import RLCheckpoint
policy = RLCheckpoint.from_checkpoint(result.checkpoint).get_policy()
print(policy)
