from pdb import set_trace as T

from ray.air.config import RunConfig
from ray.air.config import ScalingConfig
from ray.train.rl.rl_trainer import RLTrainer
from ray.tune.tuner import Tuner
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv

def env_creator():
    return None

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
print('Saved ', result.checkpoint)
