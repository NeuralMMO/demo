from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import ray
from ray.air import CheckpointConfig
from ray.air.config import RunConfig
from ray.air.config import ScalingConfig  
from ray.tune.tuner import Tuner
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.train.rl.rl_trainer import RLTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import nmmo
import pufferlib

from config.cleanrl import Train
from policy import make_policy


class NMMOLogger(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        T()

        assert len(base_env.envs) == 1, 'One env per worker'
        env = base_env.envs[0].par_env

        inv_map = {agent.policyID: agent for agent in env.config.PLAYERS}

        stats = env.terminal()
        stats = {**stats['Player'], **stats['Env']}
        policy_ids = stats.pop('PolicyID')

        for key, vals in stats.items():
            policy_stat = defaultdict(list)

            # Per-population metrics
            for policy_id, v in zip(policy_ids, vals):
                policy_stat[policy_id].append(v)

            for policy_id, vals in policy_stat.items():
                policy = inv_map[policy_id].__name__

                k = f'{policy}_{policy_id}_{key}'
                episode.custom_metrics[k] = np.mean(vals)

        return super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            **kwargs
        )


class Config(Train):
    RESPAWN = False
    HIDDEN = 2

# Dashboard fails on WSL
ray.init(include_dashboard=False, num_gpus=1)

Config.HORIZON = 16
config = Config()
ModelCatalog.register_custom_model('custom', make_policy(config)) 
env_creator = lambda: nmmo.integrations.CleanRLEnv(Config())
pufferlib.rllib.register_env('nmmo', env_creator)

test_env = env_creator()
obs = test_env.reset()

trainer = RLTrainer(
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
    algorithm="PPO",
    config={
        "num_gpus": 1,
        "num_workers": 4,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 32,
        "train_batch_size": 2**10,
        #"train_batch_size": 2**19,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 1,
        "framework": "torch",
        "env": "nmmo",
        "multiagent": {
            "count_steps_by": "agent_steps"
        },
        "model": {
            "custom_model": "custom",
            'custom_model_config': {'config': config},
            "max_seq_len": 16
        },
    }
)

tuner = Tuner(
    trainer,
    _tuner_kwargs={"checkpoint_at_end": True},
    run_config=RunConfig(
        local_dir='results',
        verbose=1,
        stop={"training_iteration": 5},
        checkpoint_config=CheckpointConfig(
            num_to_keep=5,
            checkpoint_frequency=1,
        ),
        callbacks=[
            WandbLoggerCallback(
                project='NeuralMMO',
                api_key_file='wandb_api_key',
                log_config=False,
            )
        ]
    ),
    param_space={
        'callbacks': NMMOLogger,
    }
)

result = tuner.fit()[0]
print('Saved ', result.checkpoint)

#policy = RLCheckpoint.from_checkpoint(result.checkpoint).get_policy()

'''
def multiagent_self_play(trainer: Type[Trainer]):
    new_weights = trainer.get_policy("player1").get_weights()
    for opp in Config.OPPONENT_POLICIES:
        prev_weights = trainer.get_policy(opp).get_weights()
        trainer.get_policy(opp).set_weights(new_weights)
        new_weights = prev_weights

local_weights = trainer.workers.local_worker().get_weights()
trainer.workers.foreach_worker(lambda worker: worker.set_weights(local_weights))
'''