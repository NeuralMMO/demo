from pdb import set_trace as T

import gym
import numpy as np
import requests
import random

from ray.air.checkpoint import Checkpoint
from ray.air.config import RunConfig
from ray.train.rl.rl_trainer import RLTrainer
from ray.train.rl import RLCheckpoint
from ray.air.config import ScalingConfig
from ray.train.rl.rl_predictor import RLPredictor as BaseRLPredictor
from ray.air.result import Result
from ray.serve import PredictorDeployment
from ray import serve
from ray.tune.tuner import Tuner
from ray.rllib.policy.policy import PolicySpec

#1. Fix checkpoint save/load
#2. Add multiagent env
#3. Load multiple policies
#4. Add NMMO
#5. Add renderer

from ray.tune.registry import register_env
from pettingzoo.magent import battle_v3
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from ray.rllib.env import ParallelPettingZooEnv

# define how to make the environment. This way takes an optional environment config, num_floors
env_creator = lambda config: aec_to_parallel_wrapper(battle_v3.env( 
        map_size=45, minimap_mode=False, step_reward=-0.005,
        dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
        max_cycles=1000, extra_features=False))

# register that way to make the environment under an rllib name
register_env('custom', lambda config: ParallelPettingZooEnv(env_creator(config)))

#env_creator = lambda config: gym.make('CartPole-v0')
#register_env('custom', lambda config: gym.make('CartPole-v0'))

def group_obs(obs, episode):
    groups = {}
    for k, v in obs.items():
        g = policy_mapping_fn(k, episode)
        if g not in groups:
            groups[g] = {}
        groups[g][k] = v
    return groups

def ungroup(groups):
    ungrouped = {}
    for g in groups.values():
        for k, v in g.items():
            assert k not in ungrouped
            ungrouped[k] = v
    return ungrouped

def create_policies(n):
    return {f'policy_{i}': 
        PolicySpec(
            policy_class=None,
            observation_space=None,
            action_space=None,
            config={"gamma": 0.85},
        )
        for i in range(n)
    }

def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    team = 'red' if agent_id.startswith('red') else 'blue'
    val = hash(team + str(episode)) % num_policies
    return val

class RLPredictor(BaseRLPredictor):
    def predict(self, data, **kwargs):
        data = data.reshape(-1, 13, 13, 5)
        result = super().predict(data, **kwargs)
        return result.reshape(1, -1)

'''
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
                "policy_mapping_fn":
                    lambda agent_id, episode, worker, **kwargs:
                        "red" if agent_id.startswith("red") else "blue"
            },
'''


def train_rl_ppo_online(num_workers: int, use_gpu: bool = False) -> Result:
    print("Starting online training")

    trainer = RLTrainer(
        run_config=RunConfig(stop={"training_iteration": 1}),
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
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
        },
    )

    tuner = Tuner(
        trainer,
        _tuner_kwargs={"checkpoint_at_end": True},
    )
    result = tuner.fit()[0]
    return result, trainer

def serve_rl_model(checkpoint: Checkpoint, name="RLModel") -> str:
    """Serve a RL model and return deployment URI.
    This function will start Ray Serve and deploy a model wrapper
    that loads the RL checkpoint into a RLPredictor.
    """
    serve.start(detached=True)
    deployment = PredictorDeployment.options(name=name)
    deployment.deploy(RLPredictor, checkpoint)
    return deployment.url

def evaluate_served_policy(endpoint_uri_list, num_episodes: int=1, horizon=128) -> list:
    """Evaluate a served RL policy on a local environment.
    This function will create an RL environment and step through it.
    To obtain the actions, it will query the deployed RL model.
    """
    env = env_creator(config={})
    env = gym.wrappers.RecordVideo(env, 'renders')

    for i in range(num_episodes):
        obs = env.reset()
        t = 0
        while True:
            grouped_obs = group_obs(obs, i)
            grouped_actions = {}

            for idx, vals, in grouped_obs.items():
                grouped_actions[idx] = query_action(endpoint_uri_list[idx], vals)

            actions = ungroup(grouped_actions)
            obs, r, dones, _ = env.step(actions)

            if all(list(dones.values())) or t >= horizon:
                break

            t += 1

def query_action(endpoint_uri: str, obs: np.ndarray):
    """Perform inference on a served RL model.
    This will send a HTTP request to the Ray Serve endpoint of the served
    RL policy model and return the result.
    """
    #action_dict = requests.post(endpoint_uri, json={"array": obs.tolist()}).json()
    #obs = {k: v.ravel().tolist() for k, v in obs.items()}
    #action_dict = requests.post(endpoint_uri, json=obs).json()
    vals = [v.ravel().tolist() for v in obs.values()]
    action_vals = requests.post(endpoint_uri, json={"array": vals}).json()
    action_dict = {key: val for key, val in zip(list(obs.keys()), action_vals)}

    #action_dict = requests.post(endpoint_uri, json={"array": [[1]]}).json()
    return action_dict

num_workers = 2
num_policies = 2
use_gpu = False
#result, trainer = train_rl_ppo_online(num_workers=num_workers, use_gpu=use_gpu)
#checkpoint = result.checkpoint

#checkpoint = RLCheckpoint.from_checkpoint(result.checkpoint)

'''
trainable = trainer.as_trainable()()
trainable.restore(result.checkpoint._local_path)
for key in 'red blue'.split():
    policy = trainable.get_policy(key)
    policy.export_checkpoint(key)
'''

checkpoint = Checkpoint(local_path='/home/jsuarez/ray_results/AIRPPO_2022-08-28_18-56-59/AIRPPO_c1d28_00000_0_2022-08-28_18-57-15/checkpoint_000001')

endpoint_uri_list = [serve_rl_model(checkpoint) for _ in range(num_policies)]
evaluate_served_policy(endpoint_uri_list)
serve.shutdown()
