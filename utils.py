from pdb import set_trace as T

import gym
import numpy as np
import requests
import random

import torch
from torch import nn
from torch.nn.utils import rnn

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

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray import rllib

from ray.tune.registry import register_env as tune_register_env
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from ray.rllib.env import ParallelPettingZooEnv

def group_obs(obs, policy_mapping_fn, episode):
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

def register_env(env_creator, name):
    tune_register_env(name, lambda config: ParallelPettingZooEnv(env_creator()))    

class RLPredictor(BaseRLPredictor):
    def predict(self, data, **kwargs):
        data = data.reshape(-1, 4477)
        #data = data.reshape(-1, 13, 13, 5)
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
            #"model": {
            #    'custom_model': 'custom',
            #    'max_seq_len': 16,
            #},
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

def evaluate_served_policy(policy_mapping_fn, env_creator, endpoint_uri_list, num_episodes: int=1, horizon=128) -> list:
    """Evaluate a served RL policy on a local environment.
    This function will create an RL environment and step through it.
    To obtain the actions, it will query the deployed RL model.
    """
    env = env_creator()
    env = gym.wrappers.RecordVideo(env, 'renders')

    for i in range(num_episodes):
        obs = env.reset()
        t = 0
        while True:
            grouped_obs = group_obs(obs, policy_mapping_fn, i)
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

def make_demo(env_creator, policy_mapping_fn, num_policies, num_workers, use_gpu, checkpoint_path):
    if checkpoint_path is None:
        result, trainer = train_rl_ppo_online(num_workers=2, use_gpu=False)
        checkpoint = result.checkpoint
    else:
        checkpoint = Checkpoint(checkpoint_path)

    endpoint_uri_list = [serve_rl_model(checkpoint) for _ in range(num_policies)]
    evaluate_served_policy(policy_mapping_fn, env_creator, endpoint_uri_list)
    serve.shutdown()  
