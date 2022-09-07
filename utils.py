from pdb import set_trace as T

import gym
import numpy as np
import requests
import random

from collections import defaultdict

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

import nmmo
from rating import OpenSkillRating


def group(obs, policy_mapping_fn, episode):
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
        batch = data.shape[0]
        #data = data.reshape(batch, -1)
        data = data.squeeze()
        result = super().predict(data, **kwargs)
        result = np.concatenate(list(result.values())).reshape(1, -1)
        return result
        print(result)
        return result.reshape(1, -1)

def serve_rl_model(checkpoint: Checkpoint, name="RLModel") -> str:
    """Serve a RL model and return deployment URI.
    This function will start Ray Serve and deploy a model wrapper
    that loads the RL checkpoint into a RLPredictor.
    """
    serve.start(detached=True)
    deployment = PredictorDeployment.options(name=name)
    deployment.deploy(RLPredictor, checkpoint)
    return deployment.url

def run_game(episode, policy_mapping_fn, env_creator, endpoint_uri_list, horizon=1024):
    """Evaluate a served RL policy on a local environment.
    This function will create an RL environment and step through it.
    To obtain the actions, it will query the deployed RL model.
    """
    env = env_creator()
    env = gym.wrappers.RecordVideo(env, 'renders')

    obs = env.reset()
    env.render()
    policy_rewards = defaultdict(float)
    for t in range(horizon):
        # Compute actions per policy
        grouped_actions = {}
        for idx, vals, in group(obs, policy_mapping_fn, episode).items():
            grouped_actions[idx] = query_action(endpoint_uri_list[idx], vals)
        actions = ungroup(grouped_actions)

        # Centralized env step
        obs, rewards, dones, _ = env.step(actions)
        env.render()

        # Compute policy rewards
        for key, val in rewards.items():
            policy = policy_mapping_fn(key, episode)
            policy_rewards[policy] += val

        if all(list(dones.values())):
            break

    return policy_rewards


def run_tournament(policy_mapping_fn, env_creator, endpoint_uri_list, num_games=5, horizon=16):
    agents = [i for i in range(len(endpoint_uri_list))]
    ratings = OpenSkillRating(agents, 0)

    for episode in range(num_games):
        rewards = run_game(episode, policy_mapping_fn, env_creator, endpoint_uri_list)

        ratings.update(
                policy_ids=list(rewards),
                scores=list(rewards.values())
        )

        print(ratings)

    return ratings

def query_action(endpoint_uri: str, obs: np.ndarray):
    """Perform inference on a served RL model.
    This will send a HTTP request to the Ray Serve endpoint of the served
    RL policy model and return the result.
    """
    #action_dict = requests.post(endpoint_uri, json={"array": obs.tolist()}).json()
    #obs = {k: v.ravel().tolist() for k, v in obs.items()}
    #action_dict = requests.post(endpoint_uri, json=obs).json()
    vals = [v.tolist() for v in obs.values()]
    action_vals = requests.post(endpoint_uri, json={"array": vals}).json()
    action_vals = np.array(action_vals).reshape(8, -1)
    action_dict = {key: action_vals[:, i] for i, key in enumerate(obs)}
    #action_dict = {key: val for key, val in zip(list(obs.keys()), action_vals)}

    #action_dict = requests.post(endpoint_uri, json={"array": [[1]]}).json()
    return action_dict

def make_demo(rllib_config, env_creator, policy_mapping_fn, num_policies, num_workers, use_gpu, checkpoint_path):
    checkpoints = [Checkpoint(checkpoint_path) for _ in range(num_policies)]
    endpoint_uri_list = [serve_rl_model(c) for c in checkpoints]
    run_tournament(policy_mapping_fn, env_creator, endpoint_uri_list)
    serve.shutdown()  
