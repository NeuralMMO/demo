from pdb import set_trace as T

import torch
import torch.nn as nn

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork

import nmmo
import pufferlib

from neural import policy, io, subnets

def make_policy(config):
    class Policy(RecurrentNetwork, nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            nn.Module.__init__(self)
            self.config = config
            nmmo.io.action.Action.hook(config)

            self.input  = io.Input(config,
                    embeddings=io.MixedEmbedding,
                    attributes=subnets.SelfAttention)
            self.output = io.Output(config)
            self.value  = nn.Linear(config.HIDDEN, 1)
            self.policy = policy.Simple(config)
            self.lstm = pufferlib.torch.BatchFirstLSTM(config.HIDDEN, config.HIDDEN)

        def get_initial_state(self):
            return [self.value.weight.new(1, self.config.HIDDEN).zero_(),
                    self.value.weight.new(1, self.config.HIDDEN).zero_()]

        def forward_rnn(self, x, state, seq_lens):
            B, TT, _  = x.shape
            x         = x.reshape(B*TT, -1)

            x         = nmmo.emulation.unpack_obs(self.config, x)
            lookup    = self.input(x)
            hidden, _ = self.policy(lookup)

            hidden        = hidden.view(B, TT, self.config.HIDDEN)
            hidden, state = self.lstm(hidden, state)
            hidden        = hidden.reshape(B*TT, self.config.HIDDEN)

            self.val = self.value(hidden).squeeze(-1)
            logits   = self.output(hidden, lookup)

            flat_logits = []
            for atn in nmmo.Action.edges(self.config):
                for arg in atn.edges:
                    flat_logits.append(logits[atn][arg])

            flat_logits = torch.cat(flat_logits, 1)
            return flat_logits, state

        def value_function(self):
            return self.val.view(-1)

    return Policy