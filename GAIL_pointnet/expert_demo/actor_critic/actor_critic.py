import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from actor_critic.utils.distributions import Bernoulli, Categorical, DiagGaussian
from actor_critic.utils.base import MLPBase, CNNBase

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        '''
        base_kwargs : num_inputs, recurrent, hidden_size, out_size=
        '''
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            dist = Categorical
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            dist = DiagGaussian
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            dist = Bernoulli
        else:
            raise NotImplementedError

        self.base = base(num_inputs=obs_shape[0], **base_kwargs)
        self.dist = dist(self.base.output_size, num_outputs)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs=None, masks=None, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs=None, masks=None):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, action, rnn_hxs=None, masks=None):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs