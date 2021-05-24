import pickle
import torch
from actor_critic.actor_critic import Policy
import highway_env
import gym

env = gym.make('highway-v0')

policy = Policy([35], env.action_space)
policy.load_state_dict(torch.load('actor_critic/actor_critic.pt'))
policy = lambda x: actor_critic.act(torch.FloatTensor(x.flatten()).unsqueeze(0).cuda(), True)[1]