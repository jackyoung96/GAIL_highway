import math
import torch
from torch.distributions import Normal
import numpy as np

def get_continuous_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action

def get_discrete_action(a):
	action_array = a.detach().numpy()
	b = np.zeros(action_array.squeeze().shape)
	b[np.argmax(action_array)] = 1.0
	return np.argmax(a.detach().numpy()) , action_array.squeeze() 

def get_entropy(mu, std):
    dist = Normal(mu, std)
    entropy = dist.entropy().mean()
    return entropy

def get_discrete_entropy(mu):
	entropy = torch.mul(mu, torch.log(mu))
	entropy = torch.sum(entropy, dim = 1)
	entropy = entropy.mean()
	return entropy

def log_prob_density(x, mu, std):
    log_prob_density = -(x - mu).pow(2) / (2 * std.pow(2)) \
                     - 0.5 * math.log(2 * math.pi)
    return log_prob_density.sum(1, keepdim=True)

def get_reward(discrim, state, action):
    state = torch.Tensor(state)
    action = torch.Tensor(action)
    state_action = torch.cat([state, action])
    with torch.no_grad():
        return -math.log(discrim(state_action)[0].item()) 
		# reward = negative cost, cost 는 discriminator의 log value로 사용될 수 있다.(Locally) 왜냐면 expert의 cost를 0으로 만들고 learner 는 -infty로 만드니까

def save_checkpoint(state, filename):
    torch.save(state, filename)