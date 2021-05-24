import torch
import numpy as np
from utils.utils import get_discrete_entropy, get_entropy, log_prob_density
from torch.autograd import Variable

def train_discrim(discrim, memory, discrim_optim, demonstrations, args):
	memory = np.array(memory) 
	states = np.vstack(memory[:, 0]) # 2d array구조를 유지
	actions = list(memory[:, 1]) # 그냥 1d list로 만들기

	states = torch.Tensor(states)
	actions = torch.Tensor(actions)
		
	criterion = torch.nn.BCELoss()

	for _ in range(args.discrim_update_num): 
		learner = discrim(torch.cat([states, actions], dim=1)) # state랑 action을 붙여서 사용
		demonstrations = torch.Tensor(demonstrations)
		expert = discrim(demonstrations)

		discrim_loss = criterion(learner, torch.ones((states.shape[0], 1))) + \
						criterion(expert, torch.zeros((demonstrations.shape[0], 1)))
				
		discrim_optim.zero_grad()
		discrim_loss.backward()
		discrim_optim.step()

	expert_acc = ((discrim(demonstrations) < 0.5).float()).mean()
	learner_acc = ((discrim(torch.cat([states, actions], dim=1)) > 0.5).float()).mean()

	return expert_acc, learner_acc


def train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args):
	memory = np.array(memory) 
	states = np.vstack(memory[:, 0]) 
	actions = list(memory[:, 1]) 
	rewards = list(memory[:, 2]) 
	masks = list(memory[:, 3]) 
	with torch.autograd.set_detect_anomaly(True):

		old_values = critic(torch.Tensor(states)) 
		returns, advants = get_gae(rewards, masks, old_values, args) # return, advantage -> 매 step마다 전부 계산된 것
		
		mu = actor(torch.Tensor(states)) # action probability of state

		old_policy = torch.sum(torch.mul(mu, torch.Tensor(actions)) , 1) # old policy의 probability
		# old_policy = log_prob_density(torch.Tensor(actions), mu, std) # continuous action space일 때

		criterion = torch.nn.MSELoss()
		n = len(states)
		arr = np.arange(n)

		for _ in range(args.actor_critic_update_num):
			np.random.shuffle(arr)

			for i in range(n // args.batch_size): 
				batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
				batch_index = torch.LongTensor(batch_index)
				inputs = torch.Tensor(states)[batch_index]
				actions_samples = torch.Tensor(actions)[batch_index]
				returns_samples = returns.unsqueeze(1)[batch_index]
				advants_samples = advants.unsqueeze(1)[batch_index]
				oldvalue_samples = old_values[batch_index].detach()
				
				values = critic(inputs)
				clipped_values = oldvalue_samples + \
								torch.clamp(values - oldvalue_samples,
											-args.clip_param, 
											args.clip_param)
				critic_loss1 = criterion(clipped_values, returns_samples)
				critic_loss2 = criterion(values, returns_samples)
				critic_loss = torch.max(critic_loss1, critic_loss2).mean()

				loss, ratio, entropy = surrogate_loss(actor, advants_samples, inputs,
											old_policy.detach(), actions_samples,
											batch_index)
				clipped_ratio = torch.clamp(ratio,
											1.0 - args.clip_param,
											1.0 + args.clip_param)
				clipped_loss = clipped_ratio * advants_samples
				actor_loss = -torch.min(loss, clipped_loss).mean()

				loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

				critic_optim.zero_grad()
				actor_optim.zero_grad()
				# loss.backward(retain_graph=True) 
				loss.backward()
				actor_optim.step()
				critic_optim.step()

				

def get_gae(rewards, masks, values, args): # advantage value 구하는 것
	rewards = torch.Tensor(rewards)
	masks = torch.Tensor(masks)
	returns = torch.zeros_like(rewards)
	advants = torch.zeros_like(rewards)
	
	running_returns = 0
	previous_value = 0
	running_advants = 0

	for t in reversed(range(0, len(rewards))):
		running_returns = rewards[t] + (args.gamma * running_returns * masks[t])
		returns[t] = running_returns

		running_delta = rewards[t] + (args.gamma * previous_value * masks[t]) - \
										values.data[t]   ## TD error
		previous_value = values.data[t] 
		
		running_advants = running_delta + (args.gamma * args.lamda * \
											running_advants * masks[t]) ## Advantage
		advants[t] = running_advants

	advants = (advants - advants.mean()) / advants.std()  # 학습 안정성 높아짐 (normalization)
	return returns, advants

def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
	mu = actor(states)
	new_policy = torch.sum(torch.mul(mu, actions) , 1) # new policy 의 action probability
	# new_policy = log_prob_density(actions, mu, std)
	old_policy = old_policy[batch_index]

	ratio = torch.div(new_policy, old_policy)
	# ratio = torch.exp(new_policy - old_policy)
	surrogate_loss = ratio * advants
	entropy = get_discrete_entropy(mu) # discrete entropy
	# entropy = get_entropy(mu, std) # continuous entropy

	return surrogate_loss, ratio, entropy