import os
import gym
import highway_env
import pickle
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter 

from utils.utils import *
from utils.zfilter import ZFilter
from model_pointnet import Actor, Critic, Discriminator
from train_model import train_actor_critic, train_discrim

# from action_graphic import action_screen
import threading
from datetime import datetime
import os

parser = argparse.ArgumentParser(description='PyTorch GAIL')
parser.add_argument('--env_name', type=str, default="highway-v0", 
					help='name of the environment to run')
parser.add_argument('--load_model', type=str, default=None, 
					help='path to load the saved model')
parser.add_argument('--render', action="store_true", default=False, 
					help='if you dont want to render, set this to False')
parser.add_argument('--gamma', type=float, default=0.99, 
					help='discounted factor (default: 0.99)')
parser.add_argument('--lamda', type=float, default=0.98, 
					help='GAE hyper-parameter (default: 0.98)')
parser.add_argument('--hidden_size_1', type=int, default=100, 
					help='hidden unit size of actor, critic and discrim networks (default: 100)')
parser.add_argument('--hidden_size_2', type=int, default=100, 
					help='hidden unit size of actor, critic and discrim networks (default: 100)')
parser.add_argument('--learning_rate', type=float, default=3e-4, 
					help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-3, 
					help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--clip_param', type=float, default=0.2, 
					help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--discrim_update_num', type=int, default=2, 
					help='update number of discriminator (default: 2)')
parser.add_argument('--actor_critic_update_num', type=int, default=10, 
					help='update number of actor-critic (default: 10)')
parser.add_argument('--total_sample_size', type=int, default=512, 
					help='total sample size to collect before PPO update (default: 512)')
parser.add_argument('--batch_size', type=int, default=64, 
					help='batch size to update (default: 64)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.8,
					help='accuracy for suspending discriminator about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.8,
					help='accuracy for suspending discriminator about generated data (default: 0.8)')
parser.add_argument('--max_iter_num', type=int, default=4000,
					help='maximal number of main iterations (default: 4000)')
parser.add_argument('--seed', type=int, default=500,
					help='random seed (default: 500)')
parser.add_argument('--logdir', type=str, default='logs',
					help='tensorboardx logs directory')
parser.add_argument('--vehicle_count', type=int, default=10,
					help='highway env kinematic state, number of vehicle')
args = parser.parse_args()


# def render_action(action):
# 	screen = action_screen()


def main():
	env = gym.make(args.env_name)
	if args.env_name =='highway-v0':
		config = {
			"observation": {
				"type": "Kinematics",
				"vehicles_count": args.vehicle_count,
				"features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
				"features_range": {
					"x": [-100, 100],
					"y": [-100, 100],
					"vx": [-20, 20],
					"vy": [-20, 20]
				},
				"absolute": False
				# "order": "sorted"
			},
			# "manual_control": True
			"action": {
				"type": "DiscreteMetaAction"
			}
		}
		env.configure(config)
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	env.reset()

	num_inputs = env.observation_space.shape[1]
	num_cars = env.observation_space.shape[0]
	num_actions = env.action_space.n
	running_state = ZFilter((num_cars,num_inputs), clip=5)

	print('state size:', num_inputs) 
	print('action size:', num_actions)

	actor = Actor(num_inputs, num_actions, args) # actor
	critic = Critic(num_inputs, args) # critic
	discrim = Discriminator(num_inputs, num_actions, args) # discriminator

	actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate) # 
	critic_optim = optim.Adam(critic.parameters(), lr=args.learning_rate, 
							  weight_decay=args.l2_rate) # regularizer coefficient l2_rate
	discrim_optim = optim.Adam(discrim.parameters(), lr=args.learning_rate)
	
	# # load demonstrations
	expert_demo = pickle.load(open('./expert_demo/expert_demo.p', "rb")) # 수정 필요함 -> demo 형식을 어떻게? -> [state, action] 전부 붙여서 trajectory 형태로
	demonstrations = np.array(expert_demo)
	print("demonstrations.shape", demonstrations.shape)
	date = datetime.now()
	writer = SummaryWriter(os.path.join(args.logdir,date.strftime("%Y%b%d_%H_%M_%S")))

	# 잠시 테스트 ############
	# demo_action = demonstrations[:,70:]
	# demo_action = list(np.argmax(demo_action, axis = 1))
	# for i in range(5):
	# 	print("action%d"%i,demo_action.count(i)/len(demo_action))
	
	########################

	

	# if args.load_model is not None:
	#     saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
	#     ckpt = torch.load(saved_ckpt_path)

	#     actor.load_state_dict(ckpt['actor'])
	#     critic.load_state_dict(ckpt['critic'])
	#     discrim.load_state_dict(ckpt['discrim'])

	#     running_state.rs.n = ckpt['z_filter_n']
	#     running_state.rs.mean = ckpt['z_filter_m']
	#     running_state.rs.sum_square = ckpt['z_filter_s']

	#     print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

	
	episodes = 0
	train_discrim_flag = True
	# t = threading.Thread(target=render_action)
	# t.daemon = True

	for iter in range(args.max_iter_num): # default 4000번 
		actor.eval(), critic.eval()
		memory = deque()

		steps = 0
		scores = []

		while steps < args.total_sample_size: 
			state = env.reset()
			score = 0
			state = running_state(state) ## -> 이게 뭔지 잘 모르겠는데... 뭔가 normalization 해주는 느낌이다.
			action_append = []
			for _ in range(10000): 
				if args.render:
					env.render()

				steps += 1

				mu = actor(torch.Tensor(state))
				action, action_array = get_discrete_action(mu)
				action_append.append(action)
				next_state, reward, done, _ = env.step(action)
				irl_reward = get_reward(discrim, state, action_array) # negative local cost(논문에 나와있다)

				if done:
					mask = 0
				else:
					mask = 1

				memory.append([state, action_array, irl_reward, mask]) # replay buffer?

				next_state = running_state(next_state)
				state = next_state

				score += reward

				if done:
					break
			
			episodes += 1
			scores.append(score)
		
		# sample이 꽉 차면 학습 시작
		score_avg = np.mean(scores)
		print('{}:: {} episode score is {:.2f}'.format(iter, episodes, score_avg))
		writer.add_scalar('log/score', float(score_avg), iter)

		writer.add_scalars('action_distribution', {'LANE_LEFT': action_append.count(0)/len(action_append),
													'IDLE': action_append.count(1)/len(action_append),
													'LANE_RIGHT': action_append.count(2)/len(action_append),
													'FASTER': action_append.count(3)/len(action_append),
													'SLOWER': action_append.count(4)/len(action_append)}, iter)

		actor.train(), critic.train(), discrim.train()
		if train_discrim_flag:
			expert_acc, learner_acc = train_discrim(discrim, memory, discrim_optim, demonstrations, args)  # discriminator 학습하기
			print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
			# if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
			# 	train_discrim_flag = False # discriminator가 충분히 학습됨
		train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args) # generator 만 학습시킴 ( actor-critic )

		if iter % 100:
			score_avg = int(score_avg)

			model_path = os.path.join(os.getcwd(),'save_model')
			if not os.path.isdir(model_path):
				os.makedirs(model_path)

			ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

			save_checkpoint({
				'actor': actor.state_dict(),
				'critic': critic.state_dict(),
				'discrim': discrim.state_dict(),
				'z_filter_n':running_state.rs.n,
				'z_filter_m': running_state.rs.mean,
				'z_filter_s': running_state.rs.sum_square,
				'args': args,
				'score': score_avg
			}, filename=ckpt_path)

if __name__=="__main__":
	main()