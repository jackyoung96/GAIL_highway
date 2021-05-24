import os
import threading
import gym
import torch
import argparse

from model import Actor, Critic
from utils.utils import get_discrete_action
# from utils.running_state import ZFilter

import highway_env
from action_graphic import action_screen
import pygame
import PIL
from utils.zfilter import ZFilter



parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="highway-v0",
					help='name of Mujoco environement')
parser.add_argument('--iter', type=int, default=5,
					help='number of episodes to play')
parser.add_argument("--load_model", type=str, default='ckpt_27.pth.tar',
					 help="if you test pretrained file, write filename in save_model folder")
parser.add_argument('--hidden_size', type=int, default=100, 
					help='hidden unit size of actor, critic and discrim networks (default: 100)')
parser.add_argument('--vehicle_count', type=int, default=10,
					help='highway env kinematic state, number of vehicle')
args = parser.parse_args()

global_action = [0] * 5
lock = threading.Lock()

all_action = {
	0: 'left',
	1: 'idle',
	2: 'right',
	3: 'faster',
	4: 'slower'
}

def render_action(screen):
	global global_action

	# asc.display(global_action)
	
	while True:
		clock = pygame.time.Clock()
		# # lock.acquire()
		# try:
		box_size = 60
		font = pygame.font.SysFont('freesansbold.ttf', 36)
		for i,p in enumerate(global_action):
			
			a = font.render("%.2f"%p, False, (0,0,0))
			b = font.render(all_action[i],False,(0,0,0))
			color = (255,int(255*(1-p)),int(255*(1-p)))
			# pygame.draw.rect(screen, color, (i*box_size,150,box_size, box_size))
			screen.blit(a, (i*box_size,150))
			screen.blit(b, (i*box_size,180))
			# pygame.display.update()
		# finally:
		# 	lock.release()
		# clock.tick(15)


if __name__ == "__main__":
	env = gym.make(args.env)
	if args.env =='highway-v0':
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
			},
			'screen_height': 270,
 			'screen_width': 600,
			'simulation_frequency': 15,
			'offscreen_rendering': False,
			'real_time_rendering': True,
		}
		env.configure(config)
	env.seed(500)
	torch.manual_seed(500)
	env.reset()

	num_inputs = env.observation_space.shape
	num_inputs = num_inputs[0] * num_inputs[1]
	num_actions = env.action_space.n
	running_state = ZFilter((num_inputs,), clip=5)
	

	print("state size: ", num_inputs)
	print("action size: ", num_actions)

	actor = Actor(num_inputs, num_actions, args)
	critic = Critic(num_inputs, args)

	# running_state = ZFilter((num_inputs,), clip=5)
	
	if args.load_model is not None:
		pretrained_model_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))

		pretrained_model = torch.load(pretrained_model_path)

		actor.load_state_dict(pretrained_model['actor'])
		critic.load_state_dict(pretrained_model['critic'])

		# running_state.rs.n = pretrained_model['z_filter_n']
		# running_state.rs.mean = pretrained_model['z_filter_m']
		# running_state.rs.sum_square = pretrained_model['z_filter_s']

		# print("Loaded OK ex. ZFilter N {}".format(running_state.rs.n))

	else:
		assert("Should write pretrained filename in save_model folder. ex) python3 test_algo.py --load_model ppo_max.tar")


	actor.eval(), critic.eval()
	screen = pygame.display.set_mode((600, 270))
	t = threading.Thread(target=render_action, args=(screen,))
	t.daemon = True
	t.start()
	
	for episode in range(args.iter):
		state = env.reset()
		state = state.flatten()
		steps = 0
		score = 0
		
		for _ in range(10000):
			env.render()
			# print(screen)
			# image = PIL.Image.fromarray(env.render(mode='rgb_array'))
			# image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
			# screen.blit(image, (0,0))
			# # print(screen)
			# print(pygame.event.get())
			
			state = running_state(state)
			mu = actor(torch.Tensor(state).unsqueeze(0))
			action, action_array = get_discrete_action(mu)
			global_action = list(action_array)

			# box_size = 120
			# font = pygame.font.SysFont('freesansbold.ttf', 72)
			# for i,p in enumerate(global_action):
			# 	a = font.render(str(p), True, (0,0,0))
			# 	color = (255,int(255*(1-p)),int(255*(1-p)))
			# 	pygame.draw.rect(screen, color, (i*box_size,150,box_size, box_size))
			# 	screen.blit(a, (i*box_size,150))
			# pygame.display.update()

			

			next_state, reward, done, _ = env.step(action)
			# next_state = running_state(next_state)
			next_state = next_state.flatten()
			
			state = next_state
			score += reward
			
			if done:
				print("{} cumulative reward: {}".format(episode, score))
				break