import pickle
import gym
import highway_env
from matplotlib import pyplot as plt
import readchar
import threading

total_expert_data = 1000


arrow_keys = {
	readchar.key.LEFT: "SLOWER",
    readchar.key.UP: "LANE_LEFT",
    readchar.key.RIGHT: "FASTER",
	readchar.key.DOWN: "LANE_RIGHT"
}
all_action = {
	'LANE_LEFT': 0,
	'IDLE': 1,
	'LANE_RIGHT': 2,
	'FASTER': 3,
	'SLOWER': 4
}


action_key = None
Finished = False

def getAction():
	global action_key
	print('AAA')
	while True:
		key = readchar.readkey()
		print('arrow')
		action_key = arrow_keys[key]
		print(action_key)




def main():
	global action_key, Finished, total_expert_data
	env = gym.make("highway-v0")
	config = {
		"observation": {
			"type": "Kinematics",
			"vehicles_count": 10, ## 일단은 10개로 만들기
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
	}
	env.configure(config)
	
	data = []
	try:
		while len(data) < total_expert_data:
			s = env.reset()
			done = False # 40s 이상 주행하거나 충돌이 발생하면 종료됨 (충돌 발생해서 종료되면 마지막 step은 삭제하기)

			# action input by keyboard
			t = threading.Thread(target=getAction)
			t.daemon = True
			t.start()
			while not done:
				# print("data length : ",len(data))
				action = env.action_type.actions_indexes["IDLE"]
				action_idx = all_action['IDLE']
				if action_key != None:
					action = env.action_type.actions_indexes[action_key]
					action_idx = all_action[action_key]
					action_key = None
				# print("action :",action_key)
				# state 와 action을 붙여서 저장 (dim = 75)
				action_list = [0.0]*5
				action_list[action_idx] = 1.0
				state = list(s.flatten())
				print(action_list)
				state.extend(action_list)
				data.append(state)

				s,r,done,info = env.step(action)
				env.render()
			data.pop()
		with open('expert_demo.p', 'wb') as f:
			pickle.dump(data, f)
	except KeyboardInterrupt:
		print("keyboard interrupt")
	

if __name__=='__main__':
	main()