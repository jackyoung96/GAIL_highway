import gym
import highway_env
from matplotlib import pyplot as plt

env = gym.make("highway-v0")
config = {
	"observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted"
    },
	"manual_control": True
}
env.configure(config)
    
env.reset()
done = False
while not done:
	s,r,done,info = env.step(env.action_space.sample())
	env.render()
	print(s[0])