from environment.stack import stack
from environment.stack_trajectory import stack_trajectory
from environment.particle import particle
import gym

from PIL import Image

import matplotlib.pyplot as plt

import random

env = gym.make('StackTrajectory-v0', render_mode='rgb_array')
env.reset()

for _ in range(10):
    for _ in range(20):
        obs, _, _, _ = env.step([0.3, 0.3, 0.3, -1.57, 0, -1.57, -0.12])
    
    env.reset()
