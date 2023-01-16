from environment import stack, stack_trajectory
import gym

from PIL import Image

import matplotlib.pyplot as plt

import random

env = gym.make('StackTrajectory-v0', render_mode='human')
env.reset()

for _ in range(10):
    for _ in range(20):
        obs, _, _, _ = env.step([0.3, 0.3, 0.3, -1.57, 0, -1.57, -0.12])
        print(obs['image'].sum())
    
    env.reset()
