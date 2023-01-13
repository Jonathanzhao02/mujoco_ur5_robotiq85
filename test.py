from environment import stack, stack_trajectory
import gym

from PIL import Image

import matplotlib.pyplot as plt

import random

env = gym.make('StackTrajectory-v0', render_mode='rgb_array')
env.reset()

for _ in range(800):
    obs, _, _, _ = env.step([random.random() * 3 for _ in range(7)])
    print(obs['image'].sum())
