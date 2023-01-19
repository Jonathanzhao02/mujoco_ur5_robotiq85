from environment.stack.stack import StackEnv
from gym.envs import registration
from gym.spaces import Dict, Box, Text, Discrete

import numpy as np

from abr_control.controllers import Damping
from utils.mujoco.my_osc import OSC

class StackTrajectoryEnv(StackEnv):
    def __init__(self, **kwargs):
        observation_space = Dict({
            "image": Box(low=0, high=255, shape=(224,224,3), dtype=np.uint8),
            "objective": Text(100),
            "dist_to_goal": Box(low=0, high=np.inf, dtype=np.float64),
        })

        StackEnv.__init__(self, observation_space=observation_space, **kwargs)
    
    def reset_model(self):
        ob = super().reset_model()
        ob['dist_to_goal'] = np.array(0, dtype=np.float64)

        # Controller creation
        self.damping = Damping(self.robot_config, kv=10)
        self.controller = OSC(
            self.robot_config,
            kp=200,
            null_controllers=[self.damping],
            vmax=[0.5, 0.5],  # [m/s, rad/s]
            # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
            ctrlr_dof=[True, True, True, True, True, True],
            orientation_algorithm=1,
        )

        return ob
    
    # Action: [x, y, z, roll, pitch, yaw, gripper force]
    def step(self, a):
        feedback = self.mujoco_interface.get_feedback()
        u = self.controller.generate(
            q=feedback['q'],
            dq=feedback['dq'],
            target=a[:-1],
        )
        u[-1] = a[-1]
        ob, reward, terminated, _ = StackEnv.step(self, u)
        ob['dist_to_goal'] = np.array(self.dist(a[:3]))
        return ob, reward, terminated, {}

registration.register(id='StackTrajectory-v0', entry_point=StackTrajectoryEnv)
