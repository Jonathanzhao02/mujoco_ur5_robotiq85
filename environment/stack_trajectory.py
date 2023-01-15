from environment.stack import StackEnv
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
            "within_goal": Discrete(2), # 0 = no, 1 = yes
        })

        StackEnv.__init__(self, observation_space=observation_space, **kwargs)

        damping = Damping(self.robot_config, kv=10)

        self.controller = OSC(
            self.robot_config,
            kp=200,
            null_controllers=[damping],
            vmax=[0.5, 0.5],  # [m/s, rad/s]
            # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
            ctrlr_dof=[True, True, True, True, True, True],
            orientation_algorithm=1,
        )
    
    def reset_model(self):
        ob = StackEnv.reset_model(self)
        ob['within_goal'] = 0
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
        ob['within_goal'] = np.linalg.norm(a[:3])
        return ob, reward, terminated, {}

registration.register(id='StackTrajectory-v0', entry_point=StackTrajectoryEnv)
