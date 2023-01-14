from environment.stack import StackEnv
from gym.envs import registration
from gym.spaces import Dict, Box, Text, Discrete

from mujocomy_mujoco_config import MujocoConfig
from abr_control.controllers import Damping
from mujocomy_osc import OSC

class StackTrajectoryEnv(StackEnv):
    def __init__(self, **kwargs):
        observation_space = Dict({
            "image": Box(low=0, high=255, shape=(224,224,3), dtype=np.uint8),
            "objective": Text(100),
            "within_goal": Discrete(2), # 0 = no, 1 = yes
        })

        StackEnv.__init__(self, observation_space=observation_space, **kwargs)

        robot_config = MujocoConfig(self.xml_name, folder='./my_models/ur5_robotiq85')

        damping = Damping(robot_config, kv=10)

        ctrlr = OSC(
            robot_config,
            kp=200,
            null_controllers=[damping],
            vmax=[0.5, 0.5],  # [m/s, rad/s]
            # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
            ctrlr_dof=[True, True, True, True, True, True],
            orientation_algorithm=1,
        )

        robot_config._connect(
            self.sim, self.joint_pos_addrs, self.joint_vel_addrs, self.joint_dyn_addrs
        )
    
    def step(self, a):
        StackEnv.step(self, a)

registration.register(id='StackTrajectory-v0', entry_pont=StackTrajectoryEnv)
