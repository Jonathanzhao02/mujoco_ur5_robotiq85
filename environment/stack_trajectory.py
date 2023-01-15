from environment.stack import StackEnv
from gym.envs import registration
from gym.spaces import Dict, Box, Text, Discrete

import numpy as np
import mujoco as mj

from utils.mujoco.my_mujoco_config import MujocoConfig
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

        self.joint_names=['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'finger_joint']
        self.joints = [self.model.jnt(name) for name in self.joint_names]
        # Assumes all hinge joints (1 dof)
        self.joint_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.joint_pos_addrs = [joint.qposadr for joint in self.joints]
        self.joint_vel_addrs = [joint.dofadr for joint in self.joints]

        # Need to also get the joint rows of the Jacobian, inertia matrix, and
        # gravity vector. This is trickier because if there's a quaternion in
        # the joint (e.g. a free joint or a ball joint) then the joint position
        # address will be different than the joint Jacobian row. This is because
        # the quaternion joint will have a 4D position and a 3D derivative. So
        # we go through all the joints, and find out what type they are, then
        # calculate the Jacobian position based on their order and type.
        index = 0
        self.joint_dyn_addrs = []
        for ii, joint_type in enumerate(self.model.jnt_type):
            if ii in self.joint_ids:
                self.joint_dyn_addrs.append(index)
            if joint_type == mj.mjtJoint.mjJNT_FREE:  # free joint
                # self.joint_dyn_addrs += [jj + index for jj in range(1, 6)]
                # index += 6  # derivative has 6 dimensions
                continue
            elif joint_type == mj.mjtJoint.mjJNT_BALL:  # ball joint
                self.joint_dyn_addrs += [jj + index for jj in range(1, 3)]
                index += 3  # derivative has 3 dimension
            else:  # slide or hinge joint
                index += 1  # derivative has 1 dimensions
        
        robot_config = MujocoConfig(self.model, self.data, self.joint_pos_addrs, self.joint_vel_addrs, self.joint_dyn_addrs)
        damping = Damping(robot_config, kv=10)

        self.controller = OSC(
            robot_config,
            kp=200,
            null_controllers=[damping],
            vmax=[0.5, 0.5],  # [m/s, rad/s]
            # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
            ctrlr_dof=[True, True, True, True, True, True],
            orientation_algorithm=1,
        )
    
    def get_feedback(self):
        """Return a dictionary of information needed by the controller.

        Returns the joint angles and joint velocities in [rad] and [rad/sec],
        respectively
        """

        q = np.copy(self.data.qpos[self.joint_pos_addrs])
        dq = np.copy(self.data.qvel[self.joint_vel_addrs])

        return {"q": q, "dq": dq}
    
    def reset_model(self):
        ob = StackEnv.reset_model(self)
        ob['within_goal'] = 0
        return ob
    
    # Action: [x, y, z, roll, pitch, yaw, gripper force]
    def step(self, a):
        feedback = self.get_feedback()
        u = self.controller.generate(
            q=feedback['q'],
            dq=feedback['dq'],
            target=a[:-1],
        )
        u[-1] = a[-1]
        ob, reward, terminated, _ = StackEnv.step(self, u)
        return ob, reward, terminated, {}

registration.register(id='StackTrajectory-v0', entry_point=StackTrajectoryEnv)
