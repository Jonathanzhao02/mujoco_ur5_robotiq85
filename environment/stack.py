import gym
from gym.envs import registration
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box, Text, Dict

from utils.xml.parse_xml import parse_xml
from utils.xml.tag_replacers import ColorTagReplacer, ScaleTagReplacer
from utils.mujoco.mujoco_interface import Mujoco
from utils.mujoco.my_mujoco_config import MujocoConfig

import mujoco as mj

import numpy as np
import random
import os
from pathlib import Path

obj_names = [
    'container',
    'target2',
    'bowl',
    'bowl_2',
    'bowl_3',
    'plate',
    'plate_2',
    'plate_3',
    'mug',
    'mug_2',
    'mug_3',
][::-1]

obj_idxes = {obj_names[i]: i for i in range(len(obj_names))}

combos = [
    ['bowl', 'plate'],
    ['mug', 'plate'],
    ['mug', 'bowl'],
    ['bowl', 'mug'],
]

def random_place(qpos, objs):
    qpos = qpos.copy()

    def l2(pos1, pos2):
        assert len(pos1) == len(pos2)
        d = 0
        for i in range(len(pos1)):
            d = d + (pos1[i] - pos2[i]) ** 2
        d = d ** (1 / 2)
        return d

    point_list = []

    for obj in objs:
        i = obj_idxes[obj]
        tries = 0

        while tries < 100000:
            x = (np.random.rand(1) - 0.5) * 0.5 # [-0.25, 0.25]
            y = (np.random.rand(1) - 0.5) * 0.25 + 0.7475 - 0.4 # [0.2225, 0.4725]
            too_close = False
            for j in range(len(point_list)):
                if l2(point_list[j], (x, y)) < 0.2 or abs(point_list[j][0] - x) < 0.1:
                    too_close = True
            if not too_close:
                point_list.append((x, y))
                qpos[-7 - i * 7] = x
                qpos[-6 - i * 7] = y
                break
            tries += 1

        if tries >= 100000:
            raise Exception("Tried to place too many times")
    
    return qpos

class StackEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array", # only use this mode for now
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(self, collisions=True, max_episode_steps=800, **kwargs):
        if not collisions:
            xml_template_path = 'my_models/ur5_robotiq85/ur5_tabletop_template_nocol.xml'
            xml_path = 'my_models/ur5_robotiq85/ur5_tabletop_nocol.xml'
            self.xml_name = 'ur5_tabletop_nocol.xml'
        else:
            xml_template_path = 'my_models/ur5_robotiq85/ur5_tabletop_template.xml'
            xml_path = 'my_models/ur5_robotiq85/ur5_tabletop.xml'
            self.xml_name = 'ur5_tabletop.xml'

        if 'observation_space' not in kwargs:
            kwargs['observation_space'] = Dict({
                "image": Box(low=0, high=255, shape=(224,224,3), dtype=np.uint8),
                "objective": Text(100),
            })

        # Randomize object attributes
        gen_tags = parse_xml(
            Path(os.getcwd()).joinpath(xml_template_path),
            '__template',
            Path(os.getcwd()).joinpath(xml_path),
            {
                'color': ColorTagReplacer(),
                'scale': ScaleTagReplacer(),
            }
        )
        gen_colors = gen_tags['color']
        gen_scales = gen_tags['scale']

        # Randomize selected objects for objective
        sel = random.choice(combos)

        # NOTE!!! Atribute randomization can only happen during initialization due to limitations of using an XML file to define Mujoco

        self.objective = f'stack the {gen_scales[sel[0] + "_mesh"][0]} {gen_colors[sel[0]][0]} {sel[0]} on the {gen_scales[sel[1] + "_mesh"][0]} {gen_colors[sel[1]][0]} {sel[1]}'
        self.colors = gen_colors
        self.scales = gen_scales
        self.max_episode_steps = max_episode_steps
        self.steps = 0

        MujocoEnv.__init__(
            self,
            str(Path(os.getcwd()).joinpath(xml_path)),
            1,
            **kwargs
        )

        # Define action space
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # Interface creation
        self.robot_config = MujocoConfig(self.model, self.data)
        self.mujoco_interface = Mujoco(self.robot_config)
        self.mujoco_interface.connect(['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'finger_joint'])

        # Set starting angles
        self.START_ANGLES = self.model.numeric("START_ANGLES").data
        self.init_qpos[self.mujoco_interface.joint_pos_addrs] = self.START_ANGLES
    
    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        self.steps += 1
        ob = self._get_obs()
        terminated = self.steps > self.max_episode_steps
        return ob, reward, terminated, {}

    def reset_model(self):
        # Randomize object positions
        qpos = random_place(self.init_qpos, [
            'bowl',
            'plate',
            'mug'
        ])
        self.set_state(qpos, self.init_qvel)

        return self._get_obs()

    def _get_obs(self):
        # self._get_viewer('rgb_array').render(224,224,camera_id=0)
        # data = self._get_viewer('rgb_array').read_pixels(224, 224, depth=False)
        # image = data[::-1, :, :]
        self._get_viewer('human').render()
        image = np.zeros((224, 224, 3))

        return {
            "image": image,
            "objective": self.objective,
        }

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.type = mj.mjtCamera.mjCAMERA_FIXED
        v.cam.fixedcamid = 0

registration.register(id='Stack-v0', entry_point=StackEnv)
