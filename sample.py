import numpy as np
import h5py
from PIL import Image
from abr_control.utils import transformations

import torch
import torch.nn as nn
import torch.optim as optim
import clip

import torchvision
from torchvision import transforms
from models.basic_model import Backbone
from models.film_model import Backbone as FiLMBackbone

from enum import Enum

Modes = Enum('Modes', ['REL', 'ABS'])

# Film model
# Returns trajectory of length 10
class TrajectoryModelV0():
    def __init__(self, ckpt, iters, n_points=10, ret_points=5):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.iters = iters
        self.n_points = n_points
        self.ret_points = ret_points
        self.sentence = None

        self.img_preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = FiLMBackbone(action_size=100).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

        if ckpt is not None:
            self.model.load_state_dict(torch.load(ckpt)['model'], strict=True)
        
    def write_objective(self, sentence):
        self.sentence = clip.tokenize(sentence).to(self.device)
    
    def sample(self, img, ee_pos, ee_rot, feedback):
        img = np.rot90(img, -1, (0,1))
        img = torch.tensor(img.T.copy()).float() / 255

        if self.img_preprocess:
            img = self.img_preprocess(img)

        img = torch.stack((img, img), axis=0).to(self.device)
        sentence = torch.cat((self.sentence, self.sentence), axis=0)
        
        # to calculate future pos, rot, gripper, add to current ee_pos, ee_rot, use gripper as force
        # transform sin/cos back to theta for ee_rot
        c_ee_pos = (torch.rand(size=(2,self.n_points,3)) - 0.5) * 0.2
        c_ee_rot = (torch.rand(size=(2,self.n_points,6)) - 0.5) * 2
        c_gripper = (torch.rand(size=(2,self.n_points,1)) - 0.5) * 2
        c_action = torch.cat((c_ee_pos, c_ee_rot, c_gripper), -1).float()
        c_action = c_action.to(self.device)
        c_action.requires_grad = True

        optimizer = optim.Adam([c_action], lr=1e-2)

        for i in range(self.iters):
            optimizer.zero_grad()
            y_hat = self.model(img, sentence, c_action)
            y_hat.sum().backward()
            optimizer.step()

        print(y_hat[0].item(), y_hat[1].item())

        if y_hat[0].item() < y_hat[1].item():
            c_action = c_action[0].detach().squeeze(0).cpu().numpy()
        else:
            c_action = c_action[1].detach().squeeze(0).cpu().numpy()

        dpos = c_action[:,:3]
        drot = c_action[:,3:9]
        grip = c_action[:,9]

        # ordered cos, sin
        drotx = np.arctan2(drot[:,1], drot[:,0])
        droty = np.arctan2(drot[:,3], drot[:,2])
        drotz = np.arctan2(drot[:,5], drot[:,4])
        drot = np.stack([drotx, droty, drotz], axis=-1)

        return dpos[:self.ret_points], drot[:self.ret_points], grip[:self.ret_points]

class ModelV1():
    def __init__(self, ckpt, iters):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.iters = iters
        self.sentence = None

        self.img_preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = Backbone(action_size=10).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

        if ckpt is not None:
            self.model.load_state_dict(torch.load(ckpt)['model'], strict=True)
        
    def write_objective(self, sentence):
        self.sentence = clip.tokenize(sentence).to(self.device)
    
    def sample(self, img, ee_pos, ee_rot, feedback):
        img = np.rot90(img, -1, (0,1))
        img = torch.tensor(img.T.copy()).float() / 255

        if self.img_preprocess:
            img = self.img_preprocess(img)

        img = img.unsqueeze(0).to(self.device)
        
        # to calculate future pos, rot, gripper, add to current ee_pos, ee_rot, use gripper as force
        # transform sin/cos back to theta for ee_rot
        c_ee_pos = (torch.rand(size=(1,3)) - 0.5) * 0.2
        c_ee_rot = (torch.rand(size=(1,6)) - 0.5) * 2
        c_gripper = (torch.rand(size=(1,1)) - 0.5) * 2
        c_action = torch.cat((c_ee_pos, c_ee_rot, c_gripper), -1).float()
        c_action = c_action.to(self.device)
        c_action.requires_grad = True

        optimizer = optim.Adam([c_action], lr=1e-2)
        y = torch.ones((1,1)).float().to(self.device)

        for i in range(self.iters):
            optimizer.zero_grad()
            y_hat = self.model(img, self.sentence, c_action)
            y_hat.backward()
            optimizer.step()

        print(y_hat[0].item())
        c_action = c_action.detach().squeeze(0).cpu().numpy()

        dpos = c_action[:3]
        drot = c_action[3:9]
        dgrip = c_action[9]

        drot = np.array([np.arctan2(drot[1], drot[0]), np.arctan2(drot[3], drot[2]), np.arctan2(drot[5], drot[4])]) # ordered cos, sin

        target = np.hstack(
            [
                ee_pos + dpos,
                ee_rot + drot,
            ]
        )

        return target, dgrip

# version 0
# has reversed offsets, relative angle gripper, progress
class ModelV0():
    def __init__(self, ckpt, iters):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.iters = iters
        self.sentence = None

        self.img_preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = Backbone(action_size=12).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

        if ckpt is not None:
            self.model.load_state_dict(torch.load(ckpt)['model'], strict=True)
        
    def write_objective(self, sentence):
        self.sentence = clip.tokenize(sentence).to(self.device)
    
    def sample(self, img, ee_pos, ee_rot, feedback):
        img = np.rot90(img, -1, (0,1))
        img = torch.tensor(img.T.copy()).float() / 255

        if self.img_preprocess:
            img = self.img_preprocess(img)

        img = img.unsqueeze(0).to(self.device)
        
        # to calculate future pos, rot, gripper, add to current ee_pos, ee_rot, gripper
        # transform sin/cos back to theta for ee_rot, gripper
        c_ee_pos = (torch.rand(size=(1,3)) - 0.5) * 0.2
        c_ee_rot = (torch.rand(size=(1,6)) - 0.5) * 2
        c_gripper = (torch.rand(size=(1,2)) - 0.5) * 2
        c_progress = torch.rand(size=(1,1))
        c_action = torch.cat((c_ee_pos, c_ee_rot, c_gripper, c_progress), -1).float()
        c_action = c_action.to(self.device)
        c_action.requires_grad = True

        optimizer = optim.Adam([c_action], lr=1e-2)
        y = torch.ones((1,1)).float().to(self.device)

        for i in range(self.iters):
            optimizer.zero_grad()
            y_hat = self.model(img, self.sentence, c_action)
            y_hat.backward()
            optimizer.step()

        print(y_hat[0].item())
        c_action = c_action.detach().squeeze(0).cpu().numpy()

        dpos = c_action[:3]
        drot = c_action[3:9]
        dgrip = c_action[9:11]
        prog = c_action[11:12]

        drot = np.array([np.arctan2(drot[1], drot[0]), np.arctan2(drot[3], drot[2]), np.arctan2(drot[5], drot[4])]) # ordered cos, sin
        dgrip = np.arctan2(dgrip[0], dgrip[1]) # ordered sin, cos

        target = np.hstack(
            [
                ee_pos - dpos,
                ee_rot - drot,
            ]
        )

        return target, dgrip

class TrajectorySampler():
    def __init__(self, controller, interface, start_angles, model, dof=7, sentence=None, width=224, height=224, max_timesteps=800, error_limit=0.01, mode=Modes.ABS):
        self.controller = controller
        self.dof = dof
        self.width = width
        self.height = height
        self.max_timesteps = max_timesteps
        self.error_limit = error_limit
        self.mode = mode
        self.success = True
        self.cnt = 0
        self.model = model

        interface.send_target_angles(start_angles)
        self.model.write_objective(sentence)
        print(sentence)
    
    # MODIFY TO ACCOUNT FOR NOW-CORRECT OFFSETS + GRIPPER
    def sample(self, interface):
        if self.cnt < self.max_timesteps:
            feedback = interface.get_feedback()
            q = feedback['q']

            img = interface.sim.render(self.width,self.height,camera_name='111').astype('<B')
            while img.sum() == 0:
                img = interface.sim.render(self.width,self.height,camera_name='111').astype('<B')

            ee_pos = interface.get_xyz('EE')
            ee_rot = transformations.euler_from_quaternion(interface.get_orientation('EE'), "rxyz")

            dpos, drot, gripper = self.model.sample(img, ee_pos, ee_rot, feedback)

            if self.mode == Modes.ABS:
                ee_poses = np.cumsum(dpos, axis=0) + ee_pos
                ee_rots = np.cumsum(drot, axis=0) + ee_rot

            for i in range(len(gripper)):
                if self.mode == Modes.REL:
                    ee_pos = interface.get_xyz('EE')
                    ee_rot = transformations.euler_from_quaternion(interface.get_orientation('EE'), "rxyz")

                    target = np.hstack([
                        ee_pos + dpos[i],
                        ee_rot + drot[i]
                    ])
                elif self.mode == Modes.ABS:
                    target = np.hstack([
                        ee_poses[i],
                        ee_rots[i]
                    ])

                while np.linalg.norm(target[:3] - ee_pos) > self.error_limit:
                    feedback = interface.get_feedback()
                    u = self.controller.generate(
                        q=feedback['q'],
                        dq=feedback['dq'],
                        target=target,
                    )
                    u[-1] = gripper[i]
                    interface.send_forces(u, update_display=False)

                    print(f'Finished step {self.cnt}')

                    self.cnt += 1
        else:
            raise RuntimeError(f'Exceeded max time {self.max_timesteps}')
        
    def write_objective(self, sentence):
        self.model.write_objective(sentence)

class Sampler():
    def __init__(self, controller, interface, start_angles, model, dof=7, sentence=None, width=224, height=224, max_timesteps=800, dt=1):
        self.controller = controller
        self.dof = dof
        self.width = width
        self.height = height
        self.max_timesteps = max_timesteps
        self.success = True
        self.cnt = 0
        self.dt = dt
        self.model = model

        interface.send_target_angles(start_angles)
        self.model.write_objective(sentence)
        print(sentence)
    
    # MODIFY TO ACCOUNT FOR NOW-CORRECT OFFSETS + GRIPPER
    def sample(self, interface):
        if self.cnt < self.max_timesteps:
            feedback = interface.get_feedback()
            q = feedback['q']

            img = interface.sim.render(self.width,self.height,camera_name='111').astype('<B')
            while img.sum() == 0:
                img = interface.sim.render(self.width,self.height,camera_name='111').astype('<B')

            ee_pos = interface.get_xyz('EE')
            ee_rot = transformations.euler_from_quaternion(interface.get_orientation('EE'), "rxyz")

            target, gripper = self.model.sample(img, ee_pos, ee_rot, feedback)

            for _ in range(self.dt):
                feedback = interface.get_feedback()
                u = self.controller.generate(
                    q=feedback['q'],
                    dq=feedback['dq'],
                    target=target,
                )
                u[-1] = gripper
                interface.send_forces(u, update_display=False)

            print(f'Finished step {self.cnt}')

            self.cnt += self.dt
        else:
            raise RuntimeError(f'Exceeded max time {self.max_timesteps}')
        
    def write_objective(self, sentence):
        self.model.write_objective(sentence)
