import numpy as np
import h5py
from abr_control.utils import transformations

import torch
import torch.nn as nn
import torch.optim as optim
import clip

from torchvision import transforms
from models.basic_model import Backbone

class Sampler():
    def __init__(self, controller, iters=50, dof=7, sentence=None, width=224, height=224, max_timesteps=800, verifier=lambda x: True):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.controller = controller
        self.dof = dof
        self.width = width
        self.height = height
        self.max_timesteps = max_timesteps
        self.sentence = clip.tokenize(sentence).to(self.device)
        self.verify = verifier
        self.success = True
        self.iters = iters

        self.img_preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = Backbone().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def set_verifier(self, verifier):
        self.verify = verifier
    
    def sample(self, interface):
        if self.cnt < self.max_timesteps:
            feedback = interface.get_feedback()

            img = interface.sim.render(self.width,self.height,camera_name='111').astype('<B')
            while img.sum() == 0:
                img = interface.sim.render(self.width,self.height,camera_name='111').astype('<B')
            img = np.flip(img, 0)
            img = torch.tensor((img / 255).astype('<f4'))

            if self.img_preprocess:
                img = self.img_preprocess(img)

            ee_pos = interface.get_xyz('EE')
            ee_rot = transformations.euler_from_quaternion(interface.get_orientation('EE'), "rxyz")
            gripper = q[-1]

            # to calculate future pos, rot, gripper, add to current ee_pos, ee_rot, gripper
            # transform sin/cos back to theta for ee_rot, gripper
            c_ee_pos = (torch.rand(size=(1,3) - 0.5) * 0.2
            c_ee_rot = (torch.rand(size=(1,6)) - 0.5) * 2
            c_gripper = (torch.rand(size=(1,2)) - 0.5) * 2
            c_progress = torch.rand(size=(1,1))
            c_action = torch.cat((c_ee_pos, c_ee_rot, c_gripper, c_progress), -1).float().to(self.device)
            c_action.requires_grad = True

            optimizer = optim.Adam([c_action], lr=1e-3)
            y = torch.zeros((1,1)).to(self.device)

            for i in range(self.iters):
                optimizer.zero_grad()
                y_hat = self.model(img, self.sentence, c_action)
                loss = self.criterion(y_hat, y)
                loss.backward()
                optimizer.step()
            
            c_action = c_action.squeeze(0).cpu().numpy()

            dpos = c_action[:3]
            drot = c_action[3:9]
            dgrip = c_action[9:11]
            prog = c_action[11:12]

            drot = np.array([np.arctan2(drot[1], drot[0]), np.arctan2(drot[3], drot[2]), np.arctan2(drot[5], drot[4])]) # ordered cos, sin
            dgrip = np.arctan2(dgrip[0], dgrip[1]) # ordered sin, cos

            target = np.hstack(
                [
                    ee_pos + dpos,
                    ee_rot + drot,
                ]
            )

            u = self.controller.generate(
                q=feedback['q'],
                dq=feedback['dq'],
                target=target,
            )
            u[-1] = gripper + dgrip
            interface.send_forces(u, update_display=True)
            print(prog)

            self.cnt += 1
        else:
            self.success = False
            raise RuntimeError(f'Exceeded max time {self.max_timesteps}')
        
    def write_objective(self, sentence):
        self.sentence = clip.tokenize(sentence).to(self.device)
    
    def __enter__(self):
        return self

    def close(self):
        self.success = self.verify(self) and self.cnt < self.max_timesteps
        return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
