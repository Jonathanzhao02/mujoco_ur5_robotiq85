from pathlib import Path
from PIL import Image
import numpy as np
import h5py
from abr_control.utils import transformations

targets_type = np.dtype([('obj1', 'S64'), ('obj2', 'S64'), ('pos', '<f4', 3), ('rot', '<f4', 3)])
objective_type = np.dtype([('timestep', 'uint'), ('action', 'S64'), ('targets', targets_type)])

class Recorder():
    def __init__(self, obj_names, gen_attrs, out_fname, out_folder_name, dof=7, objective=None, width=224, height=224, max_timesteps=800, verifier=lambda x: True):
        self.obj_names = ['EE'] + obj_names
        self.gen_attrs = gen_attrs
        self.out_fname = out_fname
        self.dof = dof
        self.width = width
        self.height = height
        self.max_timesteps = max_timesteps

        Path(out_fname).parent.mkdir(parents=True, exist_ok=True)

        self.out_folder = Path(out_folder_name)
        self.out_folder.mkdir(parents=True, exist_ok=True)

        self._f = h5py.File(self.out_fname, 'w')

        if self._f:
            grp = self._f.create_group('gen_attrs')

            for attr in gen_attrs.keys():
                attr_grp = grp.create_group(attr)

                for obj_name,val in gen_attrs[attr].items():
                    if val is not None:
                        attr_grp.create_dataset(obj_name, data=val)

            self.objective_data = self._f.create_dataset('objective', shape=(0), maxshape=(max_timesteps), dtype=objective_type, chunks=True)
            self.q_data = self._f.create_dataset('q', shape=(max_timesteps, dof), dtype='<f4')
            self.dq_data = self._f.create_dataset('dq', shape=(max_timesteps, dof), dtype='<f4')
            # self.img_data = self._f.create_dataset('img', shape=(max_timesteps, width, height, 3), dtype='<B')
            self.pos_data = self._f.create_dataset('pos', shape=(max_timesteps, len(self.obj_names), 3), dtype='<f4')
            self.rot_data = self._f.create_dataset('rot', shape=(max_timesteps, len(self.obj_names), 3), dtype='<f4')
            self._f.attrs['success'] = True

            self.cnt = 0
            self.ocnt = 0
            self.verify = verifier

            if objective is not None:
                self.write_objective(objective)
        else:
            raise Exception('Failed to open h5py file')
        
    def set_verifier(self, verifier):
        self.verify = verifier
    
    def record(self, interface):
        if self.cnt < self.max_timesteps:
            feedback = interface.get_feedback()
            self.q_data[self.cnt] = feedback['q']
            self.dq_data[self.cnt] = feedback['dq']

            img = interface.sim.render(self.width,self.height,camera_name='111').astype('<B')
            while img.sum() == 0:
                img = interface.sim.render(self.width,self.height,camera_name='111').astype('<B')
            # self.img_data[self.cnt] = img
            img = np.flip(img, 0)
            Image.fromarray(img).save(str(self.out_folder.joinpath(f'{self.cnt}.png')))

            pos = np.empty((len(self.obj_names), 3), dtype='<f4')
            rot = np.empty((len(self.obj_names), 3), dtype='<f4')
            for i in range(len(self.obj_names)):
                pos[i] = interface.get_xyz(self.obj_names[i])
                rot[i] = transformations.euler_from_quaternion(interface.get_orientation(self.obj_names[i]), "rxyz")
            self.pos_data[self.cnt] = pos
            self.rot_data[self.cnt] = rot

            self.cnt += 1
        else:
            self._f.attrs['success'] = False
            raise RuntimeError(f'Exceeded max time {self.max_timesteps}')
        
    def write_objective(self, objective):
        self.objective_data.resize((self.ocnt + 1,))
        self.objective_data[self.ocnt]['timestep'] = self.cnt
        self.objective_data[self.ocnt]['action'] = objective['action']

        targets = objective['targets']

        if 'obj1' in targets:
            self.objective_data[self.ocnt]['targets']['obj1'] = targets['obj1']
        if 'obj2' in targets:
            self.objective_data[self.ocnt]['targets']['obj2'] = targets['obj2']
        if 'pos' in targets:
            self.objective_data[self.ocnt]['targets']['pos'] = targets['pos']
        if 'rot' in targets:
            self.objective_data[self.ocnt]['targets']['rot'] = targets['rot']

        self.ocnt += 1
    
    def __enter__(self):
        return self

    def close(self):
        try:
            self._f.attrs['final_timestep'] = self.cnt
            self._f.attrs['success'] = self.verify() and self.cnt < self.max_timesteps
        except:
            pass
        
        self._f.close()
        return False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
