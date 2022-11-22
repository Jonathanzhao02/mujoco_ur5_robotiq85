import numpy as np

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

MUG_PICKUP_DX = 0.04
MUG_PICKUP_DZ = 0.065
BOWL_PICKUP_DX = 0.08
BOWL_PICKUP_DZ = 0.042

MUG_PLACE_DZ = 0.2
BOWL_PLACE_DZ = 0.2
PLATE_PLACE_DZ = 0.1

DIST_MAX = 0.75
AVG_CHANGE = 0.15

# Randomly place the target object
def random_place(interface, objs):

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
                interface.sim.data.qpos[-7 - i * 7] = x
                interface.sim.data.qpos[-6 - i * 7] = y
                break
            tries += 1
    
    print(interface.sim.data.qpos)

def change_objective(recorder, action, targets):
    def __func():
        recorder.objective = {
            'action': action,
            'targets': targets
        }
    return __func

def create_verifier(interface, selection):
    def _verify(recorder):
        f = recorder._f
        valid_demo = False

        if f.attrs['success']:
            objs = f['objs']
            start = f['pos'][0]
            end = f['pos'][f.attrs['final_timestep'] - 1]

            dpos = end - start
            dist = np.linalg.norm(dpos, axis=1)[1:] # exclude EE

            if np.max(dist) < DIST_MAX and np.sum(dist > AVG_CHANGE) <= 1:
                idx = -1
            
                for k in range(len(objs) - 1):
                    if dist[k] > AVG_CHANGE:
                        idx = k + 1
                
                if idx != -1:
                    sim_idx = -1
                    sim = np.inf

                    for k in range(1,len(objs)):
                        d = np.linalg.norm(end[k] - end[idx])
                        if k != idx and d < sim:
                            sim = d
                            sim_idx = k
                    
                    obj1 = bytes.decode(objs[idx])
                    obj2 = bytes.decode(objs[sim_idx])

                    objective = f['objectives']['0']['targets']

                    if objective.attrs['obj1'] == obj1 and objective.attrs['obj2'] == obj2:
                        if sim < 0.08:
                            rot_final = f['rot'][f.attrs['final_timestep'] - 1]
                            rot_initial = f['rot'][0]
                            rot_diff = np.linalg.norm(rot_final[idx] - rot_initial[idx]) + np.linalg.norm(rot_final[sim_idx] - rot_initial[sim_idx])

                            if rot_diff < 0.85:
                                valid_demo = True
        return valid_demo
    return _verify

class SamplingRecorder():
    def __init__(self, recorder, mod=1):
        self.recorder = recorder
        self.mod = mod
        self.step = 0
    
    def __call__(self, interface):
        if self.step % self.mod == 0:
            self.step += 1
            return self.recorder.record(interface)
        else:
            self.step += 1

if __name__ == '__main__':
    import mujoco_py

    import random
    from sequential_actions_interface import *
    from abr_control.controllers import Damping
    from my_osc import OSC
    from mujoco_interface import Mujoco
    from abr_control.utils import transformations
    from my_mujoco_config import MujocoConfig as arm
    from record import Recorder
    from parse_xml import parse_xml
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser(description="Generate a single demonstration")
    parser.add_argument("idx", help="Index of demonstration", type=int)

    args = parser.parse_args()

    i = args.idx

    combos = [
        ['bowl', 'plate'],
        ['mug', 'plate'],
        ['mug', 'bowl'],
        ['bowl', 'mug'],
    ]

    objs = [
        'bowl',
        'plate',
        'mug'
    ]

    sel = random.choice(combos)
    
    gen_colors, gen_sizes, gen_scales = parse_xml(
        Path('my_models/ur5_robotiq85/ur5_tabletop_template.xml'),
        '__template',
        Path('my_models/ur5_robotiq85/ur5_tabletop.xml')
    )

    recorder = None
    interface = None

    sentence = f'stack the {gen_scales[sel[0] + '_mesh']} {gen_colors[sel[0]]} {sel[0]} on the {gen_scales[sel[1] + '_mesh']} {gen_colors[sel[1]]} {sel[1]}'

    try:
        recorder = Recorder(objs, {
                'color': {obj: gen_colors[obj] if obj in gen_colors.keys() else None for obj in objs},
                'size': {obj: gen_sizes[obj] if obj in gen_sizes.keys() else None for obj in objs},
                'scale': {obj: gen_scales[obj + '_mesh'] if obj + '_mesh' in gen_scales.keys() else None for obj in objs}
            },
            f'demos/demo{i}.data',
            f'demos/demo{i}_imgs',
            objective={
                'action': 'stack',
                'targets': {
                    'obj1': sel[0],
                    'obj2': sel[1]
                }
            },
            max_timesteps=400,
        )
        # create our Mujoco interface
        robot_config = arm('ur5_tabletop.xml', folder='./my_models/ur5_robotiq85')
        interface = Mujoco(robot_config, dt=0.008, on_step=SamplingRecorder(recorder, 2))
        interface.connect(joint_names=['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'finger_joint'], camera_id=0)
        random_place(interface, objs)

        recorder.set_verifier(create_verifier(interface, sel))
        
        # damp the movements of the arm
        damping = Damping(robot_config, kv=10)
        # instantiate controller
        ctrlr = OSC(
            robot_config,
            kp=200,
            null_controllers=[damping],
            vmax=[0.5, 0.5],  # [m/s, rad/s]
            # control (x, y, z) out of [x, y, z, alpha, beta, gamma]
            ctrlr_dof=[True, True, True, True, True, True],
            orientation_algorithm=1,
        )

        e = Executor(interface, robot_config.START_ANGLES, -0.05)

        from tasks import move, pick_up, push, place, rotate, rotate_place, stack, cover, idle
    
        if sel[1] == 'mug':
            place_dz = MUG_PLACE_DZ
        elif sel[1] == 'plate':
            place_dz = PLATE_PLACE_DZ
        elif sel[1] == 'bowl':
            place_dz = BOWL_PLACE_DZ

        if sel[0] == 'mug':
            mug_scale1 = gen_scales['mug_mesh']
            mug_color1 = gen_colors['mug']
            stack(e, interface, ctrlr, target_name='mug', container_name=sel[1], pickup_dz=MUG_PICKUP_DZ, pickup_dx=MUG_PICKUP_DX * mug_scale1[1], place_dz=place_dz, place_dx=MUG_PICKUP_DX * mug_scale1[1], theta=0, rot_time=0, grip_time=100, grip_force=0.12, terminator=True)
        else:
            bowl_scale1 = gen_scales['bowl_mesh']
            bowl_color1 = gen_colors['bowl']
            stack(e, interface, ctrlr, target_name='bowl', container_name=sel[1], pickup_dz=BOWL_PICKUP_DZ / bowl_scale1[1], pickup_dx=BOWL_PICKUP_DX * bowl_scale1[1], place_dz=place_dz, place_dx=BOWL_PICKUP_DX * bowl_scale1[1], theta=0, rot_time=0, grip_time=100, grip_force=0.12, terminator=True)

        e.execute()
    finally:
        if recorder is not None:
            recorder.close()
        if interface is not None:
            interface.disconnect()
