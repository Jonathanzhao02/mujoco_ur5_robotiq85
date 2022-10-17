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

        while True:
            x = (np.random.rand(1) - 0.5) * 0.5
            y = (np.random.rand(1) - 0.5) * 0.25 + 0.7475 - 0.4
            too_close = False
            for j in range(len(point_list)):
                if l2(point_list[j], (x, y)) < 0.1 or abs(point_list[j][0] - x) < 0.1:
                    too_close = True
            if not too_close:
                point_list.append((x, y))
                interface.sim.data.qpos[-7 - i * 7] = x
                interface.sim.data.qpos[-6 - i * 7] = y
                break
    
    print(interface.sim.data.qpos)

def change_objective(recorder, action, targets):
    def __func():
        recorder.objective = {
            'action': action,
            'targets': targets
        }
    return __func

if __name__ == '__main__':

    import numpy as np
    from sequential_actions_interface import *
    from abr_control.controllers import Damping
    from my_osc import OSC
    from mujoco_interface import Mujoco
    from abr_control.utils import transformations
    from my_mujoco_config import MujocoConfig as arm
    from record import Recorder
    from parse_xml import parse_xml
    from pathlib import Path

    import matplotlib.pyplot as plt

    objs = [
        'bowl_3',
        'mug_3',
        'mug_2',
        'mug'
    ]
    
    gen_colors, gen_sizes, gen_scales = parse_xml(
        Path('my_models/ur5_robotiq85/ur5_tabletop_template.xml'),
        '__template',
        Path('my_models/ur5_robotiq85/ur5_tabletop.xml')
    )

    with Recorder(objs, {
            'color': {obj: gen_colors[obj] if obj in gen_colors.keys() else None for obj in objs},
            'size': {obj: gen_sizes[obj] if obj in gen_sizes.keys() else None for obj in objs},
            'scale': {obj: gen_scales[obj] if obj in gen_scales.keys() else None for obj in objs}
        },
        'demo.data',
        objective={
            'action': 'stack',
            'targets': {
                'obj1': 'bowl_3',
                'obj2': 'mug_3'
            }
        }
    ) as recorder:
        # create our Mujoco interface
        robot_config = arm('ur5_tabletop.xml', folder='./my_models/ur5_robotiq85')
        interface = Mujoco(robot_config, dt=0.008, on_step=recorder.record)
        interface.connect(joint_names=['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'finger_joint'], camera_id=0)
        random_place(interface, objs)
        
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

        from tasks import move, pick_up, push, place, rotate, rotate_place, stack, cover

        mug_scale = gen_scales['mug_mesh']
        mug_color3 = gen_colors['mug_3']
        mug_color2 = gen_colors['mug_2']
        mug_color1 = gen_colors['mug']

        bowl_scale = gen_scales['bowl_mesh']
        bowl_color = gen_colors['bowl_3']

        # e.append(PrintAction(interface, ctrlr, f'Stack the {mug_scale[0]} {mug_color3[0]} mug on the {bowl_scale[0]} {bowl_color[0]} bowl'))
        stack(e, interface, ctrlr, target_name='mug_3', container_name='bowl_3', pickup_dz=0.06, pickup_dx=0.04 * mug_scale[1], place_dz=0.1, place_dx=0.04 * mug_scale[1], theta=0, rot_time=0, grip_time=100, grip_force=0.12, terminator=False)
        move(e, interface, ctrlr, dz=0.1, terminator=False, on_finish=change_objective(recorder, 'cover', { 'bottom': 'mug_3', 'top': 'mug_2' }))
        # e.append(PrintAction(interface, ctrlr, f'Stack the {mug_scale[0]} {mug_color2[0]} mug on the {mug_scale[0]} {mug_color3[0]} mug'))
        stack(e, interface, ctrlr, target_name='mug_2', container_name='mug_3', pickup_dz=0.06, pickup_dx=0.04 * mug_scale[1], place_dz=0.15, place_dx=0.04 * mug_scale[1], theta=0, rot_time=0, grip_time=100, grip_force=0.12, terminator=False)
        move(e, interface, ctrlr, dz=0.1, terminator=False, on_finish=change_objective(recorder, 'cover', { 'bottom': 'mug_2', 'top': 'mug' }))
        # e.append(PrintAction(interface, ctrlr, f'Stack the {mug_scale[0]} {mug_color1[0]} mug on the {mug_scale[0]} {mug_color2[0]} mug'))
        stack(e, interface, ctrlr, target_name='mug', container_name='mug_2', pickup_dz=0.06, pickup_dx=0.04 * mug_scale[1], place_dz=0.15, place_dx=0.04 * mug_scale[1], theta=0, rot_time=0, grip_time=100, grip_force=0.12, terminator=False)
        move(e, interface, ctrlr, dz=0.1, terminator=True)

        e.execute()
