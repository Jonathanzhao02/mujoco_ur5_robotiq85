from sequential_actions_interface import *


if __name__ == '__main__':
    import math
    import numpy as np
    from abr_control.controllers import Damping
    from my_osc import OSC
    from mujoco_interface import Mujoco
    from abr_control.utils import transformations
    from my_mujoco_config import MujocoConfig as arm

    target_name = "target2"
    base_name = "base_link"

    def l2(pos1, pos2):
        assert len(pos1) == len(pos2)
        d = 0
        for i in range(len(pos1)):
            d = d + (pos1[i] - pos2[i]) ** 2
        d = d ** (1 / 2)
        return d

    def target_func_factory(x=None, y=None, z=None, obj_name=None, dx=0, dy=0, dz=0, rotated=False):
        assert obj_name is not None or (x is not None and y is not None and z is not None), 'At least one of obj_name or (x, y, z) must not be None'

        def _target_func(interface):
            if obj_name is not None:
                target_xyz = interface.get_xyz(obj_name)
                target_xyz[-1] += dz
                target_xyz[-2] += dy
                target_xyz[-3] += dx

                if x is not None:
                    target_xyz[-3] = x
                
                if y is not None:
                    target_xyz[-2] = y
                
                if z is not None:
                    target_xyz[-1] = z
            else:
                target_xyz = (x, y, z)
            
            rot = transformations.euler_from_quaternion(interface.get_orientation("EE"), "rxyz")
            
            if rotated:
                base_xyz = interface.get_xyz(base_name)
                target_xy = target_xyz[:-1]
                base_xy = base_xyz[:-1]

                vx = target_xy[0] - base_xy[0]
                vy = target_xy[1] - base_xy[1]
                
                if vy == 0:
                    if vx > 0:
                        theta = math.pi
                    else:
                        theta = 0
                else:
                    theta = math.atan(vx / vy) + math.pi / 2
            else:
                theta = rot[-1]
            
            target = np.hstack(
                [
                    target_xyz,
                    (3.14, 0, theta),
                ]
            )
            return target
        return _target_func

    target_func = target_func_factory(obj_name=target_name, dy=-0.15, rotated=True)
    target_func_2 = target_func_factory(obj_name=target_name, z=0.08, rotated=True)
    target_func_3 = target_func_factory(obj_name='EE', z=0.3, rotated=True)
    target_func_4 = target_func_factory(obj_name='container', z=0.3)
    target_func_5 = target_func_factory(obj_name='container', z=0.15)

    def gripper_control_func_factory(force=None):
        def _gripper_control_func(u, gripper):
            if force is not None:
                gripper.set_gripper_status(force)
            u[-1] = gripper.get_gripper_status()
            return u
        return _gripper_control_func
    
    gripper_idle_func = gripper_control_func_factory()
    gripper_open_func = gripper_control_func_factory(-0.2)
    gripper_close_func = gripper_control_func_factory(0.2)

    # create our Mujoco interface
    robot_config = arm('ur5_insertion.xml', folder='./my_models/ur5_robotiq85')
    interface = Mujoco(robot_config, dt=0.008)
    interface.connect(joint_names=['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'finger_joint'], camera_id=1)

    # Randomly place the target object
    def gen_target(interface):
        point_list = []

        for i in range(2):
            while True:
                x = (np.random.rand(1) - 0.5) * 0.7
                y = (np.random.rand(1) - 0.5) * 0.25 + 0.7475 - 0.3
                too_close = False
                for j in range(len(point_list)):
                    if l2(point_list[j], (x, y)) < 0.1 or abs(point_list[j][0] - x) < 0.1:
                        too_close = True
                if not too_close:
                    point_list.append((x, y))
                    interface.sim.data.qpos[-7 - i * 7] = x
                    interface.sim.data.qpos[-6 - i * 7] = y
                    break

    gen_target(interface)
    print(interface.sim.data.qpos)
    # exit()
    
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
    )

    e = Executor(interface, robot_config.START_ANGLES, -0.05)
    e.append(MoveTo(interface, ctrlr, target_func, gripper_idle_func, error_limit=0.02))
    e.append(MoveTo(interface, ctrlr, target_func_2, gripper_idle_func, time_limit=100))
    e.append(MoveTo(interface, ctrlr, target_func_2, gripper_close_func, time_limit=25))
    e.append(MoveTo(interface, ctrlr, target_func_3, gripper_idle_func, error_limit=0.02))
    e.append(MoveTo(interface, ctrlr, target_func_4, gripper_idle_func, error_limit=0.02))
    e.append(MoveTo(interface, ctrlr, target_func_5, gripper_idle_func, time_limit=50))
    e.append(MoveTo(interface, ctrlr, target_func_5, gripper_open_func, time_limit=10000))
    e.execute()
