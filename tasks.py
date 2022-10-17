from sequential_actions_interface import *
from abr_control.utils import transformations
import math

ROT_TIME = 400
GRIP_FORCE = 0.2
GRIP_TIME = 25
TERMINATOR_TIME = 100

class StaticPosition:
    def __init__(self, interface, obj_name):
        self.interface = interface
        self.obj_name = obj_name
        self.pos = None
    
    def get_pos(self):
        if self.pos is None:
            self.pos = self.interface.get_xyz(self.obj_name)

        return self.pos

class DynamicPosition:
    def __init__(self, interface, obj_name):
        self.interface = interface
        self.obj_name = obj_name
    
    def get_pos(self):
        return self.interface.get_xyz(self.obj_name)

def target_func_factory(interface, x=None, y=None, z=None, obj_pos=None, dx=0, dy=0, dz=0, rx=None, ry=None, rz=None):
    assert obj_pos is not None or (x is not None and y is not None and z is not None), 'At least one of obj_name or (x, y, z) must not be None'

    def _target_func(interface):
        if obj_pos is not None:
            target_xyz = obj_pos.get_pos()
            target_xyz = [
                target_xyz[0] + dx,
                target_xyz[1] + dy,
                target_xyz[2] + dz,
            ]

            if x is not None:
                target_xyz[-3] = x
            
            if y is not None:
                target_xyz[-2] = y
            
            if z is not None:
                target_xyz[-1] = z
        else:
            target_xyz = (x + dx, y + dy, z + dz)
        
        rot = transformations.euler_from_quaternion(interface.get_orientation("EE"), "rxyz")

        target = np.hstack(
            [
                target_xyz,
                (rx if rx is not None else rot[0]), (ry if ry is not None else rot[1]), (rz if rz is not None else rot[2]),
            ]
        )
        return target
    return _target_func

def gripper_control_func_factory(force=None):
    def _gripper_control_func(u, gripper):
        if force is not None:
            gripper.set_gripper_status(force)
        u[-1] = gripper.get_gripper_status()
        return u
    return _gripper_control_func

def move(executor, interface, controller, dx=0, dy=0, dz=0, time_limit=None, terminator=False, on_finish=None):
    target_func = target_func_factory(interface, obj_pos=StaticPosition(interface, 'EE'), dx=dx, dy=dy, dz=dz, rx=-1.57, ry=0, rz=-1.57)
    gripper_idle_func = gripper_control_func_factory()

    if terminator:
        executor.append(MoveTo(interface, controller, target_func, gripper_idle_func, time_limit=TERMINATOR_TIME, on_finish=on_finish))
    elif time_limit is None:
        executor.append(MoveTo(interface, controller, target_func, gripper_idle_func, error_limit=0.02, on_finish=on_finish))
    else:
        executor.append(MoveTo(interface, controller, target_func, gripper_idle_func, time_limit=time_limit, on_finish=on_finish))

def idle(executor, interface, controller, time=100, terminator=False, on_finish=None):
    move(executor, interface, controller, time_limit=time, terminator=terminator, on_finish=on_finish)

def push(executor, interface, controller, target_name, theta=0, grip_force=GRIP_FORCE, terminator=False, on_finish=None):
    target_func = target_func_factory(interface, obj_pos=DynamicPosition(interface, target_name), dz=0.15, dx=math.sin(theta) / 8, dy=-math.cos(theta) / 8, rx=-1.57, ry=0, rz=-1.57)
    target_func_2 = target_func_factory(interface, obj_pos=DynamicPosition(interface, target_name), z=0.08, dx=math.sin(theta) / 8, dy=-math.cos(theta) / 8, rx=-1.57, ry=0, rz=-1.57)
    target_func_3 = target_func_factory(interface, obj_pos=StaticPosition(interface, target_name), z=0.08, dx=-math.sin(theta) / 8, dy=math.cos(theta) / 8, rx=-1.57, ry=0, rz=-1.57)
    target_func_4 = target_func_factory(interface, obj_pos=StaticPosition(interface, 'EE'), z=0.3, rx=-1.57, ry=0, rz=-1.57)

    gripper_idle_func = gripper_control_func_factory()
    gripper_open_func = gripper_control_func_factory(-grip_force)
    gripper_close_func = gripper_control_func_factory(grip_force)

    executor.append(MoveTo(interface, controller, target_func, gripper_close_func, error_limit=0.02))
    executor.append(MoveTo(interface, controller, target_func_2, gripper_idle_func, error_limit=0.02))
    executor.append(MoveTo(interface, controller, target_func_3, gripper_idle_func, error_limit=0.02))

    if terminator:
        executor.append(MoveTo(interface, controller, target_func_4, gripper_idle_func, time_limit=TERMINATOR_TIME, on_finish=on_finish))
    else:
        executor.append(MoveTo(interface, controller, target_func_4, gripper_idle_func, error_limit=0.02, on_finish=on_finish))

def pick_up(executor, interface, controller, target_name, theta=0, dx=0, dy=0, dz=0, rot_time=ROT_TIME, grip_time=GRIP_TIME, grip_force=GRIP_FORCE, terminator=False, on_finish=None):
    target_func = target_func_factory(interface, obj_pos=DynamicPosition(interface, target_name), dx=dx, dy=dy, dz=0.15+dz, rx=-1.57, ry=0, rz=-1.57)
    target_func_2 = target_func_factory(interface, obj_pos=DynamicPosition(interface, target_name), dx=dx, dy=dy, dz=dz, rx=-1.57, ry=0, rz=-1.57)
    target_func_3 = target_func_factory(interface, obj_pos=StaticPosition(interface, 'EE'), z=0.3, rx=-1.57, ry=0, rz=-1.57)
    target_func_4 = target_func_factory(interface, obj_pos=StaticPosition(interface, 'EE'), z=0.3, rx=-1.57, ry=theta, rz=-1.57)

    gripper_idle_func = gripper_control_func_factory()
    gripper_open_func = gripper_control_func_factory(-grip_force)
    gripper_close_func = gripper_control_func_factory(grip_force)

    executor.append(MoveTo(interface, controller, target_func, gripper_open_func, error_limit=0.02))
    executor.append(MoveTo(interface, controller, target_func_2, gripper_idle_func, error_limit=0.02))
    executor.append(MoveTo(interface, controller, target_func_2, gripper_close_func, time_limit=grip_time))
    executor.append(MoveTo(interface, controller, target_func_3, gripper_idle_func, error_limit=0.02))

    if terminator:
        executor.append(MoveTo(interface, controller, target_func_4, gripper_idle_func, time_limit=TERMINATOR_TIME, on_finish=on_finish))
    else:
        executor.append(MoveTo(interface, controller, target_func_4, gripper_idle_func, time_limit=rot_time, on_finish=on_finish))

def place(executor, interface, controller, target_name=None, target_pos=None, dx=0, dy=0, dz=0, grip_time=GRIP_TIME, grip_force=GRIP_FORCE, terminator=False, on_finish=None):
    assert target_name is not None or target_pos is not None

    if target_name is None:
        pos_args = {
            'x': target_pos[0],
            'y': target_pos[1],
            'z': target_pos[2],
        }
    else:
        pos_args = {
            'obj_pos': DynamicPosition(interface, target_name)
        }

    target_func = target_func_factory(interface, **pos_args, dx=dx, dy=dy, dz=0.15+dz, rx=-1.57, rz=-1.57)
    target_func_2 = target_func_factory(interface, **pos_args, dx=dx, dy=dy, dz=dz, rx=-1.57, rz=-1.57)
    target_func_3 = target_func_factory(interface, **pos_args, dx=dx, dy=dy, dz=0.15, rx=-1.57, rz=-1.57)

    gripper_idle_func = gripper_control_func_factory()
    gripper_open_func = gripper_control_func_factory(-grip_force)
    gripper_close_func = gripper_control_func_factory(grip_force)

    executor.append(MoveTo(interface, controller, target_func, gripper_close_func, error_limit=0.02))
    executor.append(MoveTo(interface, controller, target_func_2, gripper_idle_func, error_limit=0.02))
    executor.append(MoveTo(interface, controller, target_func_2, gripper_open_func, time_limit=grip_time))

    if terminator:
        executor.append(MoveTo(interface, controller, target_func_3, gripper_idle_func, time_limit=TERMINATOR_TIME, on_finish=on_finish))
    else:
        executor.append(MoveTo(interface, controller, target_func_3, gripper_idle_func, error_limit=0.02, on_finish=on_finish))

def rotate(executor, interface, controller, theta=0, rot_time=ROT_TIME, terminator=False, on_finish=None):
    target_func = target_func_factory(interface, obj_pos=StaticPosition(interface, 'EE'), rx=-1.57, ry=theta, rz=-1.57)
    gripper_idle_func = gripper_control_func_factory()

    if terminator:
        executor.append(MoveTo(interface, controller, target_func, gripper_idle_func, time_limit=TERMINATOR_TIME, on_finish=on_finish))
    else:
        executor.append(MoveTo(interface, controller, target_func, gripper_idle_func, time_limit=rot_time, on_finish=on_finish))

def rotate_place(executor, interface, controller, target_name=None, target_pos=None, dx=0, dy=0, dz=0, theta=0, rot_time=ROT_TIME, grip_time=GRIP_TIME, grip_force=GRIP_FORCE, terminator=False, on_finish=None):
    rotate(executor, interface, controller, theta, rot_time, False)
    place(executor, interface, controller, target_name, target_pos, dx, dy, dz, grip_time, grip_force, terminator, on_finish)

def stack(executor, interface, controller, target_name=None, container_name=None, pickup_dx=0, pickup_dy=0, pickup_dz=0, place_dx=0, place_dy=0, place_dz=0, theta=0, rot_time=ROT_TIME, grip_time=GRIP_TIME, grip_force=GRIP_FORCE, terminator=False, on_finish=None):
    pick_up(executor, interface, controller, target_name, theta, pickup_dx, pickup_dy, pickup_dz, rot_time, grip_time, grip_force, False)
    rotate_place(executor, interface, controller, container_name, dx=place_dx, dy=place_dy, dz=place_dz, theta=theta, rot_time=rot_time, grip_time=grip_time, grip_force=grip_force, terminator=terminator, on_finish=on_finish)

def cover(executor, interface, controller, target_name=None, container_name=None, pickup_dx=0, pickup_dy=0, pickup_dz=0, place_dx=0, place_dy=0, place_dz=0, theta=0, rot_time=ROT_TIME, grip_time=GRIP_TIME, grip_force=GRIP_FORCE, terminator=False, on_finish=None):
    stack(executor, interface, controller, container_name, target_name, pickup_dx, pickup_dy, pickup_dz, place_dx, place_dy, place_dz, theta, rot_time, grip_time, grip_force, terminator, on_finish)
