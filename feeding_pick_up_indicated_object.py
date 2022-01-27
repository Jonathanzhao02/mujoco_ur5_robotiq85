import glfw
import numpy as np


class GripperStatus:
    def __init__(self, gripper_status):
        self.gripper_status = gripper_status

    def get_gripper_status(self):
        return self.gripper_status

    def set_gripper_status(self, new_gripper_status):
        self.gripper_status = new_gripper_status


class Action:
    def __init__(self, interface, controller):
        self.interface = interface
        self.controller = controller
        return
    
    # abstract method in base class
    # https://stackoverflow.com/questions/4382945/abstract-methods-in-python
    def execute(self):
        raise NotImplementedError("Please Implemente this Method")


class MoveTo(Action):
    def __init__(self, interface, controller, target_func, gripper_control_func, time_limit=None, error_limit=None):
        assert time_limit is not None or error_limit is not None, "at least 1 of time limit or error limit should be indicated"

        super().__init__(interface, controller)
        self.target_func = target_func
        self.gripper_control_func = gripper_control_func
        self._gripper = None
        self.time_limit = time_limit
        self.error_limit = error_limit

    def _set_gripper(self, gripper):
        self._gripper = gripper

    def execute(self):
        time_step = 0
        while True:

            # Check Window
            if self.interface.viewer.exit:
                glfw.destroy_window(self.interface.viewer.window)
                break

            # Get Target
            target = self.target_func(self.interface)

            # Calculate Forces
            feedback = self.interface.get_feedback()
            u = self.controller.generate(
                q=feedback["q"],
                dq=feedback["dq"],
                target=target,
            )

            # Set gripper force
            if self._gripper is not None:
                u = self.gripper_control_func(u, self._gripper)

            # send forces into Mujoco, step the sim forward
            self.interface.send_forces(u)

            # calculate time step
            time_step += 1
            # calculate error
            # ee_xyz = robot_config.Tx("EE", q=feedback["q"])
            ee_xyz = self.interface.get_xyz("EE")
            error = np.linalg.norm(ee_xyz - target[:3])

            # whether stop criterion has been reached
            if self.time_limit is not None:
                if time_step >= self.time_limit:
                    break
            if self.error_limit is not None:
                if error <= self.error_limit:
                    break
        

class Executor:
    def __init__(self, interface, start_angles, start_gripper_status):
        self.interface = interface
        self.action_list = []
        interface.send_target_angles(start_angles)
        self.gripper = GripperStatus(start_gripper_status)

    def append(self, action):
        action._set_gripper(self.gripper)
        self.action_list.append(action)

    def execute(self):
        for i in range(len(self.action_list)):
            self.action_list[i].execute()

    def execute_action(self, action):
        action._set_gripper(self.gripper)
        action.execute()

    def get_state(self):
        # Check Window
        if hasattr(self.interface, 'viewer'):
            if self.interface.viewer.exit:
                glfw.destroy_window(self.interface.viewer.window)
                return

        state = {}
        feedback = self.interface.get_feedback()
        state['q'] = feedback['q']
        state['dq'] = feedback['dq']
        state['target'] = self.action_list[self.stage].target_func(self.interface)
        state['objects_to_track'] = self.get_objects_to_track()
        state['stage'] = self.stage
        state['step_in_stage'] = self.step_in_stage
        state['goal_object'] = self.goal_object
        return state

    def render_img(self):
        self.offscreen.render(224, 224, 0)
        img = self.offscreen.read_pixels(224, 224)[0]
        return img

class Trajectory:
    def __init__(self, traj):
        # traj: (seq_len, 3)
        assert traj.shape[1] == 3
        self.traj = traj
        self.idx = 0
    
    def get_traj(self, a):
        to_return = np.concatenate((self.traj[self.idx], np.array([3.14, 0, 1.57])), axis=0)
        self.idx += 1
        if self.idx == self.traj.shape[0]:
            self.idx -= 1
        return to_return


if __name__ == '__main__':

    import numpy as np
    from abr_control.controllers import Damping
    from my_osc import OSC
    from mujoco_interface import Mujoco
    from abr_control.utils import transformations
    from my_mujoco_config import MujocoConfig as arm

    targets = ["target1", "target2", "target3"]
    # target_name = targets[np.random.randint(len(targets))]
    target_name = 'target1'

    def target_func(interface):
        target_xyz = interface.get_xyz(target_name)
        target_xyz[-2] -= 0.15
        target = np.hstack(
            [
                target_xyz,
                (3.14, 0, transformations.euler_from_quaternion(interface.get_orientation(target_name), "rxyz")[-1] + 1.57),
            ]
        )
        return target

    def gripper_control_func(u, gripper):
        u[-1] = gripper.get_gripper_status()
        return u

    # create our Mujoco interface
    robot_config = arm('ur5_3_objects.xml', folder='./my_models/ur5_robotiq85')
    interface = Mujoco(robot_config, dt=0.008)
    interface.connect(joint_names=['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'finger_joint'], camera_id=0)

    # Randomly place the target object
    def gen_target(interface):

        def l2(pos1, pos2):
            assert len(pos1) == len(pos2)
            d = 0
            for i in range(len(pos1)):
                d = d + (pos1[i] - pos2[i]) ** 2
            d = d ** (1 / 2)
            return d

        point_list = []

        for i in range(3):
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

    model_to_test = 'backbone_improved_dmp'
    if model_to_test == 'dmp':
        from module_traj_generator import *
        model = Backbone(num_traces=3, embedding_size=192, num_weight_points=31, device='cpu')
        model.load_state_dict(torch.load('/share/yzhou298/ckpts/module-traj-generator-30-points-wo-bn-larger1-discount-1/7000.pth'), strict=False)
    elif model_to_test = 'backbone_improved_dmp':
        from main_dmp_improved_dmp import *
        model = Backbone(img_size=128, num_traces=3, num_joints=7, num_tasks=3, embedding_size=192, num_weight_points=31)
        model.load_state_dict(torch.load('/share/yzhou298/ckpts/train1-7-improved-dmp-sim-ur5-2/33000.pth'), strict=False)
    mean = np.array([2.97563984e-02,  4.47217117e-01,  8.45049397e-02])
    var = np.array([4.52914246e-02, 5.01675921e-03, 4.19371463e-03])

    # Example of live feed
    e = Executor(interface, robot_config.START_ANGLES, -0.05)
    t_lim = 300
    t_interval = 150
    i = 0
    while i < t_lim:
        # Provide input from simulator
        ee_pos = torch.tensor((interface.get_xyz('EE') - mean) / var, dtype=torch.float32).unsqueeze(0)
        target_pos = torch.tensor((interface.get_xyz(target_name) - mean) / var, dtype=torch.float32).unsqueeze(0)
        phis = torch.tensor(np.linspace(0.0, 1.01, t_lim-i, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        # phis = torch.tensor(np.linspace(0.0, 1.01, 200, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        target = torch.zeros(1, dtype=torch.int64)
        img = torch.tensor(e.render_img() / 3).unsqueeze(0)
        print(img.shape)
        exit()
        
        # Run the model
        if model_to_test == 'dmp'
            traj, dmp_weights = model(ee_pos[:, :3], target_pos[:, :3], phis)
        elif model_to_test = 'backbone_improved_dmp':
            traj, _, _, _, _, _, _, _, _ = model(img, ee_pos[:, :3], joint_angles, target, phis)

        # Format the predictions
        traj = np.transpose(traj.detach().squeeze().numpy()) * var + mean
        print(traj.shape)
        
        # Execute the predictions
        traj_obj = Trajectory(traj)
        e.execute_action(MoveTo(interface, ctrlr, traj_obj.get_traj, gripper_control_func, time_limit=t_interval))
        
        # Count steps
        i += t_interval
