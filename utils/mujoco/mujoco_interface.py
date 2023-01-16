import numpy as np
import mujoco as mj

from abr_control.utils import transformations
from abr_control.interfaces.interface import Interface


class Mujoco(Interface):
    """An interface for accessing MuJoCo data using the mujoco package.
    Based on ABRControl

    Parameters
    ----------
    robot_config: class instance
        contains all relevant information about the arm
        such as: number of joints, number of links, mass information etc.
    """

    def __init__(self, robot_config):
        Interface.__init__(self, robot_config)
        self.model = robot_config.model
        self.data = robot_config.data
    
    def connect(self, joint_names):
        self.joint_names = joint_names
        self.joints = [self.model.jnt(name) for name in self.joint_names]
        # Assumes all hinge joints (1 dof)
        self.joint_ids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.joint_pos_addrs = [joint.qposadr[0] for joint in self.joints]
        self.joint_vel_addrs = [joint.dofadr[0] for joint in self.joints]

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
        
        self.robot_config.connect(
            self.joint_pos_addrs,
            self.joint_vel_addrs,
            self.joint_dyn_addrs,
        )
    
    def get_feedback(self):
        """Return a dictionary of information needed by the controller.

        Returns the joint angles and joint velocities in [rad] and [rad/sec],
        respectively
        """

        q = np.copy(self.data.qpos[self.joint_pos_addrs])
        dq = np.copy(self.data.qvel[self.joint_vel_addrs])

        return {"q": q, "dq": dq}

    def get_orientation(self, name, object_type="body"):
        """Returns the orientation of an object as the [w x y z] quaternion [radians]

        Parameters
        ----------
        name: string
            the name of the object of interest
        object_type: string, Optional (Default: body)
            The type of mujoco object to get the orientation of.
            Can be: mocap, body, geom, site
        """
        if object_type == "mocap":  # commonly queried to find target
            quat = self.data.mocap(name).quat
        elif object_type == "body":
            quat = self.data.body(name).xquat
        elif object_type == "geom":
            xmat = self.data.geom(name).xmat
            quat = transformations.quaternion_from_matrix(xmat.reshape((3, 3)))
        elif object_type == "site":
            xmat = self.data.site(name).xmat
            quat = transformations.quaternion_from_matrix(xmat.reshape((3, 3)))
        else:
            raise Exception(
                f"get_orientation for {object_type} object type not supported"
            )
        return np.copy(quat)

    def get_xyz(self, name, object_type="body"):
        """Returns the xyz position of the specified object

        name: string
            name of the object you want the xyz position of
        object_type: string
            type of object you want the xyz position of
            Can be: mocap, body, geom, site
        """
        if object_type == "mocap":  # commonly queried to find target
            xyz = self.data.mocap(name).pos
        elif object_type == "body":
            xyz = self.data.body(name).xpos
        elif object_type == "geom":
            xyz = self.data.geom(name).xpos
        elif object_type == "site":
            xyz = self.data.site(name).xpos
        else:
            raise Exception(f"get_xyz for {object_type} object type not supported")

        return np.copy(xyz)
    
    def set_xyz(self, name, xyz, object_type="body"):
        if object_type == "body":
            self.data.body(name).qpos[:3] = xyz[:3]
