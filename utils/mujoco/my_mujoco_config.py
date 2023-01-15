import os
from xml.etree import ElementTree

import mujoco as mj
import numpy as np

from abr_control.utils import download_meshes


class MujocoConfig:
    """A wrapper on the Mujoco simulator to generate all the kinematics and
    dynamics calculations necessary for controllers.
    """

    def __init__(self, model, data):
        """

        Parameters
        ----------
        model: MjModel
            Model to use
        data: MjData
            Data to use
        """
        self.model = model
        self.data = data
        self.N_GRIPPER_JOINTS = self.model.numeric("N_GRIPPER_JOINTS").data[0]

    def connect(self, joint_pos_addrs, joint_vel_addrs, joint_dyn_addrs):
        self.joint_pos_addrs = np.copy(joint_pos_addrs)
        self.joint_vel_addrs = np.copy(joint_vel_addrs)
        self.joint_dyn_addrs = np.copy(joint_dyn_addrs)

        self.N_JOINTS = len(self.joint_dyn_addrs)

        # number of joints in the Mujoco simulation
        N_ALL_JOINTS = self.model.nv

        # need to calculate the joint_dyn_addrs indices in flat vectors returned
        # for the Jacobian
        self.jac_indices = np.hstack(
            # 6 because position and rotation Jacobians are 3 x N_JOINTS
            [self.joint_dyn_addrs + (ii * N_ALL_JOINTS) for ii in range(3)]
        )

        # for the inertia matrix
        self.M_indices = [
            ii * N_ALL_JOINTS + jj
            for jj in self.joint_dyn_addrs
            for ii in self.joint_dyn_addrs
        ]

        # a place to store data returned from Mujoco
        self._g = np.zeros(self.N_JOINTS)
        self._J3NP = np.zeros(3 * N_ALL_JOINTS)
        self._J3NR = np.zeros(3 * N_ALL_JOINTS)
        self._J6N = np.zeros((6, self.N_JOINTS))
        self._MNN_vector = np.zeros((N_ALL_JOINTS, N_ALL_JOINTS))
        self._MNN = np.zeros(self.N_JOINTS ** 2)
        self._R9 = np.zeros(9)
        self._R = np.zeros((3, 3))
        self._x = np.ones(4)
        self.N_ALL_JOINTS = N_ALL_JOINTS

    def g(self, q=None):
        """Returns qfrc_bias variable, which stores the effects of Coriolis,
        centrifugal, and gravitational forces

        Parameters
        ----------
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        # TODO: For the Coriolis and centrifugal functions, setting the
        # velocity before calculation is important, how best to do this?
        g = -1 * self.data.qfrc_bias[self.joint_dyn_addrs]

        return g

    def dJ(self, name, q=None, dq=None, x=None):
        """Returns the derivative of the Jacobian wrt to time

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        dq: float numpy.array, optional (Default: None)
            The joint velocities of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        """
        # TODO if ever required
        # Note from Emo in Mujoco forums:
        # 'You would have to use a finate-difference approximation in the
        # general case, check differences.cpp'
        raise NotImplementedError

    def J(self, name, q=None, x=None, object_type="body"):
        """Returns the Jacobian for the specified Mujoco object

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        object_type: string, the Mujoco object type, optional (Default: body)
            options: body, geom, site
        """
        if x is not None and not np.allclose(x, 0):
            raise Exception("x offset currently not supported, set to None")

        if object_type == "body":
            # TODO: test if using this function is faster than the old way
            # NOTE: for bodies, the Jacobian for the COM is returned
            mj.mj_jacBodyCom(
                self.model,
                self.data,
                self._J3NP.reshape(3, -1),
                self._J3NR.reshape(3, -1),
                mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name),
            )
        else:
            if object_type == "geom":
                jacp = self.data.get_geom_jacp
                jacr = self.data.get_geom_jacr
            elif object_type == "site":
                jacp = self.data.get_site_jacp
                jacr = self.data.get_site_jacr
            else:
                raise Exception("Invalid object type specified: ", object_type)

            jacp(name, self._J3NP)[self.jac_indices]  # pylint: disable=W0106
            jacr(name, self._J3NR)[self.jac_indices]  # pylint: disable=W0106

        # get the position Jacobian hstacked (1 x N_JOINTS*3)
        self._J6N[:3] = self._J3NP[self.jac_indices].reshape((3, self.N_JOINTS))
        # get the rotation Jacobian hstacked (1 x N_JOINTS*3)
        self._J6N[3:] = self._J3NR[self.jac_indices].reshape((3, self.N_JOINTS))

        return np.copy(self._J6N)

    def M(self, q=None):
        """Returns the inertia matrix in task space

        Parameters
        ----------
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        # stored in mjData.qM, stored in custom sparse format,
        # convert qM to a dense matrix with mj_fullM
        mj.mj_fullM(self.model, self._MNN_vector, self.data.qM)
        M = self._MNN_vector.flatten()[self.M_indices]
        M = M.reshape((self.N_JOINTS, self.N_JOINTS))

        return np.copy(M)

    def R(self, name, q=None):
        """Returns the rotation matrix of the specified body

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        mj.mju_quat2Mat(self._R9, self.data.body(name).xquat)
        self._R = self._R9.reshape((3, 3))

        return self._R

    def quaternion(self, name, q=None):
        """Returns the quaternion of the specified body
        Parameters
        ----------

        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        """
        quaternion = np.copy(self.data.get_body_xquat(name))

        return quaternion

    def C(self, q=None, dq=None):
        """NOTE: The Coriolis and centrifugal effects (and gravity) are
        already accounted for by Mujoco in the qfrc_bias variable. There's
        no easy way to separate these, so all are returned by the g function.
        To prevent accounting for these effects twice, this function will
        return an error instead of qfrc_bias again.
        """
        raise NotImplementedError(
            "Coriolis and centrifugal effects already accounted "
            + "for in the term return by the gravity function."
        )

    def T(self, name, q=None, x=None):
        """Get the transform matrix of the specified body.

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        """
        # TODO if ever required
        raise NotImplementedError

    def Tx(self, name, q=None, x=None, object_type="body"):
        """Returns the Cartesian coordinates of the specified Mujoco body

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        object_type: string, the Mujoco object type, optional (Default: body)
            options: body, geom, site, camera, light, mocap
        """
        if x is not None and not np.allclose(x, 0):
            raise Exception("x offset currently not supported: ", x)

        if object_type == "body":
            Tx = np.copy(self.data.body(name).xpos)
        elif object_type == "geom":
            Tx = np.copy(self.data.geom(name).xpos)
        elif object_type == "joint":
            Tx = np.copy(self.data.joint(name).xanchor)
        elif object_type == "site":
            Tx = np.copy(self.data.site(name).xpos)
        elif object_type == "camera":
            Tx = np.copy(self.data.cam(name).xpos)
        elif object_type == "light":
            Tx = np.copy(self.data.light(name).xpos)
        elif object_type == "mocap":
            Tx = np.copy(self.data.mocap(name).pos)
        else:
            raise Exception("Invalid object type specified: ", object_type)

        return Tx

    def T_inv(self, name, q=None, x=None):
        """Get the inverse transform matrix of the specified body.

        Parameters
        ----------
        name: string
            The name of the Mujoco body to retrieve the Jacobian for
        q: float numpy.array, optional (Default: None)
            The joint angles of the robot. If None the current state is
            retrieved from the Mujoco simulator
        x: float numpy.array, optional (Default: None)
        """
        # TODO if ever required
        raise NotImplementedError
