# mujoco_ur5_robotiq85
## mujoco, mujoco-py and abr_control
mujoco is a c++(I believe) program. Currently the default version is mujoco210. We use mujoco-py as a bridge to control mujoco from python. 
The default order is to install mujoco first and then install mujoco-py. 

If we want to use controllers, e.g., IK controllers, we need to install abr_controll. abr_controll creates its own version of mujoco-py, which needs mujoco200.
We need to intall this customized mujoco-py instead of the original one. We also need to install mujoco200 together with mujoco210.

Therefore, the package install order is:
- mujoco210
- mujoco200
- abr_control
- mujoco-py (from abr_control)
