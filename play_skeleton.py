# import mujoco
# import mujoco.viewer
# import numpy as np
# import os

# # Path to the XML model
# xml_path = "loco_mujoco/models/skeleton/skeleton_torque.xml"
# xml_path = os.path.abspath(xml_path)

# # Load model and data
# model = mujoco.MjModel.from_xml_path(xml_path)
# data = mujoco.MjData(model)

# # Viewer loop
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         # Simple sinusoidal action
#         data.ctrl[:] = 0.3 * np.sin(2 * np.pi * 0.5 * data.time)
#         mujoco.mj_step(model, data)
#         viewer.sync()


import mujoco
import mujoco.viewer
import numpy as np
import os
import time
import h5py

xml_path = os.path.abspath("loco_mujoco/models/skeleton/skeleton_muscle.xml")
file_path = "/home/hlz/repos/humenv-tutorial/data_preparation/humenv_amass/0-ACCAD_Female1General_c3d_A1-Stand_poses.hdf5"

with h5py.File(file_path, "r") as f:
    qpos = f["ep_0/qpos"][:]  # (541, 76) float32

print(qpos.shape)

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


fps = 30
dt = 1.0 / fps

with mujoco.viewer.launch_passive(model, data) as viewer:
    for qp in qpos:
        data.qpos[:] = qp
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(dt)
