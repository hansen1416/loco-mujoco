# from loco_mujoco.task_factories import (
#     ImitationFactory,
#     LAFAN1DatasetConf,
#     DefaultDatasetConf,
#     AMASSDatasetConf,
# )


# # # example --> you can add as many datasets as you want in the lists!
# env = ImitationFactory.make(
#     "UnitreeH1",
#     default_dataset_conf=DefaultDatasetConf(["squat"]),
#     lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]),
#     # if SMPL and AMASS are installed, you can use the following:
#     # amass_dataset_conf=AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses"])
# )

# env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)


import numpy as np
from loco_mujoco import ImitationFactory


# create the environment and task
env = ImitationFactory.make(
    "FourierGR1T2", default_dataset_conf=dict(task="stepinplace1")
)

# get the dataset for the chosen environment and task -- can be used for GAIL-like algorithms
# expert_data = env.create_dataset()

action_dim = env.info.action_space.shape[0]

env.reset()

env.render()
absorbing = False
i = 0

while True:
    if i == 1000 or absorbing:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)
    nstate, reward, absorbing, done, info = env.step(action)

    env.render()
    i += 1
