import os
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from hybrid_osc14 import WrenchAugmentedOSC14

bd = benchmark.get_benchmark_dict()
suite = bd["libero_90"]()
names = suite.get_task_names()
target = "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet"
i = names.index(target)
task = suite.get_task(i)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.reset()
rs = env.env
ctrl = rs.robots[0].controller
print("task:", target)
print("controller class:", type(ctrl).__name__)
print("action_dim:", rs.action_dim)

ctrl.__class__ = WrenchAugmentedOSC14
ctrl.init_hybrid()
ctrl.set_force_target([0, 0, -10, 0, 0, 0], axes=(2,))
for _ in range(5):
    r = rs.robots[0]
    w = -np.concatenate([np.array(r.ee_force), np.array(r.ee_torque)])
    ctrl.set_measured_wrench(w)
    rs.step(np.zeros(rs.action_dim))
print("HYBRID CONTROLLER ATTACHED + STEPPED IN LIBERO OK")
