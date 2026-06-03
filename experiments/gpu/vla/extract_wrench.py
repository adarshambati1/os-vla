import os, sys
import numpy as np
import h5py
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

TASKS = [
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE7_open_the_microwave",
    "KITCHEN_SCENE6_close_the_microwave",
    "KITCHEN_SCENE3_turn_on_the_stove",
    "KITCHEN_SCENE8_turn_off_the_stove",
]
DSDIR = "/workspace/LIBERO/libero/datasets/libero_90"
OUT = "/workspace/vla_data/wrench"

def run(name):
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task = suite.get_task(suite.get_task_names().index(name))
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.reset()
    sim = env.env.sim
    robot = env.env.robots[0]
    f = h5py.File(f"{DSDIR}/{name}_demo.hdf5", "r")
    data = f["data"]
    out = {}
    for dn in data.keys():
        states = data[dn]["states"][:]
        w = np.empty((len(states), 6), np.float32)
        lo = np.array([-100, -100, -100, -20, -20, -20], np.float32)
        hi = -lo
        for t in range(len(states)):
            sim.set_state_from_flattened(states[t])
            sim.forward()
            raw = -np.concatenate([np.array(robot.ee_force), np.array(robot.ee_torque)])
            w[t] = np.clip(raw, lo, hi)
        out[dn] = w
    f.close()
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{name}.npz", **out)
    mx = max(float(np.abs(v).max()) for v in out.values())
    print(f"{name}: {len(out)} demos saved, max|wrench|={mx:.1f}", flush=True)

if __name__ == "__main__":
    run(TASKS[int(sys.argv[1])])
