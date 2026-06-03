import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from hybrid_osc14 import WrenchAugmentedOSC14

def run(hybrid):
    np.random.seed(0)
    cfg = load_controller_config(default_controller="OSC_POSE")
    env = suite.make("Lift", robots="Panda", controller_configs=cfg,
                     has_renderer=False, use_camera_obs=False, use_object_obs=True,
                     horizon=70, control_freq=20)
    np.random.seed(0)
    env.reset()
    ctrl = env.robots[0].controller
    if hybrid:
        ctrl.__class__ = WrenchAugmentedOSC14
        ctrl.init_hybrid()
    acts = np.random.RandomState(1).uniform(-0.3, 0.3, (60, env.action_dim))
    torques = []
    for a in acts:
        env.step(a)
        torques.append(np.array(env.robots[0].controller.torques).copy())
    return np.array(torques)

stock = run(False)
hyb = run(True)
n = min(len(stock), len(hyb))
diff = np.abs(stock[:n] - hyb[:n]).max()
print("controller attr OK; steps", n)
print("max torque diff (force-off vs stock OSC):", diff)
print("GATE A", "PASS" if diff < 1e-6 else "FAIL")
