import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from hybrid_osc14 import WrenchAugmentedOSC14

def measure(target):
    np.random.seed(0)
    cfg = load_controller_config(default_controller="OSC_POSE")
    env = suite.make("Wipe", robots="Panda", controller_configs=cfg,
                     has_renderer=False, use_camera_obs=False, horizon=400, control_freq=20)
    np.random.seed(0)
    obs = env.reset()
    ctrl = env.robots[0].controller
    ctrl.__class__ = WrenchAugmentedOSC14
    ctrl.init_hybrid()

    touched = False
    for _ in range(150):
        a = np.zeros(env.action_dim); a[2] = -1.0
        obs, _, _, _ = env.step(a)
        if bool(obs.get("robot0_contact", 0.0)):
            touched = True
            break

    ctrl.set_force_target([0, 0, target, 0, 0, 0], axes=(2,))
    mz, contact = [], []
    for _ in range(150):
        r = env.robots[0]
        w = -np.concatenate([np.array(r.ee_force), np.array(r.ee_torque)])
        ctrl.set_measured_wrench(w)
        obs, _, _, _ = env.step(np.zeros(env.action_dim))
        mz.append(float(r.ee_force[2]))
        contact.append(float(obs.get("robot0_contact", 0.0)))
    return np.array(mz), touched, float(np.mean(contact[-50:]))

for T in [20.0, -20.0]:
    mz, touched, c = measure(T)
    tail = mz[-50:]
    print(f"target {T:+.0f}: approached={touched}  measured_z last50 mean {tail.mean():+.2f} "
          f"std {tail.std():.2f} contact {c:.2f}")
