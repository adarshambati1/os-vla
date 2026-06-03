import numpy as np
from opspace_vla.env import PHASE_APPROACH, PHASE_WIPE, OSC_ACTION_DIM, IMPEDANCE_DIM

def stiffness_schedule(in_contact):
    kp = np.full(IMPEDANCE_DIM, 150.0, np.float32)
    if in_contact:
        kp[2] = 300.0
    return kp

def scripted_osc_action(obs, env):
    g = np.asarray(obs["gripper_to_wipe_centroid"], np.float32) #easy one
    in_contact = bool(obs["robot0_contact"])
    phase = PHASE_WIPE if in_contact else PHASE_APPROACH
    obs["_phase"] = phase

    a = np.zeros(OSC_ACTION_DIM, np.float32)
    a[0] = np.clip(5.0 * g[0], -1, 1)
    a[1] = np.clip(5.0 * g[1], -1, 1)
    a[2] = -1.0 if not in_contact else -0.5

    return a, phase, stiffness_schedule(in_contact)
