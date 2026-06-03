import numpy as np
from opspace_vla.env import make_env, install_hybrid_controller
from opspace_vla.scripted_policy import scripted_osc_action

def test_action_dims():
    assert make_env("osc", horizon=20).action_dim == 6
    assert make_env("joint", horizon=20).action_dim == 7

def test_hybrid_force_off_matches_osc():
    def run(use_hybrid):
        env = make_env("osc", horizon=60, control_freq=10, seed=7)
        obs = env.reset()
        obs["_phase"] = 0
        if use_hybrid:
            install_hybrid_controller(env)
        torques = []
        for _ in range(60):
            action, phase, _ = scripted_osc_action(obs, env)
            obs, _, done, _ = env.step(action)
            obs["_phase"] = phase
            right = env.robots[0].composite_controller.part_controllers["right"]
            torques.append(np.array(right.torques).copy())
            if done:
                break
        return np.array(torques)

    plain = run(False)
    hybrid = run(True)
    n = min(len(plain), len(hybrid))
    assert np.abs(plain[:n] - hybrid[:n]).max() < 1e-6
