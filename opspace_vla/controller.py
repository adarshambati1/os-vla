import numpy as np

from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.utils.control_utils import (
    opspace_matrices, nullspace_torques, orientation_error,
)

class WrenchAugmentedOSC(OperationalSpaceController):

    def init_hybrid(self, Kfp=0.5, Kfi=0.02, i_clamp=20.0):
        self._S = np.zeros(6)
        self._F_d = np.zeros(6)
        self._F_meas = np.zeros(6)
        self._I = np.zeros(6)
        self._Kfp, self._Kfi, self._i_clamp = Kfp, Kfi, i_clamp

    def set_force_target(self, wrench_6d, axes=(2,)):
        self._F_d = np.asarray(wrench_6d, float).reshape(6)
        S = np.zeros(6)
        for a in axes:
            S[a] = 1.0
        if not np.array_equal(S, self._S):
            self._I = np.zeros(6)
        self._S = S

    def set_measured_wrench(self, wrench_6d):
        self._F_meas = np.asarray(wrench_6d, float).reshape(6)

    def reset_hybrid(self):
        self._S = np.zeros(6); self._F_d = np.zeros(6)
        self._F_meas = np.zeros(6); self._I = np.zeros(6)

    def run_controller(self):
        if not hasattr(self, "_S"):
            return super().run_controller()

        self.update()

        if self.input_ref_frame == "base":
            desired_world_pos = self.origin_pos + np.dot(self.origin_ori, self.goal_pos)
            desired_world_ori = np.dot(self.origin_ori, self.goal_ori)
        else:
            desired_world_pos = self.goal_pos
            desired_world_ori = self.goal_ori
        ori_error = orientation_error(desired_world_ori, self.ref_ori_mat)

        position_error = desired_world_pos - self.ref_pos
        base_pos_vel = np.array(self.sim.data.get_site_xvelp(f"{self.naming_prefix}{self.part_name}_center"))
        vel_pos_error = -(self.ref_pos_vel - base_pos_vel)
        desired_force = np.multiply(position_error, self.kp[0:3]) + np.multiply(vel_pos_error, self.kd[0:3])

        base_ori_vel = np.array(self.sim.data.get_site_xvelr(f"{self.naming_prefix}{self.part_name}_center"))
        vel_ori_error = -(self.ref_ori_vel - base_ori_vel)
        desired_torque = np.multiply(ori_error, self.kp[3:6]) + np.multiply(vel_ori_error, self.kd[3:6])

        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            self.mass_matrix, self.J_full, self.J_pos, self.J_ori
        )
        if self.uncoupling:
            W_m = np.concatenate([np.dot(lambda_pos, desired_force), np.dot(lambda_ori, desired_torque)])
        else:
            W_m = np.dot(lambda_full, np.concatenate([desired_force, desired_torque]))

        if not np.any(self._S):
            wrench = W_m
        else:
            err = (self._F_d - self._F_meas) * self._S
            self._I = np.clip(self._I + self._Kfi * err, -self._i_clamp, self._i_clamp)
            F_force = self._F_d + self._Kfp * err + self._I
            wrench = (1.0 - self._S) * W_m + self._S * F_force

        self.torques = np.dot(self.J_full.T, wrench) + self.torque_compensation
        self.torques += nullspace_torques(
            self.mass_matrix, nullspace_matrix, self.initial_joint, self.joint_pos, self.joint_vel
        )

        super(OperationalSpaceController, self).run_controller()
        return self.torques
