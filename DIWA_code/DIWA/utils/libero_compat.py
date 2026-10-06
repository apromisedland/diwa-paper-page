"""Narrow runtime compatibility helpers for supported LIBERO installs."""

from __future__ import annotations

import inspect


def apply_robosuite_mujoco3_compat() -> bool:
    """Adapt robosuite 1.4's mass-matrix call to the MuJoCo 3 API.

    LIBERO 0.1.1 leaves ``mujoco`` unpinned.  On Python versions for which
    MuJoCo 2 wheels are unavailable, pip can therefore resolve MuJoCo 3 while
    still installing robosuite 1.4.  That robosuite release calls the old
    ``mj_fullM(model, dst, qM)`` API.  Patch the controller method in memory
    only when that exact incompatible combination is detected.

    Returns ``True`` when the compatibility method was installed.
    """

    import mujoco
    import numpy as np
    from robosuite.controllers.base_controller import Controller

    try:
        mujoco_major = int(mujoco.__version__.split(".", 1)[0])
    except (AttributeError, ValueError):
        return False
    if mujoco_major < 3:
        return False
    if getattr(Controller, "_diwa_mujoco3_compat", False):
        return False

    try:
        update_source = inspect.getsource(Controller.update)
    except (OSError, TypeError):
        update_source = ""
    if ".qM" not in update_source:
        # A newer robosuite already uses the MuJoCo 3 signature.
        return False

    def update(self, force=False):
        if self.new_update or force:
            self.sim.forward()

            site_id = self.sim.model.site_name2id(self.eef_name)
            self.ee_pos = np.asarray(self.sim.data.site_xpos[site_id]).copy()
            self.ee_ori_mat = np.asarray(
                self.sim.data.site_xmat[site_id].reshape(3, 3)
            ).copy()
            self.ee_pos_vel = np.asarray(
                self.sim.data.get_site_xvelp(self.eef_name)
            ).copy()
            self.ee_ori_vel = np.asarray(
                self.sim.data.get_site_xvelr(self.eef_name)
            ).copy()
            self.joint_pos = np.asarray(
                self.sim.data.qpos[self.qpos_index]
            ).copy()
            self.joint_vel = np.asarray(
                self.sim.data.qvel[self.qvel_index]
            ).copy()
            self.J_pos = np.asarray(
                self.sim.data.get_site_jacp(self.eef_name).reshape(3, -1)[
                    :, self.qvel_index
                ]
            ).copy()
            self.J_ori = np.asarray(
                self.sim.data.get_site_jacr(self.eef_name).reshape(3, -1)[
                    :, self.qvel_index
                ]
            ).copy()
            self.J_full = np.vstack((self.J_pos, self.J_ori))

            mass_matrix = np.empty(
                (self.sim.model.nv, self.sim.model.nv),
                dtype=np.float64,
                order="C",
            )
            mujoco.mj_fullM(
                self.sim.model._model,
                self.sim.data._data,
                mass_matrix,
            )
            self.mass_matrix = mass_matrix[
                self.qvel_index, :
            ][:, self.qvel_index]
            self.new_update = False

    Controller.update = update
    Controller._diwa_mujoco3_compat = True
    return True
