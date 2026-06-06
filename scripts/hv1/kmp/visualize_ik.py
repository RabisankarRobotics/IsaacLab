"""Interactive viewer for IK solutions.

Loads HV1 in MuJoCo, samples a random V3-style command, solves the staged IK,
snaps the robot to the resulting joint pose, and draws sphere markers at the
commanded wrist targets so you can eyeball how close the solver got.

Press SPACE to roll a new random command.

Usage:
  conda activate env_isaaclab
  python scripts/hv1/kmp/visualize_ik.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_kmp_dataset import (  # noqa: E402
    ACTUATED_28, DEFAULT_URDF, IKWorker, sample_command,
)


HV1_XML = "/home/rabisankar/IsaacLab/deploy/mujoco/scene.xml"


def build_qpos_remap(mj_model) -> tuple[np.ndarray, int]:
    """Map V3 action order (28,) -> MuJoCo qpos slot for each joint.
    Also returns the qpos start index of the floating base (or -1 if fixed)."""
    qpos_idx = np.zeros(len(ACTUATED_28), dtype=np.int64)
    for i, name in enumerate(ACTUATED_28):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint '{name}' not in MuJoCo model")
        qpos_idx[i] = mj_model.jnt_qposadr[jid]
    base_adr = -1
    for j in range(mj_model.njnt):
        if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            base_adr = int(mj_model.jnt_qposadr[j])
            break
    return qpos_idx, base_adr


def add_sphere_marker(scn, pos, rgba, radius=0.04):
    """Append a sphere to the viewer's user scene."""
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0, 0]),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


class IKVizState:
    """Shared mutable state between the keyboard callback and main loop."""

    def __init__(self):
        self.cmd: Optional[dict] = None
        self.q_v3: Optional[np.ndarray] = None
        self.err_foot_mm: float = 0.0
        self.err_wrist_mm: float = 0.0
        self.need_new = True   # triggers initial solve
        self.seed = int(time.time()) & 0xFFFFFFFF


def solve_new_command(state: IKVizState, ik: IKWorker, rng: np.random.Generator) -> None:
    state.cmd = sample_command(rng)
    out = ik.solve(state.cmd)
    state.q_v3 = out["q"]
    state.err_foot_mm = out["err_foot"] * 1000
    state.err_wrist_mm = out["err_wrist"] * 1000
    c = state.cmd
    print("-" * 70)
    print(f"new cmd: h={c['h']:.3f}  "
          f"L=({c['left_pos'][0]:+.2f}, {c['left_pos'][1]:+.2f}, {c['left_pos'][2]:+.2f})  "
          f"R=({c['right_pos'][0]:+.2f}, {c['right_pos'][1]:+.2f}, {c['right_pos'][2]:+.2f})  "
          f"alpha={c['alpha_t']:.2f}")
    print(f"  IK err: foot={state.err_foot_mm:5.2f}mm   wrist={state.err_wrist_mm:6.2f}mm")


def main():
    print(f"[LOAD] MuJoCo model: {HV1_XML}")
    mj_model = mujoco.MjModel.from_xml_path(HV1_XML)
    mj_data = mujoco.MjData(mj_model)
    qpos_idx, base_adr = build_qpos_remap(mj_model)

    print(f"[LOAD] Pinocchio model: {DEFAULT_URDF}")
    ik = IKWorker(DEFAULT_URDF)

    state = IKVizState()
    rng = np.random.default_rng(state.seed)

    def key_callback(keycode: int):
        # GLFW spacebar
        if keycode == 32:
            state.need_new = True

    print("\n[CONTROLS]")
    print("  SPACE — sample a new random command + solve IK")
    print("  ESC   — quit")
    print("  Drag, scroll = standard MuJoCo viewer controls\n")

    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=key_callback
    ) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135

        while viewer.is_running():
            if state.need_new:
                solve_new_command(state, ik, rng)
                # Push IK joints into MuJoCo qpos
                mj_data.qpos[qpos_idx] = state.q_v3
                # Float the pelvis so feet sit on the ground (target z = -h
                # relative to pelvis -> pelvis world z = h).
                if base_adr >= 0:
                    mj_data.qpos[base_adr + 0] = 0.0
                    mj_data.qpos[base_adr + 1] = 0.0
                    mj_data.qpos[base_adr + 2] = state.cmd["h"]
                    # quat (w, x, y, z) identity
                    mj_data.qpos[base_adr + 3:base_adr + 7] = [1, 0, 0, 0]
                # mj_forward: pure kinematics — no physics step, robot just snaps
                mujoco.mj_forward(mj_model, mj_data)
                state.need_new = False

            # Draw target markers on user scene. Wrist commands are in pelvis
            # frame; with pelvis at (0,0,h) the world coord is just offset by h.
            viewer.user_scn.ngeom = 0
            if state.cmd is not None:
                pelvis_z = state.cmd["h"]
                lp = state.cmd["left_pos"]  + np.array([0, 0, pelvis_z])
                rp = state.cmd["right_pos"] + np.array([0, 0, pelvis_z])
                add_sphere_marker(viewer.user_scn, lp,
                                  rgba=(0.2, 0.8, 0.2, 0.7))   # green = left wrist target
                add_sphere_marker(viewer.user_scn, rp,
                                  rgba=(0.2, 0.2, 0.9, 0.7))   # blue  = right wrist target
                # Foot targets land on the ground (world z = 0)
                add_sphere_marker(viewer.user_scn, [-0.0035,  0.211, 0.0],
                                  rgba=(0.9, 0.6, 0.1, 0.5), radius=0.03)
                add_sphere_marker(viewer.user_scn, [-0.0035, -0.211, 0.0],
                                  rgba=(0.9, 0.6, 0.1, 0.5), radius=0.03)

            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
