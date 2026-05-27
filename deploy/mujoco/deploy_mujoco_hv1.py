# Copyright (c) 2026, Loco-Manip HV1 Sim-to-Sim
# SPDX-License-Identifier: BSD-3-Clause
"""
Deploy a trained HV1 loco-manip policy in MuJoCo.

Reads the YAML produced by `dump_mujoco_config.py` and runs the policy in a
MuJoCo viewer. Mirrors the Unitree `deploy_mujoco.py` pattern but adapted for
HV1 (26-d action: legs + arms, 114-d obs with EE pose commands).

Usage:
  python deploy_mujoco_hv1.py \\
      --config <run_dir>/exported/mujoco_config.yaml \\
      --cmd_lin_x 0.0 --cmd_lin_y 0.0 --cmd_ang_z 0.0 \\
      --left_ee   0.30  0.26 0.24  1.0 0.0 0.0 0.0 \\
      --right_ee  0.30 -0.26 0.24  1.0 0.0 0.0 0.0

Pipeline:
  1. Load YAML config.
  2. Load URDF in MuJoCo (URDF parses with auto free joint at root).
  3. Build joint-order maps:
       Isaac order  ↔  MuJoCo order
       Action order  →  Isaac order  →  MuJoCo order
  4. Load TorchScript policy from `policy.jit_path`.
  5. Main loop:
       Physics @ sim_dt (200 Hz). PD computed every step.
       Policy @ policy_dt (50 Hz). Action -> PD target -> torque.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


# ----------------------------------------------------------------------------
# Math helpers — quaternion conventions match Isaac Lab (w, x, y, z)
# ----------------------------------------------------------------------------

def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v from world frame to body frame.

    q: body-to-world unit quaternion (w, x, y, z)
    v: world-frame vector
    Returns body-frame vector.

    Matches isaaclab.utils.math.quat_apply_inverse.
    """
    qw = q[0]
    qvec = q[1:4]
    a = v * (2.0 * qw * qw - 1.0)
    b = np.cross(qvec, v) * (2.0 * qw)
    c = qvec * (2.0 * np.dot(qvec, v))
    return a - b + c


def projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Gravity vector [0, 0, -1] (world) rotated into body frame."""
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))


# ----------------------------------------------------------------------------
# MuJoCo helpers
# ----------------------------------------------------------------------------

def list_mujoco_joint_names(m: mujoco.MjModel) -> List[str]:
    """Return hinge/slide joint names in MuJoCo's internal order (skipping the free joint)."""
    names: List[str] = []
    for i in range(m.njnt):
        if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))
    return names


def build_joint_maps(
    isaac_joint_names: List[str],
    action_joint_names: List[str],
    mj_joint_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the three index arrays we need:

    isaac_to_mj[i]:    given Isaac index i, where is that joint in MuJoCo's order?
    mj_to_isaac[j]:    given MuJoCo index j, where is that joint in Isaac's order?
    action_to_isaac[a]:given action index a, where is that joint in Isaac's order?
    """
    # Sanity check: same joint set
    isaac_set = set(isaac_joint_names)
    mj_set = set(mj_joint_names)
    if isaac_set != mj_set:
        only_isaac = isaac_set - mj_set
        only_mj = mj_set - isaac_set
        raise RuntimeError(
            f"Joint name mismatch between Isaac and MuJoCo.\n"
            f"  In Isaac, not in MuJoCo: {sorted(only_isaac)}\n"
            f"  In MuJoCo, not in Isaac: {sorted(only_mj)}"
        )
    if set(action_joint_names) - isaac_set:
        raise RuntimeError(
            f"Action joints not in Isaac joint list: {set(action_joint_names) - isaac_set}"
        )

    isaac_to_mj = np.array([mj_joint_names.index(n) for n in isaac_joint_names], dtype=np.int64)
    mj_to_isaac = np.array([isaac_joint_names.index(n) for n in mj_joint_names], dtype=np.int64)
    action_to_isaac = np.array([isaac_joint_names.index(n) for n in action_joint_names], dtype=np.int64)
    return isaac_to_mj, mj_to_isaac, action_to_isaac


# ----------------------------------------------------------------------------
# Initial state setup
# ----------------------------------------------------------------------------

def set_default_pose(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    q_default_isaac: np.ndarray,
    isaac_to_mj: np.ndarray,
    base_height: float | None,
) -> None:
    """Place robot in standing pose.

    Preferred path: load the "home" keyframe from the MJCF (clears all internal
    buffers, sets a known-good qpos that matches the Isaac default pose).

    Fallback path: zero qpos/qvel, set base xyz+quat manually, write joint pos
    in MuJoCo order. Used only if the MJCF has no keyframe.

    qpos layout (with free joint): [x, y, z, qw, qx, qy, qz, joint_0, joint_1, ...]
    The 31 elements after position+quat are MuJoCo joint order.
    """
    # Always reset data first — clears qvel, qacc, contact buffers, etc.
    mujoco.mj_resetData(m, d)

    use_keyframe = m.nkey > 0
    if use_keyframe:
        # Load qpos from the "home" keyframe baked into hv1.xml. This matches
        # the Isaac default joint pose by construction; just override base z
        # if the user asked for a different height.
        mujoco.mj_resetDataKeyframe(m, d, 0)
        if base_height is not None:
            d.qpos[2] = base_height
        print(f"[deploy] init pose: keyframe 0 loaded, base z = {d.qpos[2]:.3f}")
    else:
        # Fallback: build qpos from YAML default.
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[2] = base_height if base_height is not None else 0.95
        d.qpos[3] = 1.0
        # argsort(isaac_to_mj) == mj_to_isaac (inverse permutation)
        q_default_mj = q_default_isaac[np.argsort(isaac_to_mj)]
        d.qpos[7 : 7 + len(q_default_mj)] = q_default_mj
        print(f"[deploy] init pose: built from YAML, base z = {d.qpos[2]:.3f}")

    mujoco.mj_forward(m, d)


# ----------------------------------------------------------------------------
# Observation builder — matches Isaac Lab HV1 loco-manip 114-d layout
# ----------------------------------------------------------------------------
#
# [0:3]    base_lin_vel        (3)  — base-frame linear vel
# [3:6]    base_ang_vel        (3)  — base-frame angular vel
# [6:9]    projected_gravity   (3)  — gravity in base frame
# [9:12]   velocity_commands   (3)  — [vx, vy, wz]
# [12:43]  joint_pos_rel       (31) — q - q_default, Isaac order
# [43:74]  joint_vel           (31) — Isaac order
# [74:100] last_action         (26) — action order
# [100:107] left_ee_pose_cmd   (7)  — [px, py, pz, qw, qx, qy, qz] in pelvis frame
# [107:114] right_ee_pose_cmd  (7)

def build_observation(
    d: mujoco.MjData,
    isaac_to_mj: np.ndarray,
    q_default_isaac: np.ndarray,
    last_action: np.ndarray,
    vel_cmd: np.ndarray,
    left_ee_cmd: np.ndarray,
    right_ee_cmd: np.ndarray,
) -> np.ndarray:
    # base state — MuJoCo freejoint convention:
    #   qvel[0:3]  linear velocity in WORLD frame    → rotate into body frame
    #   qvel[3:6]  angular velocity in BODY frame    → use directly (do NOT rotate)
    # Isaac Lab's `base_lin_vel` and `base_ang_vel` observations are both in
    # body frame (root_lin_vel_b / root_ang_vel_b), so matching that here.
    base_quat = d.qpos[3:7].astype(np.float32).copy()    # (w, x, y, z)
    lin_vel_w = d.qvel[0:3].astype(np.float32)
    lin_vel_b = quat_rotate_inverse(base_quat, lin_vel_w)
    ang_vel_b = d.qvel[3:6].astype(np.float32).copy()
    grav_b = projected_gravity(base_quat)

    # Joints: convert from MJ order to Isaac order.
    # isaac_to_mj[i] = MJ index of the joint that lives at Isaac index i.
    # So q_isaac[i] = q_mj[isaac_to_mj[i]] = q_mj[isaac_to_mj].
    q_mj = d.qpos[7:].astype(np.float32)
    dq_mj = d.qvel[6:].astype(np.float32)
    q_isaac = q_mj[isaac_to_mj]
    dq_isaac = dq_mj[isaac_to_mj]
    joint_pos_rel = q_isaac - q_default_isaac

    obs = np.concatenate(
        [
            lin_vel_b,            # 3
            ang_vel_b,            # 3
            grav_b,               # 3
            vel_cmd,              # 3
            joint_pos_rel,        # 31
            dq_isaac,             # 31
            last_action,          # 26
            left_ee_cmd,          # 7
            right_ee_cmd,         # 7
        ]
    ).astype(np.float32)
    return obs


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deploy HV1 loco-manip policy in MuJoCo.")
    parser.add_argument("--config", type=str, required=True, help="Path to mujoco_config.yaml")
    parser.add_argument("--policy", type=str, default=None, help="Override policy.pt path (default: from YAML).")
    parser.add_argument("--xml", type=str, default=None, help="Override MJCF/XML path (default: from YAML).")
    parser.add_argument("--duration", type=float, default=120.0, help="Simulation duration in seconds.")
    parser.add_argument(
        "--base_height",
        type=float,
        default=None,
        help="Override initial base height (m). Default: use 'home' keyframe from MJCF.",
    )
    # commands
    parser.add_argument("--cmd_lin_x", type=float, default=0.0)
    parser.add_argument("--cmd_lin_y", type=float, default=0.0)
    parser.add_argument("--cmd_ang_z", type=float, default=0.0)
    parser.add_argument(
        "--left_ee",
        nargs=7,
        type=float,
        default=[0.30, 0.26, 0.24, 1.0, 0.0, 0.0, 0.0],
        help="Left EE pose in pelvis frame: px py pz qw qx qy qz",
    )
    parser.add_argument(
        "--right_ee",
        nargs=7,
        type=float,
        default=[0.30, -0.26, 0.24, 1.0, 0.0, 0.0, 0.0],
        help="Right EE pose in pelvis frame: px py pz qw qx qy qz",
    )
    parser.add_argument("--realtime", action="store_true", default=True, help="Sleep to match wall clock.")
    args = parser.parse_args()

    # ---- Load config ------------------------------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    xml_path = args.xml or cfg["robot"]["xml_path"]
    policy_path = args.policy or cfg["policy"]["jit_path"]

    sim_dt = float(cfg["control"]["sim_dt"])
    decimation = int(cfg["control"]["decimation"])

    joint_names_isaac = cfg["robot"]["joint_names_isaac_order"]
    joint_names_action = cfg["action"]["joint_names_action_order"]
    q_default_isaac = np.array(cfg["robot"]["default_joint_pos"], dtype=np.float32)
    kp_isaac = np.array(cfg["robot"]["kp"], dtype=np.float32)
    kd_isaac = np.array(cfg["robot"]["kd"], dtype=np.float32)
    action_scale = float(cfg["action"]["scale"])
    use_default_offset = bool(cfg["action"]["use_default_offset"])
    n_dof = int(cfg["robot"]["num_dof_total"])
    action_dim = int(cfg["action"]["dim"])

    print(f"[deploy] config        : {args.config}")
    print(f"[deploy] xml           : {xml_path}")
    print(f"[deploy] policy        : {policy_path}")
    print(f"[deploy] sim_dt        : {sim_dt}  decimation={decimation}  policy_dt={sim_dt*decimation}")
    print(f"[deploy] n_dof_total   : {n_dof}    action_dim={action_dim}")
    print(f"[deploy] action_scale  : {action_scale}    use_default_offset={use_default_offset}")

    # ---- Load MuJoCo model ------------------------------------------------
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    mj_joint_names = list_mujoco_joint_names(m)
    assert len(mj_joint_names) == n_dof, (
        f"MuJoCo has {len(mj_joint_names)} hinge/slide joints, YAML says {n_dof}."
    )

    isaac_to_mj, mj_to_isaac, action_to_isaac = build_joint_maps(
        joint_names_isaac, joint_names_action, mj_joint_names
    )

    print("[deploy] joint maps built — Isaac/MuJoCo joint sets match.")

    # MuJoCo-ordered Kp/Kd for PD law
    kp_mj = kp_isaac[mj_to_isaac]
    kd_mj = kd_isaac[mj_to_isaac]
    q_default_mj = q_default_isaac[mj_to_isaac]

    # ---- Torque limits from MJCF actuator force range ---------------------
    # Each motor in hv1.xml has ctrlrange="-X X" (X = joint motor's hard limit).
    # We use these to clamp tau before writing to qfrc_applied — without this,
    # a single large PD error can produce kilonewton-meter torques and blow up
    # the simulator (NaN qpos → invisible robot in viewer).
    tau_limit_mj = np.zeros(n_dof, dtype=np.float32)
    for j_mj, name in enumerate(mj_joint_names):
        # find actuator that drives this joint
        for a in range(m.nu):
            if m.actuator_trnid[a, 0] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name):
                lo, hi = m.actuator_ctrlrange[a]
                tau_limit_mj[j_mj] = max(abs(lo), abs(hi))
                break
        else:
            # no actuator → unbounded (shouldn't happen for HV1 — 31/31 covered)
            tau_limit_mj[j_mj] = 1e6

    print(f"[deploy] torque clamps  : min={tau_limit_mj.min():.1f}  max={tau_limit_mj.max():.1f} Nm")

    # ---- Load policy ------------------------------------------------------
    policy = torch.jit.load(policy_path, map_location="cpu").eval()
    print(f"[deploy] policy loaded ({sum(p.numel() for p in policy.parameters()):,} params)")

    # ---- Initial state ----------------------------------------------------
    set_default_pose(m, d, q_default_isaac, isaac_to_mj, args.base_height)

    # Sanity-check the initial pose: warn if the lowest geom is below z=0 (foot
    # in ground → contact explosion on first step).
    lowest_z = float("inf")
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            continue  # skip world geoms (floor, etc.)
        lowest_z = min(lowest_z, float(d.geom_xpos[g, 2]))
    print(f"[deploy] init lowest geom z = {lowest_z:.4f}  (negative ⇒ ground penetration)")

    # state buffers
    last_action_isaac_order = np.zeros(action_dim, dtype=np.float32)
    # Initial PD target = default pose (so robot holds itself before first policy step)
    target_dof_pos_mj = q_default_mj.copy()

    vel_cmd = np.array([args.cmd_lin_x, args.cmd_lin_y, args.cmd_ang_z], dtype=np.float32)
    left_ee_cmd = np.array(args.left_ee, dtype=np.float32)
    right_ee_cmd = np.array(args.right_ee, dtype=np.float32)

    print(f"[deploy] cmd_velocity  : {vel_cmd.tolist()}")
    print(f"[deploy] left_ee_cmd   : {left_ee_cmd.tolist()}")
    print(f"[deploy] right_ee_cmd  : {right_ee_cmd.tolist()}")
    print(f"[deploy] starting viewer …")

    counter = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < args.duration:
            step_start = time.time()

            # ---- PD control every physics step ----
            q_mj = d.qpos[7:]
            dq_mj = d.qvel[6:]
            tau = kp_mj * (target_dof_pos_mj - q_mj) - kd_mj * dq_mj
            # Clamp to per-joint actuator torque limits — prevents blow-up
            tau = np.clip(tau, -tau_limit_mj, tau_limit_mj)
            # Write joint torques. qfrc_applied is in qvel layout: [6 free dofs, 31 joints]
            d.qfrc_applied[6:] = tau

            mujoco.mj_step(m, d)
            counter += 1

            # ---- NaN guard ---------------------------------------------------
            # Mesh self-collisions or extreme torques can drive qpos to NaN,
            # which renders as the robot disappearing in the viewer. Catch it.
            if not np.all(np.isfinite(d.qpos)):
                print(f"[deploy] !! qpos went NaN at step {counter} — stopping.")
                print(f"[deploy]    base z just before NaN: {d.qpos[2]:.4f}")
                print(f"[deploy]    last tau abs max: {np.abs(tau).max():.2f}")
                break

            # ---- Policy every `decimation` steps ----
            if counter % decimation == 0:
                obs = build_observation(
                    d=d,
                    isaac_to_mj=isaac_to_mj,
                    q_default_isaac=q_default_isaac,
                    last_action=last_action_isaac_order,
                    vel_cmd=vel_cmd,
                    left_ee_cmd=left_ee_cmd,
                    right_ee_cmd=right_ee_cmd,
                )
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).unsqueeze(0)
                    action_t = policy(obs_t)
                action_isaac = action_t.detach().numpy().squeeze(0).astype(np.float32)
                last_action_isaac_order = action_isaac.copy()

                # Build full PD target in Isaac order: start from default,
                # overwrite the 26 actioned joints with q_default + scale * action.
                target_dof_pos_isaac = q_default_isaac.copy()
                if use_default_offset:
                    target_dof_pos_isaac[action_to_isaac] = (
                        q_default_isaac[action_to_isaac] + action_scale * action_isaac
                    )
                else:
                    target_dof_pos_isaac[action_to_isaac] = action_scale * action_isaac
                # Convert to MuJoCo order for PD law
                target_dof_pos_mj = target_dof_pos_isaac[mj_to_isaac]

                # Periodic heartbeat so the user can see policy is running.
                policy_steps = counter // decimation
                if policy_steps % 50 == 1:  # every ~1s at 50 Hz policy
                    print(
                        f"[deploy] t={d.time:6.2f}s  base_z={d.qpos[2]:.3f}  "
                        f"|action|={np.abs(action_isaac).max():.2f}  "
                        f"|tau|={np.abs(tau).max():.1f}"
                    )

            viewer.sync()

            # ---- Wall-clock pacing ----
            if args.realtime:
                slack = sim_dt - (time.time() - step_start)
                if slack > 0:
                    time.sleep(slack)

    print(f"[deploy] simulation ended after {time.time() - start:.1f}s, {counter} physics steps")


if __name__ == "__main__":
    sys.exit(main())
