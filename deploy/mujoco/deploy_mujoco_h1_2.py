# Copyright (c) 2026, H1_2 legs-only Sim-to-Sim
# SPDX-License-Identifier: BSD-3-Clause
"""
Deploy a trained H1_2 legs-only velocity-tracking policy in MuJoCo for
sim-to-sim validation (task: Isaac-Velocity-Flat-Legs-H1_2-v0).

This is the tahiti_c1 / hv1_2 runner adapted for H1_2's arm-aware, legs-only
control contract. Two structural differences from those 12-DoF-total robots:

  1. The MuJoCo model has 27 DoF (12 legs + torso + 14 arm joints), but the
     policy only ACTS on the 12 legs. The 15 upper-body joints are held at
     their default pose by the same PD loop (target = default, constant) —
     exactly matching Isaac, where only `joint_pos` (legs) is an action term
     and the upper body stays at default.

  2. The observation is 8 terms, not 6. Isaac's PolicyCfg order is:
        base_ang_vel(3), projected_gravity(3), velocity_commands(3),
        joint_pos(12 legs), joint_vel(12 legs), actions(12),
        upper_body_joint_pos(15), upper_body_joint_vel(15)   -> 75 total
     The runner reads all 8 term names + dims from the dumped YAML and builds
     this exact concatenation. joint_pos/joint_vel slice the LEG subset; the
     two upper-body terms slice the UPPER subset. Both subsets are derived from
     joint names (articulation order), so they match Isaac's SceneEntityCfg
     resolution (preserve_order=False -> ascending / articulation order).

Torque clamp: H1_2's MJCF <motor> actuators declare no ctrlrange, so reading
actuator_ctrlrange would give 0 and zero out every torque. This runner clamps
from the YAML `effort_limit` (the real per-joint motor spec Isaac trains with),
falling back to the MJCF ctrlrange, then to 1e6 (no clamp).

YAML produced by (run on the machine with the trained checkpoint + Isaac):
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/dump_mujoco_config.py \\
      --task Isaac-Velocity-Flat-Legs-H1_2-Play-v0 --num_envs 1

The MJCF (h1_2_handless.xml) is already a complete, self-contained scene
(skybox, groundplane, floor, lights, IMU sensors) — unlike the bare
tahiti/hv1_2 robot XMLs, it needs NO scene.xml wrapper, so --xml points
straight at it (this is also the runner's default).

Usage:
  python deploy/mujoco/deploy_mujoco_h1_2.py \\
      --config <run>/exported/mujoco_config.yaml --no_gamepad --cmd_lin_x 0.5

  # With Xbox gamepad (left stick = vx/wz, right stick horizontal = vy):
  python deploy/mujoco/deploy_mujoco_h1_2.py \\
      --config <run>/exported/mujoco_config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
try:
    from gamepad import GamepadReader  # noqa: E402
    _GAMEPAD_IMPORT_ERR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover — pygame not installed etc.
    GamepadReader = None  # type: ignore[assignment]
    _GAMEPAD_IMPORT_ERR = _e


# ---------------------------------------------------------------------------
# Math — matches Isaac Lab quaternion convention (w, x, y, z)
# ---------------------------------------------------------------------------

def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector v into body frame using body-to-world quat q."""
    qw = q[0]
    qvec = q[1:4]
    a = v * (2.0 * qw * qw - 1.0)
    b = np.cross(qvec, v) * (2.0 * qw)
    c = qvec * (2.0 * np.dot(qvec, v))
    return a - b + c


def projected_gravity_in_frame(quat: np.ndarray) -> np.ndarray:
    """Gravity unit vector rotated into the body frame defined by `quat`."""
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------------

def list_mujoco_joint_names(m: mujoco.MjModel) -> List[str]:
    """Return MuJoCo joint names in qpos[7:]/qvel[6:] order (free joint excluded)."""
    names: List[str] = []
    for i in range(m.njnt):
        if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))
    return names


class FootMetrics:
    """Per-foot air-time, knee-swing-amplitude, and contact-force diagnostics for
    L/R asymmetry (identical to the tahiti runner; body weight auto-read from the
    model so the xBW ratios are correct for H1_2's heavier mass)."""

    def __init__(
        self,
        m: mujoco.MjModel,
        foot_body_names: Tuple[str, str],
        knee_isaac_indices: Tuple[int, int],
        body_weight_N: float,
        window: int = 6,
    ):
        self.body_ids = tuple(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) for name in foot_body_names
        )
        self.knee_isaac_indices = knee_isaac_indices
        self.body_weight_N = body_weight_N
        self.window = window
        self.in_contact = [True, True]
        self.last_liftoff_t = [0.0, 0.0]
        self.airtimes = [[], []]
        self.swing_min_knee = [1e9, 1e9]
        self.swing_max_knee = [-1e9, -1e9]
        self.knee_amplitudes = [[], []]
        self.peak_force = [0.0, 0.0]
        self.contact_force_samples = [[], []]

    @staticmethod
    def _push(buf: List[float], value: float, window: int) -> None:
        buf.append(value)
        if len(buf) > window:
            buf.pop(0)

    def update(self, m: mujoco.MjModel, d: mujoco.MjData, q_isaac: np.ndarray) -> None:
        contact_now = [False, False]
        step_force = [0.0, 0.0]
        force6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = m.geom_bodyid[c.geom1]
            b2 = m.geom_bodyid[c.geom2]
            for foot_idx, bid in enumerate(self.body_ids):
                if b1 == bid or b2 == bid:
                    contact_now[foot_idx] = True
                    mujoco.mj_contactForce(m, d, i, force6)
                    step_force[foot_idx] += abs(float(force6[0]))
        for foot_idx in range(2):
            if step_force[foot_idx] > 0.0:
                if step_force[foot_idx] > self.peak_force[foot_idx]:
                    self.peak_force[foot_idx] = step_force[foot_idx]
                self._push(self.contact_force_samples[foot_idx], step_force[foot_idx], self.window * 20)

        for foot_idx in range(2):
            knee_val = float(q_isaac[self.knee_isaac_indices[foot_idx]])
            if not contact_now[foot_idx]:
                self.swing_min_knee[foot_idx] = min(self.swing_min_knee[foot_idx], knee_val)
                self.swing_max_knee[foot_idx] = max(self.swing_max_knee[foot_idx], knee_val)
            was = self.in_contact[foot_idx]
            now = contact_now[foot_idx]
            if was and not now:
                self.last_liftoff_t[foot_idx] = float(d.time)
                self.swing_min_knee[foot_idx] = knee_val
                self.swing_max_knee[foot_idx] = knee_val
            elif (not was) and now:
                airtime = float(d.time) - self.last_liftoff_t[foot_idx]
                if 0.05 < airtime < 2.0:
                    self._push(self.airtimes[foot_idx], airtime, self.window)
                    amp = self.swing_max_knee[foot_idx] - self.swing_min_knee[foot_idx]
                    self._push(self.knee_amplitudes[foot_idx], amp, self.window)
            self.in_contact[foot_idx] = now

    def summary(self) -> str:
        def _mean(buf: List[float]) -> float:
            return sum(buf) / len(buf) if buf else 0.0

        atL, atR = _mean(self.airtimes[0]), _mean(self.airtimes[1])
        kaL, kaR = _mean(self.knee_amplitudes[0]), _mean(self.knee_amplitudes[1])
        nL, nR = len(self.airtimes[0]), len(self.airtimes[1])
        at_asym = (atL - atR) / max(atL + atR, 1e-6) * 100.0
        ka_asym = (kaL - kaR) / max(kaL + kaR, 1e-6) * 100.0
        fL_peak, fR_peak = self.peak_force[0], self.peak_force[1]
        fL_mean = _mean(self.contact_force_samples[0])
        fR_mean = _mean(self.contact_force_samples[1])
        bw = self.body_weight_N
        return (
            f"air_time  L={atL:.3f}s R={atR:.3f}s  asym={at_asym:+.1f}%  (n={nL}/{nR})  "
            f"knee_amp  L={kaL:.3f} R={kaR:.3f}  asym={ka_asym:+.1f}%\n"
            f"force     L peak={fL_peak:.0f}N ({fL_peak/bw:.2f}xBW) mean={fL_mean:.0f}N   "
            f"R peak={fR_peak:.0f}N ({fR_peak/bw:.2f}xBW) mean={fR_mean:.0f}N"
        )


def build_joint_maps(
    isaac_joint_names: List[str],
    action_joint_names: List[str],
    mj_joint_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permutations between Isaac and MuJoCo joint orderings.

    isaac_to_mj : q_mj[isaac_to_mj] -> Isaac order.
    mj_to_isaac : kp_isaac[mj_to_isaac] -> MuJoCo order.
    action_to_isaac : policy-action index -> Isaac-joint index.
    """
    isaac_set = set(isaac_joint_names)
    mj_set = set(mj_joint_names)
    if isaac_set != mj_set:
        only_isaac = isaac_set - mj_set
        only_mj = mj_set - isaac_set
        raise RuntimeError(
            f"Joint name mismatch.\n"
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


def effort_limits_isaac(
    cfg: dict,
    m: mujoco.MjModel,
    mj_joint_names: List[str],
    mj_to_isaac: np.ndarray,
) -> np.ndarray:
    """Per-joint torque clamp in Isaac order.

    Priority: YAML effort_limit (Isaac's real motor spec) > MJCF actuator
    ctrlrange > 1e6 (no clamp). H1_2's MJCF <motor>s have no ctrlrange, so the
    YAML path is what keeps torque non-zero.
    """
    n = len(mj_joint_names)
    eff_isaac = np.full(n, 1e6, dtype=np.float32)
    yaml_eff = cfg["robot"].get("effort_limit")
    if yaml_eff is not None and all(v is not None for v in yaml_eff):
        eff_isaac = np.asarray(yaml_eff, dtype=np.float32)
        # Guard against a 0/negative that would kill torque.
        eff_isaac = np.where(eff_isaac > 0.0, eff_isaac, 1e6).astype(np.float32)
        return eff_isaac
    # Fallback: MJCF ctrlrange (in MuJoCo order) -> Isaac order.
    eff_mj = np.zeros(n, dtype=np.float32)
    for j_mj, name in enumerate(mj_joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        matched = False
        for a in range(m.nu):
            if m.actuator_trnid[a, 0] == jid:
                lo, hi = m.actuator_ctrlrange[a]
                lim = max(abs(lo), abs(hi))
                eff_mj[j_mj] = lim if lim > 0 else 1e6
                matched = True
                break
        if not matched:
            eff_mj[j_mj] = 1e6
    # mj order -> isaac order
    eff_isaac = np.empty(n, dtype=np.float32)
    eff_isaac[mj_to_isaac] = eff_mj
    return eff_isaac


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def set_default_pose(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    q_default_isaac: np.ndarray,
    mj_to_isaac: np.ndarray,
    base_height: float,
) -> None:
    """Init MuJoCo state: all 27 joints at the Isaac default pose, base at base_height."""
    mujoco.mj_resetData(m, d)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[2] = base_height
        print(f"[deploy] init pose: keyframe 0 + base z override = {d.qpos[2]:.3f}")
    else:
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[2] = base_height
        d.qpos[3] = 1.0  # qw
        q_default_mj = q_default_isaac[mj_to_isaac]
        d.qpos[7 : 7 + len(q_default_mj)] = q_default_mj
        print(f"[deploy] init pose: built from YAML defaults, base z = {d.qpos[2]:.3f}")
    mujoco.mj_forward(m, d)


# ---------------------------------------------------------------------------
# Observation builder — single-frame, no history, arm-aware (8 terms)
# ---------------------------------------------------------------------------

class H1_2ObsBuilder:
    """Builds the single-frame 75-dim observation for the H1_2 legs-only policy.

    Order matches Isaac Lab's PolicyCfg declaration order:
      base_ang_vel, projected_gravity, velocity_commands,
      joint_pos(legs), joint_vel(legs), actions,
      upper_body_joint_pos, upper_body_joint_vel.

    joint_pos/joint_vel emit the LEG subset; the upper-body terms emit the
    UPPER subset. Both subsets index a full-27 Isaac-order vector, so they
    reproduce Isaac's SceneEntityCfg(joint_names=...) resolution exactly.
    """

    def __init__(
        self,
        m: mujoco.MjModel,
        d: mujoco.MjData,
        isaac_to_mj: np.ndarray,
        q_default_isaac: np.ndarray,
        leg_isaac_indices: np.ndarray,
        upper_isaac_indices: np.ndarray,
        action_dim: int,
        obs_term_order: List[str],
        history_length: int = 1,
    ):
        self.m = m
        self.d = d
        self.isaac_to_mj = isaac_to_mj
        self.q_default_isaac = q_default_isaac.astype(np.float32).copy()
        self.leg_isaac_indices = leg_isaac_indices
        self.upper_isaac_indices = upper_isaac_indices
        self.obs_term_order = obs_term_order
        self.last_action = np.zeros(action_dim, dtype=np.float32)
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.history_length = int(history_length)

        self._term_fns = {
            "base_ang_vel": self._base_ang_vel,
            "projected_gravity": self._projected_gravity,
            "velocity_commands": self._velocity_commands,
            "joint_pos": self._joint_pos,
            "joint_vel": self._joint_vel,
            "actions": self._actions,
            "upper_body_joint_pos": self._upper_body_joint_pos,
            "upper_body_joint_vel": self._upper_body_joint_vel,
        }
        missing = [t for t in obs_term_order if t not in self._term_fns]
        if missing:
            raise RuntimeError(f"No compute function for obs terms: {missing}")
        self._history: dict[str, list[np.ndarray]] = {t: [] for t in obs_term_order}

    # --- full-27 helpers in Isaac order ---
    def _q_isaac(self) -> np.ndarray:
        return self.d.qpos[7:].astype(np.float32)[self.isaac_to_mj]

    def _dq_isaac(self) -> np.ndarray:
        return self.d.qvel[6:].astype(np.float32)[self.isaac_to_mj]

    # --- terms ---
    def _base_ang_vel(self) -> np.ndarray:
        return self.d.qvel[3:6].astype(np.float32).copy()

    def _projected_gravity(self) -> np.ndarray:
        base_quat = self.d.qpos[3:7].astype(np.float32)
        return projected_gravity_in_frame(base_quat)

    def _velocity_commands(self) -> np.ndarray:
        return self.vel_cmd

    def _joint_pos(self) -> np.ndarray:
        rel = self._q_isaac() - self.q_default_isaac
        return rel[self.leg_isaac_indices]

    def _joint_vel(self) -> np.ndarray:
        return self._dq_isaac()[self.leg_isaac_indices]

    def _actions(self) -> np.ndarray:
        return self.last_action

    def _upper_body_joint_pos(self) -> np.ndarray:
        rel = self._q_isaac() - self.q_default_isaac
        return rel[self.upper_isaac_indices]

    def _upper_body_joint_vel(self) -> np.ndarray:
        return self._dq_isaac()[self.upper_isaac_indices]

    def step(self) -> np.ndarray:
        for name in self.obs_term_order:
            current = self._term_fns[name]().astype(np.float32)
            buf = self._history[name]
            if not buf:
                for _ in range(self.history_length):
                    buf.append(current.copy())
            else:
                buf.append(current)
                if len(buf) > self.history_length:
                    buf.pop(0)
        parts: List[np.ndarray] = []
        for name in self.obs_term_order:
            for frame in self._history[name]:
                parts.append(frame)
        return np.concatenate(parts, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Deploy H1_2 legs-only velocity policy in MuJoCo (sim-to-sim)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to mujoco_config.yaml.")
    parser.add_argument("--policy_format", choices=["pt", "onnx"], default="onnx",
                        help="Inference backend. 'onnx' (default) or 'pt' (TorchScript).")
    parser.add_argument("--policy", type=str, default=None,
                        help="Override policy file path (else YAML default for the chosen format).")
    parser.add_argument("--xml", type=str,
                        default="/home/rabisankar/IsaacLab/source/isaaclab_assets/data/"
                                "custom_robot/urdf_mesh/h1_2_hand_less/h1_2_handless.xml",
                        help="MJCF file. h1_2_handless.xml is already a full scene "
                             "(ground/lights/skybox), so it is loaded directly.")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Sim duration in seconds. 0 = run until viewer closed.")
    parser.add_argument("--base_height", type=float, default=1.03,
                        help="Initial pelvis z (m). H1_2 straight-leg ceiling ~1.028, "
                             "default-pose ~1.015; 1.03 drops in cleanly.")
    parser.add_argument("--realtime", action="store_true", default=True,
                        help="Sleep between steps so sim runs at wall-clock speed.")
    parser.add_argument("--headless", action="store_true",
                        help="Run without the MuJoCo viewer (for servers / validation). "
                             "Prints the same heartbeat; --duration defaults to 10s.")
    parser.add_argument("--settle", type=float, default=0.5,
                        help="Seconds to hold the default pose before engaging the policy, "
                             "so the drop-in transient decays first (0 to disable).")
    parser.add_argument("--viewer_sync_every", type=int, default=4,
                        help="Call viewer.sync() every N physics steps (default 4 = policy rate).")
    parser.add_argument("--cmd_lin_x", type=float, default=0.0,
                        help="Fallback forward velocity when --no_gamepad (m/s).")
    parser.add_argument("--cmd_lin_y", type=float, default=0.0,
                        help="Fallback sideways velocity when --no_gamepad (m/s).")
    parser.add_argument("--cmd_ang_z", type=float, default=0.0,
                        help="Fallback yaw rate when --no_gamepad (rad/s).")

    # -- Gamepad options (defaults = H1_2 training command ranges) --
    parser.add_argument("--no_gamepad", action="store_true",
                        help="Disable the Xbox gamepad; drive from --cmd_lin_x/y/z.")
    parser.add_argument("--vx_max", type=float, default=1.0,
                        help="Forward-velocity at full stick (m/s). Match lin_vel_x range.")
    parser.add_argument("--vy_max", type=float, default=0.5,
                        help="Lateral-velocity at full stick (m/s). Match lin_vel_y range.")
    parser.add_argument("--wz_max", type=float, default=0.5,
                        help="Yaw-rate at full stick (rad/s). Match ang_vel_z range.")
    parser.add_argument("--deadzone", type=float, default=0.1, help="Stick deadzone in [0,1].")
    parser.add_argument("--gamepad_index", type=int, default=0, help="pygame joystick index.")

    # -- Actuator delay (match Isaac DelayedPDActuatorCfg if used) --
    parser.add_argument("--delay_min", type=int, default=0,
                        help="Min per-joint actuator delay in physics steps.")
    parser.add_argument("--delay_max", type=int, default=0,
                        help="Max per-joint actuator delay in physics steps. Set to match "
                             "the DelayedPDActuatorCfg range if H1_2_CFG uses one (else 0).")
    parser.add_argument("--delay_seed", type=int, default=None,
                        help="Seed for the per-joint delay sample (default: unseeded).")
    parser.add_argument("--explicit_pd", action="store_true",
                        help="Legacy PD: apply -kd*qvel explicitly in the loop with Euler "
                             "integration (as the tahiti/hv1_2 runner does). UNSTABLE for "
                             "H1_2's low armature (0.01) — leave off to use the default "
                             "implicitfast + damping-in-model path that is stable at 0.01.")
    args = parser.parse_args()

    # ---- Load config ------------------------------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    xml_path = args.xml
    if args.policy_format == "onnx":
        policy_path = args.policy or cfg["policy"].get("onnx_path")
    else:
        policy_path = args.policy or cfg["policy"].get("jit_path")
    if not policy_path:
        raise RuntimeError(
            f"No policy path for format '{args.policy_format}'. Pass --policy or check YAML."
        )
    sim_dt = float(cfg["control"]["sim_dt"])
    decimation = int(cfg["control"]["decimation"])
    joint_names_isaac = cfg["robot"]["joint_names_isaac_order"]
    joint_names_action = cfg["action"]["joint_names_action_order"]
    q_default_isaac = np.array(cfg["robot"]["default_joint_pos"], dtype=np.float32)
    kp_isaac = np.array(cfg["robot"]["kp"], dtype=np.float32)
    kd_isaac = np.array(cfg["robot"]["kd"], dtype=np.float32)
    # Joint physics Isaac trains with (recorded by the dump). H1_2's MJCF sets
    # NONE of these (armature/damping/frictionloss all 0), which makes the PD
    # explode (see below). Default to 0.0 arrays if a field is absent/None.
    def _phys(field):
        v = cfg["robot"].get(field)
        if v is None or any(x is None for x in v):
            return np.zeros(len(kp_isaac), dtype=np.float32)
        return np.asarray(v, dtype=np.float32)
    armature_isaac = _phys("armature")
    friction_isaac = _phys("friction")            # Coulomb -> dof_frictionloss
    viscous_isaac = _phys("viscous_friction")     # passive viscous -> dof_damping
    action_scale_raw = cfg["action"]["scale"]
    if isinstance(action_scale_raw, (list, tuple)):
        action_scale = np.asarray(action_scale_raw, dtype=np.float32)
    else:
        action_scale = float(action_scale_raw)
    use_default_offset = bool(cfg["action"]["use_default_offset"])
    n_dof = int(cfg["robot"]["num_dof_total"])
    action_dim = int(cfg["action"]["dim"])
    total_obs_dim_yaml = int(cfg["observation"]["total_dim"])
    obs_history_length = int(cfg["observation"].get("history_length", 1))
    obs_terms_yaml = cfg["observation"]["terms"]
    obs_term_order = [entry["name"] for entry in obs_terms_yaml]

    print(f"[deploy] config            : {args.config}")
    print(f"[deploy] xml               : {xml_path}")
    print(f"[deploy] policy ({args.policy_format})       : {policy_path}")
    print(f"[deploy] sim_dt            : {sim_dt}  decimation={decimation}  policy_dt={sim_dt * decimation}")
    print(f"[deploy] n_dof_total       : {n_dof}    action_dim={action_dim}")
    print(f"[deploy] action_scale      : {action_scale}    use_default_offset={use_default_offset}")
    print(f"[deploy] total_obs_dim     : {total_obs_dim_yaml}  (from YAML)")
    print(f"[deploy] obs term order    : {obs_term_order}")

    # ---- Load MuJoCo model -----------------------------------------------
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt
    # Implicit-in-velocity integration. The PD damping is applied via the model
    # (dof_damping, integrated implicitly) rather than as an explicit -kd*qvel
    # force. At H1_2's low armature (0.01) the explicit form is numerically
    # unstable (kd*dt/inertia = 5*0.005/0.01 = 2.5 > 1) and the PD blows up in a
    # few steps -> MuJoCo auto-resets every blow-up -> the viewer shows a frozen/
    # flickering robot. implicitfast integrates the damping stably.
    if not args.explicit_pd:
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    mj_joint_names = list_mujoco_joint_names(m)
    assert len(mj_joint_names) == n_dof, (
        f"MuJoCo has {len(mj_joint_names)} hinge/slide joints, YAML says {n_dof}."
    )

    isaac_to_mj, mj_to_isaac, action_to_isaac = build_joint_maps(
        joint_names_isaac, joint_names_action, mj_joint_names
    )
    print("[deploy] joint maps built — Isaac/MuJoCo joint sets match.")

    # ---- Apply Isaac joint physics to the MuJoCo model -------------------
    # CRITICAL: h1_2_handless.xml sets NO armature/damping/frictionloss on its
    # joints (all 0). Isaac trains with armature=0.01 etc (recorded in the YAML).
    # Without this, the robot is a different dynamical system and the PD is
    # unstable. dof_* arrays are indexed [free(6) | hinges...] in MuJoCo order.
    m.dof_armature[6:] = armature_isaac[mj_to_isaac]
    m.dof_frictionloss[6:] = friction_isaac[mj_to_isaac]
    if args.explicit_pd:
        # Legacy path: explicit -kd*qvel in the loop; only passive viscous in model.
        m.dof_damping[6:] = viscous_isaac[mj_to_isaac]
    else:
        # Implicit path: total damping (active PD kd + passive viscous) in model.
        m.dof_damping[6:] = kd_isaac[mj_to_isaac] + viscous_isaac[mj_to_isaac]
    print(f"[deploy] joint physics     : armature[{m.dof_armature[6:].min():.3f},"
          f"{m.dof_armature[6:].max():.3f}]  dof_damping[{m.dof_damping[6:].min():.1f},"
          f"{m.dof_damping[6:].max():.1f}]  integrator="
          f"{'explicit-PD/Euler' if args.explicit_pd else 'implicitfast+damp-in-model'}")

    # Leg / upper-body subsets, in Isaac (articulation) order — this exactly
    # matches how Isaac's obs SceneEntityCfg(joint_names=...) resolves each term
    # (preserve_order=False => ascending index => articulation order).
    leg_name_set = set(joint_names_action)  # the 12 leg joints (action subset)
    leg_isaac_indices = np.array(
        [i for i, n in enumerate(joint_names_isaac) if n in leg_name_set], dtype=np.int64
    )
    upper_isaac_indices = np.array(
        [i for i, n in enumerate(joint_names_isaac) if n not in leg_name_set], dtype=np.int64
    )
    print(f"[deploy] leg joints   ({len(leg_isaac_indices)}): "
          f"{[joint_names_isaac[i] for i in leg_isaac_indices]}")
    print(f"[deploy] upper joints ({len(upper_isaac_indices)}): "
          f"{[joint_names_isaac[i] for i in upper_isaac_indices]}")

    # PD gains and defaults in MuJoCo order.
    kp_mj = kp_isaac[mj_to_isaac]
    kd_mj = kd_isaac[mj_to_isaac]
    q_default_mj = q_default_isaac[mj_to_isaac]

    # Per-joint torque clamp (Isaac order) from YAML effort_limit (see docstring).
    eff_isaac = effort_limits_isaac(cfg, m, mj_joint_names, mj_to_isaac)
    tau_limit_mj = eff_isaac[mj_to_isaac].astype(np.float32)
    print(f"[deploy] kp range          : [{kp_isaac.min():.1f}, {kp_isaac.max():.1f}]")
    print(f"[deploy] kd range          : [{kd_isaac.min():.2f}, {kd_isaac.max():.2f}]")
    print(f"[deploy] torque clamps     : min={tau_limit_mj.min():.1f}  max={tau_limit_mj.max():.1f} Nm "
          f"(source: {'YAML effort_limit' if cfg['robot'].get('effort_limit') else 'MJCF/fallback'})")

    # ---- Load policy -----------------------------------------------------
    if args.policy_format == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime not installed. `pip install onnxruntime` or use --policy_format pt."
            ) from e
        session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        onnx_input_name = session.get_inputs()[0].name
        onnx_output_name = session.get_outputs()[0].name
        print(f"[deploy] onnx loaded       : input='{onnx_input_name}'  output='{onnx_output_name}'")

        def run_policy(obs_np: np.ndarray) -> np.ndarray:
            out = session.run([onnx_output_name],
                              {onnx_input_name: obs_np[np.newaxis, :].astype(np.float32)})
            return out[0].squeeze(0).astype(np.float32)
    else:
        policy = torch.jit.load(policy_path, map_location="cpu").eval()
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"[deploy] policy loaded     : ({n_params:,} params)")

        def run_policy(obs_np: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_np).unsqueeze(0)
                action_t = policy(obs_t)
            return action_t.detach().numpy().squeeze(0).astype(np.float32)

    # ---- Initial state ---------------------------------------------------
    set_default_pose(m, d, q_default_isaac, mj_to_isaac, args.base_height)

    lowest_z = float("inf")
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            continue
        lowest_z = min(lowest_z, float(d.geom_xpos[g, 2]))
    print(f"[deploy] init lowest geom z = {lowest_z:.4f}  (negative => ground penetration)")

    # ---- Obs builder -----------------------------------------------------
    obs_builder = H1_2ObsBuilder(
        m=m, d=d,
        isaac_to_mj=isaac_to_mj,
        q_default_isaac=q_default_isaac,
        leg_isaac_indices=leg_isaac_indices,
        upper_isaac_indices=upper_isaac_indices,
        action_dim=action_dim,
        obs_term_order=obs_term_order,
        history_length=obs_history_length,
    )
    obs_builder.vel_cmd = np.array(
        [args.cmd_lin_x, args.cmd_lin_y, args.cmd_ang_z], dtype=np.float32
    )

    # ---- Gamepad ---------------------------------------------------------
    gamepad: Optional["GamepadReader"] = None
    if not args.no_gamepad:
        if GamepadReader is None:
            raise RuntimeError(
                f"Failed to import gamepad.py (pygame missing?): {_GAMEPAD_IMPORT_ERR}\n"
                f"Install pygame or run with --no_gamepad."
            )
        gamepad = GamepadReader(
            vx_max=args.vx_max, vy_max=args.vy_max, wz_max=args.wz_max,
            deadzone=args.deadzone, device_index=args.gamepad_index,
        )
        obs_builder.vel_cmd = gamepad.read()
        print(f"[deploy] gamepad ACTIVE — CLI --cmd_* values are unused")
    else:
        print(f"[deploy] gamepad DISABLED — driving from CLI fallback values")
    print(f"[deploy] initial vel_cmd   : {obs_builder.vel_cmd.tolist()}")

    # Verify obs dimension on first call.
    first_obs = obs_builder.step()
    if first_obs.shape[0] != total_obs_dim_yaml:
        raise RuntimeError(
            f"First obs dim {first_obs.shape[0]} != expected {total_obs_dim_yaml}. "
            f"Check obs term order/dims. YAML order: {obs_term_order}"
        )
    print(f"[deploy] first obs dim     : {first_obs.shape[0]} ✓")

    # ---- Actuator delay setup -------------------------------------------
    delay_min = int(args.delay_min)
    delay_max = int(args.delay_max)
    if delay_max < delay_min:
        raise ValueError(f"--delay_max ({delay_max}) < --delay_min ({delay_min})")
    delay_rng = np.random.default_rng(args.delay_seed)
    delays_per_joint_mj = delay_rng.integers(delay_min, delay_max + 1, size=n_dof).astype(np.int32)
    delay_buf_size = delay_max + 1
    target_history_mj = np.tile(q_default_mj.astype(np.float32), (delay_buf_size, 1))
    joint_idx_arange = np.arange(n_dof, dtype=np.int32)
    print(f"[deploy] actuator delay    : per-joint uniform [{delay_min}, {delay_max}] steps "
          f"({delay_min*sim_dt*1000:.0f}-{delay_max*sim_dt*1000:.0f} ms)")

    # ---- Foot metrics ----------------------------------------------------
    knee_isaac_L = joint_names_isaac.index("left_knee_joint")
    knee_isaac_R = joint_names_isaac.index("right_knee_joint")
    body_weight_N = float(np.sum(m.body_mass)) * 9.81
    print(f"[deploy] model total mass  : {np.sum(m.body_mass):.1f} kg  (weight {body_weight_N:.0f} N)")
    foot_metrics = FootMetrics(
        m=m,
        foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        knee_isaac_indices=(knee_isaac_L, knee_isaac_R),
        body_weight_N=body_weight_N,
        window=6,
    )

    # ---- Control loop ----------------------------------------------------
    target_dof_pos_mj = q_default_mj.copy()
    counter = 0

    # ---- Settle: hold the default pose (no policy) for a short spin-up so the
    # drop-in transient dies before the policy engages (improves first-second
    # robustness; the policy expects to start from a settled stance). --------
    n_settle = int(max(0.0, args.settle) / sim_dt)
    for _ in range(n_settle):
        tau = kp_mj * (q_default_mj - d.qpos[7:])
        if args.explicit_pd:
            tau = tau - kd_mj * d.qvel[6:]
        d.qfrc_applied[6:] = np.clip(tau, -tau_limit_mj, tau_limit_mj)
        mujoco.mj_step(m, d)
    if n_settle:
        print(f"[deploy] settled {args.settle:.2f}s: base_z={d.qpos[2]:.3f}")
    d.time = 0.0  # reset clock so the heartbeat starts at policy engage

    # ---- Control loop (viewer optional) ----------------------------------
    viewer = None if args.headless else mujoco.viewer.launch_passive(m, d)
    if args.headless:
        print(f"[deploy] HEADLESS — no viewer, running {args.duration or 10}s")
    else:
        print(f"[deploy] starting viewer …")
    try:
        start = time.time()
        last_print_wall_t = start
        last_print_sim_t = 0.0
        dur = args.duration if args.duration > 0 else (10.0 if args.headless else 0.0)
        while (viewer.is_running() if viewer is not None else True) and (
            dur <= 0 or time.time() - start < dur
        ):
            step_start = time.time()

            target_history_mj[counter % delay_buf_size] = target_dof_pos_mj
            idx = (counter - delays_per_joint_mj) % delay_buf_size
            delayed_target_mj = target_history_mj[idx, joint_idx_arange]

            q_mj = d.qpos[7:]
            dq_mj = d.qvel[6:]
            # Default (implicit) path: apply only the stiffness term; the -kd*qvel
            # damping is applied by the model (dof_damping) and integrated
            # implicitly. Legacy --explicit_pd keeps the full explicit PD.
            if args.explicit_pd:
                tau = kp_mj * (delayed_target_mj - q_mj) - kd_mj * dq_mj
            else:
                tau = kp_mj * (delayed_target_mj - q_mj)
            tau = np.clip(tau, -tau_limit_mj, tau_limit_mj)
            d.qfrc_applied[6:] = tau

            mujoco.mj_step(m, d)
            counter += 1

            q_isaac_now = d.qpos[7:][isaac_to_mj]
            foot_metrics.update(m, d, q_isaac_now)

            if not np.all(np.isfinite(d.qpos)):
                print(f"[deploy] !! qpos went NaN at step {counter} — stopping.")
                print(f"[deploy]    base z just before NaN: {d.qpos[2]:.4f}")
                print(f"[deploy]    last tau abs max: {np.abs(tau).max():.2f}")
                break

            if counter % decimation == 0:
                if gamepad is not None:
                    obs_builder.vel_cmd = gamepad.read()

                obs = obs_builder.step()
                action = run_policy(obs)
                obs_builder.last_action = action.copy()

                # PD target in Isaac order: legs = default + scale*action, the
                # 15 upper-body joints stay at default (never touched).
                target_dof_pos_isaac = q_default_isaac.copy()
                scaled = action_scale * action
                if use_default_offset:
                    target_dof_pos_isaac[action_to_isaac] = q_default_isaac[action_to_isaac] + scaled
                else:
                    target_dof_pos_isaac[action_to_isaac] = scaled
                target_dof_pos_mj = target_dof_pos_isaac[mj_to_isaac]

                policy_steps = counter // decimation
                if policy_steps % 50 == 1:
                    now = time.time()
                    dt_sim = d.time - last_print_sim_t
                    dt_wall = now - last_print_wall_t
                    rt_factor = dt_sim / dt_wall if dt_wall > 0 else 0.0
                    last_print_sim_t = d.time
                    last_print_wall_t = now
                    v_body = quat_rotate_inverse(
                        d.qpos[3:7].astype(np.float32), d.qvel[0:3].astype(np.float32)
                    )
                    print(
                        f"[deploy] t={d.time:6.2f}s rt={rt_factor:.2f}x  "
                        f"cmd=[vx={obs_builder.vel_cmd[0]:+.2f} vy={obs_builder.vel_cmd[1]:+.2f} "
                        f"wz={obs_builder.vel_cmd[2]:+.2f}]  "
                        f"actual=[vx={v_body[0]:+.2f} vy={v_body[1]:+.2f} wz={d.qvel[5]:+.2f}]  "
                        f"base_z={d.qpos[2]:.3f}  |tau|={np.abs(tau).max():.1f}"
                    )
                    q_isaac = q_mj[isaac_to_mj]
                    iL = {n: joint_names_isaac.index(n) for n in
                          ["left_hip_pitch_joint", "right_hip_pitch_joint",
                           "left_knee_joint", "right_knee_joint",
                           "left_ankle_pitch_joint", "right_ankle_pitch_joint"]}
                    print(
                        f"[deploy]     hip_pit L={q_isaac[iL['left_hip_pitch_joint']]:+.3f} "
                        f"R={q_isaac[iL['right_hip_pitch_joint']]:+.3f}  "
                        f"knee L={q_isaac[iL['left_knee_joint']]:+.3f} "
                        f"R={q_isaac[iL['right_knee_joint']]:+.3f}  "
                        f"ank_pit L={q_isaac[iL['left_ankle_pitch_joint']]:+.3f} "
                        f"R={q_isaac[iL['right_ankle_pitch_joint']]:+.3f}"
                    )
                    print(f"[deploy]     {foot_metrics.summary()}")

            if viewer is not None and counter % args.viewer_sync_every == 0:
                viewer.sync()

            if args.realtime and not args.headless:
                slack = sim_dt - (time.time() - step_start)
                if slack > 0:
                    time.sleep(slack)
    finally:
        if viewer is not None:
            viewer.close()

    print(f"[deploy] simulation ended after {time.time() - start:.1f}s, {counter} physics steps")
    if gamepad is not None:
        gamepad.close()


if __name__ == "__main__":
    sys.exit(main())
