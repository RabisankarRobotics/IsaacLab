# Copyright (c) 2026, Tahiti C1 Sim-to-Sim
# SPDX-License-Identifier: BSD-3-Clause
"""
Deploy a trained Tahiti C1 velocity-tracking policy in MuJoCo with Xbox
gamepad command input.

Observation / action contract is identical to the HV1_2 runner (same 6 obs
terms in the same order, same [vx, vy, wz] base_velocity command, same
12-DoF joint_pos action). The only functional difference vs
deploy_mujoco_hv1_2.py is that the (vx, vy, wz) command is polled live from
an Xbox controller every policy tick instead of being fixed by CLI args.

Gamepad mapping (see gamepad.py for the full stick-axis table):
  Left  stick vertical    -> vx  (up   = forward)
  Left  stick horizontal  -> wz  (left = turn left)
  Right stick horizontal  -> vy  (left = strafe left)

Scaling: raw [-1, +1] stick values map to +/-vx_max / +/-vy_max / +/-wz_max,
which default to Tahiti's training env ranges (vx +/- 1.0, vy +/- 0.5,
wz +/- 1.0). Override with --vx_max / --vy_max / --wz_max if needed.

YAML produced by:
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/dump_mujoco_config.py \\
      --task Isaac-Velocity-Flat-Tahiti_C1-Play-v0 --num_envs 1

Usage:
  python deploy/mujoco/deploy_mujoco_tahiti_c1.py \\
      --config deploy/config/tahiti_c1_mujoco_config.yaml \\
      --xml deploy/mujoco/tahiti_c1_scene.xml

  # Disable gamepad and drive from CLI (same as the HV1_2 runner):
  python deploy/mujoco/deploy_mujoco_tahiti_c1.py \\
      --config deploy/config/tahiti_c1_mujoco_config.yaml \\
      --xml deploy/mujoco/tahiti_c1_scene.xml \\
      --no_gamepad --cmd_lin_x 0.5
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

# Import GamepadReader from the sibling gamepad.py in this directory. Import
# is delayed to a try/except so a missing pygame install only breaks the
# gamepad path — --no_gamepad + CLI velocity still works.
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
    """Track per-foot air-time and knee-swing-amplitude for L/R asymmetry diagnostics.

    - Air-time: time from liftoff to next touchdown, per foot. Keeps a rolling
      window of the last N completed cycles per side and reports the mean.
    - Knee amplitude: max minus min knee angle across the current swing phase.
      Sampled every physics step while the foot is off the ground; snapshot
      into the rolling window at touchdown.
    """

    def __init__(
        self,
        m: mujoco.MjModel,
        foot_body_names: Tuple[str, str],
        knee_isaac_indices: Tuple[int, int],
        window: int = 6,
    ):
        self.body_ids = tuple(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) for name in foot_body_names
        )
        self.knee_isaac_indices = knee_isaac_indices
        self.window = window
        self.in_contact = [True, True]
        self.last_liftoff_t = [0.0, 0.0]
        self.airtimes = [[], []]  # rolling windows per foot
        self.swing_min_knee = [1e9, 1e9]
        self.swing_max_knee = [-1e9, -1e9]
        self.knee_amplitudes = [[], []]
        # 2026-07-21: per-foot contact force diagnostic. peak_force is the
        # all-time max normal force per foot; contact_force_samples is a
        # rolling per-step buffer for mean force during stance.
        self.peak_force = [0.0, 0.0]
        self.contact_force_samples = [[], []]

    @staticmethod
    def _push(buf: List[float], value: float, window: int) -> None:
        buf.append(value)
        if len(buf) > window:
            buf.pop(0)

    def update(self, m: mujoco.MjModel, d: mujoco.MjData, q_isaac: np.ndarray) -> None:
        contact_now = [False, False]
        step_force = [0.0, 0.0]  # sum of normal contact force per foot this physics step
        force6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = m.geom_bodyid[c.geom1]
            b2 = m.geom_bodyid[c.geom2]
            for foot_idx, bid in enumerate(self.body_ids):
                if b1 == bid or b2 == bid:
                    contact_now[foot_idx] = True
                    # Contact-frame force: force6[0] is normal into surface.
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
                # in swing — track min/max knee for amplitude
                self.swing_min_knee[foot_idx] = min(self.swing_min_knee[foot_idx], knee_val)
                self.swing_max_knee[foot_idx] = max(self.swing_max_knee[foot_idx], knee_val)

            # edge transitions
            was = self.in_contact[foot_idx]
            now = contact_now[foot_idx]
            if was and not now:
                # liftoff
                self.last_liftoff_t[foot_idx] = float(d.time)
                # reset swing amplitude tracking for the new swing
                self.swing_min_knee[foot_idx] = knee_val
                self.swing_max_knee[foot_idx] = knee_val
            elif (not was) and now:
                # touchdown — record air-time and knee amplitude for the completed swing
                airtime = float(d.time) - self.last_liftoff_t[foot_idx]
                if 0.05 < airtime < 2.0:  # sanity gate: ignore noise blips
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
        at_asym = (atL - atR) / max(atL + atR, 1e-6) * 100.0  # % of mean
        ka_asym = (kaL - kaR) / max(kaL + kaR, 1e-6) * 100.0
        # Force diagnostic. body_weight_N = 53.5 kg * 9.81 ≈ 525 N reference.
        # Healthy human-like walk peaks 1.2-1.5× BW. Stomping 1.8-2.5×. > 2.5× = destructive.
        fL_peak, fR_peak = self.peak_force[0], self.peak_force[1]
        fL_mean = _mean(self.contact_force_samples[0])
        fR_mean = _mean(self.contact_force_samples[1])
        body_weight_N = 53.5 * 9.81
        return (
            f"air_time  L={atL:.3f}s R={atR:.3f}s  asym={at_asym:+.1f}%  (n={nL}/{nR})  "
            f"knee_amp  L={kaL:.3f} R={kaR:.3f}  asym={ka_asym:+.1f}%\n"
            f"force     L peak={fL_peak:.0f}N ({fL_peak/body_weight_N:.2f}xBW) "
            f"mean={fL_mean:.0f}N   "
            f"R peak={fR_peak:.0f}N ({fR_peak/body_weight_N:.2f}xBW) mean={fR_mean:.0f}N"
        )


def build_joint_maps(
    isaac_joint_names: List[str],
    action_joint_names: List[str],
    mj_joint_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the permutations to translate between Isaac and MuJoCo joint orderings.

    Returns
    -------
    isaac_to_mj : array of shape (N,)
        ``q_mj[isaac_to_mj]`` gives values in Isaac order. MuJoCo → Isaac.
        (Gotcha #1 in hv1_mujoco_deploy_gotchas.)
    mj_to_isaac : array of shape (N,)
        ``kp_isaac[mj_to_isaac]`` gives values in MuJoCo order. Isaac → MuJoCo.
    action_to_isaac : array of shape (action_dim,)
        Maps policy-action index → Isaac-joint index.
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


def torque_limits_per_mj_joint(m: mujoco.MjModel, mj_joint_names: List[str]) -> np.ndarray:
    """Read |ctrlrange| from each joint's actuator. Fallback to 1e6 if unmapped.

    The clip in the control loop uses these so a startup PD spike can't blow up
    qpos (gotcha #6).
    """
    tau_limit = np.zeros(len(mj_joint_names), dtype=np.float32)
    for j_mj, name in enumerate(mj_joint_names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        matched = False
        for a in range(m.nu):
            if m.actuator_trnid[a, 0] == jid:
                lo, hi = m.actuator_ctrlrange[a]
                tau_limit[j_mj] = max(abs(lo), abs(hi))
                matched = True
                break
        if not matched:
            tau_limit[j_mj] = 1e6
    return tau_limit


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def set_default_pose(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    q_default_isaac: np.ndarray,
    isaac_to_mj: np.ndarray,
    base_height: float | None,
) -> None:
    """Init MuJoCo state from the MJCF ``home`` keyframe (matches Isaac default pose)."""
    mujoco.mj_resetData(m, d)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        if base_height is not None:
            d.qpos[2] = base_height
        print(f"[deploy] init pose: keyframe 0 loaded, base z = {d.qpos[2]:.3f}")
    else:
        # Fallback: build from YAML defaults.
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[2] = base_height if base_height is not None else 0.98
        d.qpos[3] = 1.0  # qw
        q_default_mj = q_default_isaac[np.argsort(isaac_to_mj)]
        d.qpos[7 : 7 + len(q_default_mj)] = q_default_mj
        print(f"[deploy] init pose: built from YAML, base z = {d.qpos[2]:.3f}")
    mujoco.mj_forward(m, d)


# ---------------------------------------------------------------------------
# Observation builder — single-frame, no history
# ---------------------------------------------------------------------------

class HV1_2ObsBuilder:
    """Builds the single-frame observation for the Tahiti C1 velocity policy,
    optionally wrapped in a per-term rolling history for history_length > 1.

    Order matches Isaac Lab's PolicyCfg declaration order:
      base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions.

    Flatten layout when history_length > 1 (matches Isaac Lab
    ``concatenate_terms=True`` and robo_control's ObservationHistory.flatten):
        [term0_oldest, term0_oldest+1, ..., term0_newest,
         term1_oldest, ..., term1_newest, ...]
    """

    def __init__(
        self,
        m: mujoco.MjModel,
        d: mujoco.MjData,
        isaac_to_mj: np.ndarray,
        q_default_isaac: np.ndarray,
        action_dim: int,
        obs_term_order: List[str],
        history_length: int = 1,
    ):
        self.m = m
        self.d = d
        self.isaac_to_mj = isaac_to_mj
        self.q_default_isaac = q_default_isaac.astype(np.float32).copy()
        self.obs_term_order = obs_term_order
        self.last_action = np.zeros(action_dim, dtype=np.float32)
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.history_length = int(history_length)

        # Dispatch table for term name → compute function. base_lin_vel
        # intentionally absent: the policy was trained without it (no state
        # estimator on the real robot).
        self._term_fns = {
            "base_ang_vel": self._base_ang_vel,
            "projected_gravity": self._projected_gravity,
            "velocity_commands": self._velocity_commands,
            "joint_pos": self._joint_pos,
            "joint_vel": self._joint_vel,
            "actions": self._actions,
        }
        missing = [t for t in obs_term_order if t not in self._term_fns]
        if missing:
            raise RuntimeError(f"No compute function for obs terms: {missing}")

        # Per-term rolling buffers for history. First push seeds the buffer
        # with H copies of the first observation.
        self._history: dict[str, list[np.ndarray]] = {t: [] for t in obs_term_order}

    def _base_ang_vel(self) -> np.ndarray:
        # qvel[3:6] for free joint = body-frame angular velocity in MuJoCo. (gotcha #2)
        return self.d.qvel[3:6].astype(np.float32).copy()

    def _projected_gravity(self) -> np.ndarray:
        base_quat = self.d.qpos[3:7].astype(np.float32)
        return projected_gravity_in_frame(base_quat)

    def _velocity_commands(self) -> np.ndarray:
        return self.vel_cmd

    def _joint_pos(self) -> np.ndarray:
        # joint_pos_rel = q - q_default, in Isaac joint order.
        q_mj = self.d.qpos[7:].astype(np.float32)
        q_isaac = q_mj[self.isaac_to_mj]
        return q_isaac - self.q_default_isaac

    def _joint_vel(self) -> np.ndarray:
        # joint_vel_rel = dq - default_dq, but default_dq = 0 so just dq in Isaac order.
        dq_mj = self.d.qvel[6:].astype(np.float32)
        return dq_mj[self.isaac_to_mj]

    def _actions(self) -> np.ndarray:
        return self.last_action

    def step(self) -> np.ndarray:
        # Compute current single-frame values per term, push into history,
        # flatten with the Isaac Lab per-term layout.
        for name in self.obs_term_order:
            current = self._term_fns[name]().astype(np.float32)
            buf = self._history[name]
            if not buf:
                # Seed with H copies of the first observation.
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
        description="Deploy Tahiti C1 velocity policy in MuJoCo with Xbox gamepad control."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to mujoco_config.yaml.")
    parser.add_argument(
        "--policy_format", choices=["pt", "onnx"], default="onnx",
        help="Inference backend. 'onnx' (default) uses onnxruntime — lighter dep, "
             "marginally faster on CPU, matches real-robot deploy path. "
             "'pt' uses TorchScript (torch.jit) for parity-check against PyTorch.",
    )
    parser.add_argument(
        "--policy", type=str, default=None,
        help="Override policy file path. Format determined by --policy_format "
             "(default: YAML's policy.jit_path for pt, policy.onnx_path for onnx).",
    )
    parser.add_argument(
        "--xml", type=str,
        default="/home/rabisankar/IsaacLab/deploy/mujoco/tahiti_c1_scene.xml",
        help="MJCF scene file (includes tahiti_c1.xml).",
    )
    parser.add_argument("--duration", type=float, default=120.0, help="Sim duration in seconds.")
    parser.add_argument("--base_height", type=float, default=None,
                        help="Override initial pelvis z (default = keyframe value).")
    parser.add_argument("--realtime", action="store_true", default=True,
                        help="Sleep between steps so sim runs at wall-clock speed.")
    parser.add_argument(
        "--viewer_sync_every", type=int, default=4,
        help="Call viewer.sync() every N physics steps. Default 4 matches the "
             "policy rate (50 Hz refresh at sim_dt=0.005) and gives a ~3x sim "
             "speed-up vs syncing every physics step (200 Hz). Set to 1 for "
             "max-fidelity rendering, higher for further speed-up.",
    )
    parser.add_argument("--cmd_lin_x", type=float, default=0.0,
                        help="Fallback forward velocity when --no_gamepad (m/s).")
    parser.add_argument("--cmd_lin_y", type=float, default=0.0,
                        help="Fallback sideways velocity when --no_gamepad (m/s).")
    parser.add_argument("--cmd_ang_z", type=float, default=0.0,
                        help="Fallback yaw rate when --no_gamepad (rad/s).")

    # -- Gamepad options --
    parser.add_argument("--no_gamepad", action="store_true",
                        help="Disable the Xbox gamepad and drive from --cmd_lin_x/y/z instead.")
    parser.add_argument("--vx_max", type=float, default=0.8,
                        help="Forward-velocity magnitude at full stick deflection "
                             "(m/s). Matches training lin_vel_x range (±0.8 m/s "
                             "as of 2026-07-16, was ±1.0).")
    parser.add_argument("--vy_max", type=float, default=0.5,
                        help="Lateral-velocity magnitude at full deflection (m/s). "
                             "Matches training lin_vel_y range (±0.5 m/s).")
    parser.add_argument("--wz_max", type=float, default=0.5,
                        help="Yaw-rate magnitude at full deflection (rad/s). "
                             "Matches training ang_vel_z range (±0.5 rad/s). "
                             "Was 1.0 — that was OUT of policy training range.")
    parser.add_argument("--deadzone", type=float, default=0.1,
                        help="Stick deadzone, raw units in [0, 1].")
    parser.add_argument("--gamepad_index", type=int, default=0,
                        help="pygame joystick index when multiple pads are attached.")

    # -- Actuator delay (match Isaac DelayedPDActuatorCfg 0-6 phys steps) --
    parser.add_argument("--delay_min", type=int, default=0,
                        help="Min per-joint actuator delay in physics steps. Default 0 "
                             "matches Isaac DelayedPDActuatorCfg. Sampled once at startup.")
    parser.add_argument("--delay_max", type=int, default=6,
                        help="Max per-joint actuator delay in physics steps. Default 6 "
                             "matches Isaac DelayedPDActuatorCfg (0-30 ms at sim_dt=0.005). "
                             "Set --delay_max=0 (with --delay_min=0) to disable delay for A/B.")
    parser.add_argument("--delay_seed", type=int, default=None,
                        help="Seed for the per-joint delay sample. Default: unseeded (differs "
                             "run-to-run, like Isaac's per-env DR). Pin for reproducible tests.")
    args = parser.parse_args()

    # ---- Load config ------------------------------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    xml_path = args.xml
    # Resolve policy path per backend: CLI override wins, else YAML default.
    if args.policy_format == "onnx":
        policy_path = args.policy or cfg["policy"].get("onnx_path")
    else:
        policy_path = args.policy or cfg["policy"].get("jit_path")
    if not policy_path:
        raise RuntimeError(
            f"No policy path found for format '{args.policy_format}'. "
            f"Pass --policy or check YAML's policy.{ 'onnx_path' if args.policy_format == 'onnx' else 'jit_path' }."
        )
    sim_dt = float(cfg["control"]["sim_dt"])
    decimation = int(cfg["control"]["decimation"])
    joint_names_isaac = cfg["robot"]["joint_names_isaac_order"]
    joint_names_action = cfg["action"]["joint_names_action_order"]
    q_default_isaac = np.array(cfg["robot"]["default_joint_pos"], dtype=np.float32)
    kp_isaac = np.array(cfg["robot"]["kp"], dtype=np.float32)
    kd_isaac = np.array(cfg["robot"]["kd"], dtype=np.float32)
    # viscous_friction in the YAML is a recorded motor spec — it's already
    # applied as passive damping in the MJCF (<joint damping="..."/>) and is
    # NOT subtracted from the Python PD kd. The double-count this introduces
    # is small (~2% of leg kd) and consistent with how the real motor behaves
    # (passive viscous + active controller damping coexist).
    # action_scale can be a scalar (uniform) or a list (per action-order joint).
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
    print(f"[deploy] viewer_sync_every : {args.viewer_sync_every}  (viewer refresh ≈ {1.0 / (sim_dt * args.viewer_sync_every):.0f} Hz)")
    print(f"[deploy] n_dof_total       : {n_dof}    action_dim={action_dim}")
    print(f"[deploy] action_scale      : {action_scale}    use_default_offset={use_default_offset}")
    print(f"[deploy] total_obs_dim     : {total_obs_dim_yaml}  (from YAML)")
    print(f"[deploy] obs term order    : {obs_term_order}")

    # ---- Load MuJoCo model -----------------------------------------------
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

    # Convert PD gains and defaults into MuJoCo joint order for fast indexing.
    kp_mj = kp_isaac[mj_to_isaac]
    kd_mj = kd_isaac[mj_to_isaac]
    q_default_mj = q_default_isaac[mj_to_isaac]

    # Per-joint torque clip (gotcha #6).
    tau_limit_mj = torque_limits_per_mj_joint(m, mj_joint_names)
    print(f"[deploy] kp range          : [{kp_isaac.min():.1f}, {kp_isaac.max():.1f}]")
    print(f"[deploy] kd range          : [{kd_isaac.min():.2f}, {kd_isaac.max():.2f}]")
    print(f"[deploy] torque clamps     : min={tau_limit_mj.min():.1f}  max={tau_limit_mj.max():.1f} Nm")

    # ---- Load policy -----------------------------------------------------
    # Unified inference behind run_policy(obs_np) -> action_np so the control
    # loop is backend-agnostic. Both backends produce float32 actions of shape
    # (action_dim,) given a (obs_dim,) numpy input.
    if args.policy_format == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime not installed. Install with `pip install onnxruntime` "
                "or re-run with --policy_format pt."
            ) from e
        session = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        onnx_input_name = session.get_inputs()[0].name
        onnx_output_name = session.get_outputs()[0].name
        print(f"[deploy] onnx loaded       : input='{onnx_input_name}'  output='{onnx_output_name}'")

        def run_policy(obs_np: np.ndarray) -> np.ndarray:
            out = session.run(
                [onnx_output_name],
                {onnx_input_name: obs_np[np.newaxis, :].astype(np.float32)},
            )
            return out[0].squeeze(0).astype(np.float32)
    else:  # 'pt'
        policy = torch.jit.load(policy_path, map_location="cpu").eval()
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"[deploy] policy loaded     : ({n_params:,} params)")

        def run_policy(obs_np: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_np).unsqueeze(0)
                action_t = policy(obs_t)
            return action_t.detach().numpy().squeeze(0).astype(np.float32)

    # ---- Initial state ---------------------------------------------------
    set_default_pose(m, d, q_default_isaac, isaac_to_mj, args.base_height)

    lowest_z = float("inf")
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            continue
        lowest_z = min(lowest_z, float(d.geom_xpos[g, 2]))
    print(f"[deploy] init lowest geom z = {lowest_z:.4f}  (negative ⇒ ground penetration)")

    # ---- Build obs builder ----------------------------------------------
    obs_builder = HV1_2ObsBuilder(
        m=m, d=d,
        isaac_to_mj=isaac_to_mj,
        q_default_isaac=q_default_isaac,
        action_dim=action_dim,
        obs_term_order=obs_term_order,
        history_length=obs_history_length,
    )
    obs_builder.vel_cmd = np.array(
        [args.cmd_lin_x, args.cmd_lin_y, args.cmd_ang_z], dtype=np.float32
    )

    # ---- Gamepad -----------------------------------------------------------
    gamepad: Optional["GamepadReader"] = None
    if not args.no_gamepad:
        if GamepadReader is None:
            raise RuntimeError(
                f"Failed to import gamepad.py (pygame missing?): {_GAMEPAD_IMPORT_ERR}\n"
                f"Install pygame (`pip install pygame`) or run with --no_gamepad."
            )
        gamepad = GamepadReader(
            vx_max=args.vx_max,
            vy_max=args.vy_max,
            wz_max=args.wz_max,
            deadzone=args.deadzone,
            device_index=args.gamepad_index,
        )
        # Seed vel_cmd from the first read so the first observation reflects
        # the actual (usually near-zero) stick state, not the CLI fallback.
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
            f"Check obs term order. YAML order: {obs_term_order}"
        )
    print(f"[deploy] first obs dim     : {first_obs.shape[0]} ✓")
    print(f"[deploy] starting viewer …")

    # ---- Actuator delay setup (Isaac DelayedPDActuatorCfg match) --------
    # Isaac trains under a per-joint uniform random delay [min_delay, max_delay]
    # on the PD target (TAHITI_C1_CFG DelayedPDActuatorCfg). Applying targets
    # IMMEDIATELY here creates a sim-to-sim gap the policy trained to anticipate
    # — it over-corrects, destabilizing marginal gaits (observed as MuJoCo
    # backward-walk falls that don't reproduce in Isaac PLAY).
    delay_min = int(args.delay_min)
    delay_max = int(args.delay_max)
    if delay_max < delay_min:
        raise ValueError(f"--delay_max ({delay_max}) < --delay_min ({delay_min})")
    delay_rng = np.random.default_rng(args.delay_seed)
    delays_per_joint_mj = delay_rng.integers(delay_min, delay_max + 1, size=n_dof).astype(np.int32)
    delay_buf_size = delay_max + 1
    # Rolling buffer of the last `delay_buf_size` physics-step PD targets, one
    # row per past step. Init all rows to the default pose so the first
    # (delay_max) steps see the default (matches Isaac's fresh-episode buffer).
    target_history_mj = np.tile(q_default_mj.astype(np.float32), (delay_buf_size, 1))
    joint_idx_arange = np.arange(n_dof, dtype=np.int32)
    print(f"[deploy] actuator delay    : per-joint uniform [{delay_min}, {delay_max}] steps "
          f"({delay_min*sim_dt*1000:.0f}-{delay_max*sim_dt*1000:.0f} ms)")
    print(f"[deploy] sampled delays    : {delays_per_joint_mj.tolist()}")

    # ---- Foot-metrics diagnostic (per-foot air time + knee swing amplitude) ----
    # Isaac BFS order — knees are at indices 6 (left) and 7 (right).
    knee_isaac_L = joint_names_isaac.index("left_knee_joint")
    knee_isaac_R = joint_names_isaac.index("right_knee_joint")
    foot_metrics = FootMetrics(
        m=m,
        foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        knee_isaac_indices=(knee_isaac_L, knee_isaac_R),
        window=6,
    )

    # ---- Control loop ---------------------------------------------------
    target_dof_pos_mj = q_default_mj.copy()
    counter = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        # Sliding-window timestamps for the rt_factor heartbeat — measure
        # sim-vs-wall rate over the last second instead of cumulative since
        # the first noisy startup interval.
        last_print_wall_t = start
        last_print_sim_t = 0.0
        while viewer.is_running() and time.time() - start < args.duration:
            step_start = time.time()

            # PD control every physics step (200 Hz default), using the
            # per-joint DELAYED target so this runner matches Isaac's
            # DelayedPDActuatorCfg. Push the currently-held target into the
            # rolling history first, then read out the delayed slice per joint.
            target_history_mj[counter % delay_buf_size] = target_dof_pos_mj
            idx = (counter - delays_per_joint_mj) % delay_buf_size
            delayed_target_mj = target_history_mj[idx, joint_idx_arange]

            q_mj = d.qpos[7:]
            dq_mj = d.qvel[6:]
            tau = kp_mj * (delayed_target_mj - q_mj) - kd_mj * dq_mj
            tau = np.clip(tau, -tau_limit_mj, tau_limit_mj)
            d.qfrc_applied[6:] = tau

            mujoco.mj_step(m, d)
            counter += 1

            # Update foot metrics EVERY physics step (contact events happen at
            # sim_dt=0.005 granularity — sampling only at policy tick would miss
            # short touchdowns).
            q_isaac_now = d.qpos[7:][isaac_to_mj]
            foot_metrics.update(m, d, q_isaac_now)

            # NaN guard — gotcha #6 prevents this in normal operation but keep
            # the safety net so we fail loudly instead of producing junk.
            if not np.all(np.isfinite(d.qpos)):
                print(f"[deploy] !! qpos went NaN at step {counter} — stopping.")
                print(f"[deploy]    base z just before NaN: {d.qpos[2]:.4f}")
                print(f"[deploy]    last tau abs max: {np.abs(tau).max():.2f}")
                break

            # Policy every `decimation` steps (50 Hz with sim_dt=0.005, decimation=4).
            if counter % decimation == 0:
                # Refresh command from gamepad BEFORE building the observation
                # so the policy sees the current stick state on this tick.
                if gamepad is not None:
                    obs_builder.vel_cmd = gamepad.read()

                obs = obs_builder.step()
                action = run_policy(obs)
                obs_builder.last_action = action.copy()

                # Build PD target in Isaac order: keep all non-action joints at
                # their default; override the action subset. action_scale may
                # be a scalar (uniform) or a per-action-joint array (length
                # action_dim, indexed by action order); numpy handles both.
                target_dof_pos_isaac = q_default_isaac.copy()
                scaled = action_scale * action
                if use_default_offset:
                    target_dof_pos_isaac[action_to_isaac] = (
                        q_default_isaac[action_to_isaac] + scaled
                    )
                else:
                    target_dof_pos_isaac[action_to_isaac] = scaled
                target_dof_pos_mj = target_dof_pos_isaac[mj_to_isaac]

                # Heartbeat — once per second at 50 Hz policy.
                policy_steps = counter // decimation
                if policy_steps % 50 == 1:
                    # Real-time factor over the last heartbeat interval —
                    # detects below-real-time sim even when args.realtime
                    # is on (e.g. CPU bottleneck, viewer eating frame time).
                    now = time.time()
                    dt_sim = d.time - last_print_sim_t
                    dt_wall = now - last_print_wall_t
                    rt_factor = dt_sim / dt_wall if dt_wall > 0 else 0.0
                    last_print_sim_t = d.time
                    last_print_wall_t = now
                    # Actual base velocity in body frame so it can be
                    # compared directly with the velocity command (commands
                    # are interpreted in the robot's yaw-aligned frame).
                    v_body = quat_rotate_inverse(
                        d.qpos[3:7].astype(np.float32),
                        d.qvel[0:3].astype(np.float32),
                    )
                    print(
                        f"[deploy] t={d.time:6.2f}s rt={rt_factor:.2f}x  "
                        f"cmd=[vx={obs_builder.vel_cmd[0]:+.2f} "
                        f"vy={obs_builder.vel_cmd[1]:+.2f} "
                        f"wz={obs_builder.vel_cmd[2]:+.2f}]  "
                        f"actual=[vx={v_body[0]:+.2f} "
                        f"vy={v_body[1]:+.2f} "
                        f"wz={d.qvel[5]:+.2f}]  "
                        f"base_z={d.qpos[2]:.3f}  "
                        f"|tau|={np.abs(tau).max():.1f}"
                    )
                    # Diagnostic joint printout — Isaac BFS order:
                    #   [0]L_hy [1]R_hy [2]L_hp [3]R_hp [4]L_hr [5]R_hr
                    #   [6]L_kn [7]R_kn [8]L_ap [9]R_ap [10]L_ar [11]R_ar
                    # For symmetric forward walk, L/R pairs should mirror in sign
                    # for hip_yaw and hip_roll (both near 0), match closely for
                    # hip_pitch/knee/ankle_pitch (they're symmetric in default pose).
                    q_isaac = q_mj[isaac_to_mj]
                    print(
                        f"[deploy]     hip_yaw  L={q_isaac[0]:+.3f} R={q_isaac[1]:+.3f}  "
                        f"hip_roll L={q_isaac[4]:+.3f} R={q_isaac[5]:+.3f}"
                    )
                    print(
                        f"[deploy]     hip_pit  L={q_isaac[2]:+.3f} R={q_isaac[3]:+.3f}  "
                        f"knee     L={q_isaac[6]:+.3f} R={q_isaac[7]:+.3f}"
                    )
                    print(
                        f"[deploy]     ank_pit  L={q_isaac[8]:+.3f} R={q_isaac[9]:+.3f}  "
                        f"ank_roll L={q_isaac[10]:+.3f} R={q_isaac[11]:+.3f}"
                    )
                    # Per-foot air-time and knee swing amplitude diagnostics —
                    # rolling mean of the last ~6 completed swing cycles per foot.
                    # asym% is (L - R) / (L + R) * 100 — positive means LEFT larger.
                    print(f"[deploy]     {foot_metrics.summary()}")

            # Render every Nth physics step instead of every step. Skipping
            # 3 of every 4 viewer.sync() calls is the cheapest path to higher
            # rt_factor on CPU-bottlenecked machines; physics integration
            # itself is much cheaper than the GL renderer.
            if counter % args.viewer_sync_every == 0:
                viewer.sync()

            if args.realtime:
                slack = sim_dt - (time.time() - step_start)
                if slack > 0:
                    time.sleep(slack)

    print(f"[deploy] simulation ended after {time.time() - start:.1f}s, "
          f"{counter} physics steps")

    if gamepad is not None:
        gamepad.close()


if __name__ == "__main__":
    sys.exit(main())
