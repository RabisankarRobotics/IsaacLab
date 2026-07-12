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
    """Builds the 83-d single-frame observation for HV1.2 velocity policy.

    Order matches HV1_2VelocityObservationsCfg.PolicyCfg declaration order:
      base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions.
    """

    def __init__(
        self,
        m: mujoco.MjModel,
        d: mujoco.MjData,
        isaac_to_mj: np.ndarray,
        q_default_isaac: np.ndarray,
        action_dim: int,
        obs_term_order: List[str],
    ):
        self.m = m
        self.d = d
        self.isaac_to_mj = isaac_to_mj
        self.q_default_isaac = q_default_isaac.astype(np.float32).copy()
        self.obs_term_order = obs_term_order
        self.last_action = np.zeros(action_dim, dtype=np.float32)
        self.vel_cmd = np.zeros(3, dtype=np.float32)

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
        parts: List[np.ndarray] = []
        for name in self.obs_term_order:
            parts.append(self._term_fns[name]().astype(np.float32))
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
    parser.add_argument("--vx_max", type=float, default=1.0,
                        help="Forward-velocity magnitude at full stick deflection "
                             "(m/s). Should match training lin_vel_x range.")
    parser.add_argument("--vy_max", type=float, default=0.5,
                        help="Lateral-velocity magnitude at full deflection (m/s). "
                             "Should match training lin_vel_y range.")
    parser.add_argument("--wz_max", type=float, default=1.0,
                        help="Yaw-rate magnitude at full deflection (rad/s). "
                             "Should match training ang_vel_z range.")
    parser.add_argument("--deadzone", type=float, default=0.1,
                        help="Stick deadzone, raw units in [0, 1].")
    parser.add_argument("--gamepad_index", type=int, default=0,
                        help="pygame joystick index when multiple pads are attached.")
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
    action_scale = float(cfg["action"]["scale"])
    use_default_offset = bool(cfg["action"]["use_default_offset"])
    n_dof = int(cfg["robot"]["num_dof_total"])
    action_dim = int(cfg["action"]["dim"])
    total_obs_dim_yaml = int(cfg["observation"]["total_dim"])
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

            # PD control every physics step (200 Hz default).
            q_mj = d.qpos[7:]
            dq_mj = d.qvel[6:]
            tau = kp_mj * (target_dof_pos_mj - q_mj) - kd_mj * dq_mj
            tau = np.clip(tau, -tau_limit_mj, tau_limit_mj)
            d.qfrc_applied[6:] = tau

            mujoco.mj_step(m, d)
            counter += 1

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
                # their default; override the action subset.
                target_dof_pos_isaac = q_default_isaac.copy()
                if use_default_offset:
                    target_dof_pos_isaac[action_to_isaac] = (
                        q_default_isaac[action_to_isaac] + action_scale * action
                    )
                else:
                    target_dof_pos_isaac[action_to_isaac] = action_scale * action
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
