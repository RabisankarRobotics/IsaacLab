# Copyright (c) 2026, Loco-Manip HV1 Sim-to-Sim
# SPDX-License-Identifier: BSD-3-Clause
"""
Interactive HV1 V3 deploy — drag EE targets with the mouse, see live errors.

This is `deploy_mujoco_hv1_v3.py` with an interactive control surface:

  * EE targets are read from MOCAP BODIES in the MJCF
        - `left_ee_target`        → left EE pose command
        - `right_ee_target`       → right EE pose command
        - `body_height_target`    → body height command (only Z is used)
    Drag them in the viewer with Ctrl + right-mouse-drag (translate) or
    Ctrl + left-mouse-drag (rotate). Targets persist between policy steps.

  * Body height is keyboard-controlled because the green disk's Z motion
    depends on camera angle (Ctrl+RightDrag = screen plane; usually maps to
    XY motion unless camera is set to a perfect side view). The disk still
    follows the commanded height visually.
        "=" / "-"   body_height +/- 0.01 m  (fine,  ~1 cm)
        "+" / "_"   body_height +/- 0.05 m  (coarse, 5 cm)  [shift+= / shift+-]

  * Waist alpha is keyboard-controlled (no spatial meaning):
        "," / "."   decrease / increase by 0.1
        "[" / "]"   decrease / increase by 0.5

  * Other keys:
        "r"         reset all targets + alpha + height to defaults
        "p"         print a snapshot to scrollback (so live overlay doesn't
                    eat it)
        ESC         quit (viewer handles this)

  * Live status overlay (5 Hz refresh, ANSI escape codes):
        body_height cmd vs actual + error
        per-hand EE pose cmd vs actual (pelvis frame) + position + orientation
            error in cm / degrees
        |action|, |tau|, base pitch (so you can spot the robot tilting)

Requires the new scene XML with mocap bodies — `scene_interactive.xml`.

Usage:
    python deploy_mujoco_hv1_v3_interactive.py \\
        --config logs/rsl_rl/hv1_locomanip_v3_flat/<run>/exported/mujoco_config.yaml \\
        --xml    /home/rabisankar/IsaacLab/deploy/mujoco/scene_interactive.xml \\
        --cmd_lin_x 0.0
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from typing import Callable, Dict, List, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Quaternion helpers — Isaac convention (w, x, y, z)
# ---------------------------------------------------------------------------

def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector v into body frame using body-to-world quat q."""
    qw = q[0]
    qvec = q[1:4]
    a = v * (2.0 * qw * qw - 1.0)
    b = np.cross(qvec, v) * (2.0 * qw)
    c = qvec * (2.0 * np.dot(qvec, v))
    return a - b + c


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2 with (w, x, y, z) order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_angle_error(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle (radians) between two unit quaternions."""
    dot = abs(float(np.dot(q1, q2)))
    return 2.0 * float(np.arccos(min(1.0, dot)))


def projected_gravity_in_frame(quat: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------------

TORSO_BODY_NAME = "torso_link"
PELVIS_BODY_NAME = "pelvis"
LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"

MOCAP_LEFT = "left_ee_target"
MOCAP_RIGHT = "right_ee_target"
MOCAP_HEIGHT = "body_height_target"

# Default targets in WORLD frame — set on reset.
DEFAULT_MOCAP_LEFT = np.array([0.20, 0.25, 1.00], dtype=np.float32)
DEFAULT_MOCAP_RIGHT = np.array([0.20, -0.25, 1.00], dtype=np.float32)
DEFAULT_BODY_HEIGHT = 0.90
DEFAULT_WAIST_ALPHA = 1.0

# Limits for keyboard-adjusted scalars (match training distribution)
BODY_HEIGHT_MIN, BODY_HEIGHT_MAX = 0.70, 1.00
WAIST_ALPHA_MIN, WAIST_ALPHA_MAX = 0.05, 5.0


def list_mujoco_joint_names(m: mujoco.MjModel) -> List[str]:
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
    if set(isaac_joint_names) != set(mj_joint_names):
        only_isaac = set(isaac_joint_names) - set(mj_joint_names)
        only_mj = set(mj_joint_names) - set(isaac_joint_names)
        raise RuntimeError(
            f"Joint name mismatch.\n  Isaac-only: {sorted(only_isaac)}\n  MuJoCo-only: {sorted(only_mj)}"
        )
    isaac_to_mj = np.array([mj_joint_names.index(n) for n in isaac_joint_names], dtype=np.int64)
    mj_to_isaac = np.array([isaac_joint_names.index(n) for n in mj_joint_names], dtype=np.int64)
    action_to_isaac = np.array([isaac_joint_names.index(n) for n in action_joint_names], dtype=np.int64)
    return isaac_to_mj, mj_to_isaac, action_to_isaac


def get_body_ang_vel_local(m: mujoco.MjModel, d: mujoco.MjData, body_id: int) -> np.ndarray:
    res = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, body_id, res, 1)
    return res[:3].astype(np.float32)


def set_default_pose(m, d, q_default_isaac, isaac_to_mj, base_height):
    mujoco.mj_resetData(m, d)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        if base_height is not None:
            d.qpos[2] = base_height
    else:
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[2] = base_height if base_height is not None else 0.95
        d.qpos[3] = 1.0
        q_default_mj = q_default_isaac[np.argsort(isaac_to_mj)]
        d.qpos[7 : 7 + len(q_default_mj)] = q_default_mj
    mujoco.mj_forward(m, d)


# ---------------------------------------------------------------------------
# Obs builder — same 12 terms as deploy_mujoco_hv1_v3.py
# ---------------------------------------------------------------------------

class V3ObsBuilder:
    """Builds V3 single-frame observation and maintains per-term history."""

    def __init__(self, m, d, isaac_to_mj, action_dim, n_dof_total,
                 history_length, obs_term_order, single_frame_dims):
        self.m = m
        self.d = d
        self.isaac_to_mj = isaac_to_mj
        self.history_length = history_length
        self.obs_term_order = obs_term_order
        self.single_frame_dims = single_frame_dims

        self.torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY_NAME)
        if self.torso_id < 0:
            raise RuntimeError(f"Body '{TORSO_BODY_NAME}' not found in MuJoCo model.")

        self.q_default_isaac = np.zeros(n_dof_total, dtype=np.float32)
        self.last_action = np.zeros(action_dim, dtype=np.float32)
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.left_ee_cmd = np.zeros(7, dtype=np.float32)
        self.right_ee_cmd = np.zeros(7, dtype=np.float32)
        self.body_height_cmd = np.zeros(1, dtype=np.float32)
        self.waist_alpha_cmd = np.zeros(1, dtype=np.float32)

        self.history: Dict[str, deque] = {n: deque(maxlen=history_length) for n in obs_term_order}

        self._term_fns: Dict[str, Callable[[], np.ndarray]] = {
            "base_ang_vel": lambda: self.d.qvel[3:6].astype(np.float32).copy(),
            "projected_gravity": lambda: projected_gravity_in_frame(self.d.qpos[3:7].astype(np.float32)),
            "projected_gravity_torso": lambda: projected_gravity_in_frame(self.d.xquat[self.torso_id].astype(np.float32)),
            "torso_ang_vel": lambda: get_body_ang_vel_local(self.m, self.d, self.torso_id),
            "velocity_commands": lambda: self.vel_cmd,
            "joint_pos": lambda: self.d.qpos[7:].astype(np.float32)[self.isaac_to_mj] - self.q_default_isaac,
            "joint_vel": lambda: self.d.qvel[6:].astype(np.float32)[self.isaac_to_mj],
            "actions": lambda: self.last_action,
            "left_ee_pose_command": lambda: self.left_ee_cmd,
            "right_ee_pose_command": lambda: self.right_ee_cmd,
            "body_height_command": lambda: self.body_height_cmd,
            "waist_alpha_command": lambda: self.waist_alpha_cmd,
        }
        missing = [t for t in obs_term_order if t not in self._term_fns]
        if missing:
            raise RuntimeError(f"No compute function for obs terms: {missing}")

    def reset_history(self):
        for q in self.history.values():
            q.clear()

    def step(self) -> np.ndarray:
        per_term_flat: List[np.ndarray] = []
        for name in self.obs_term_order:
            current = self._term_fns[name]().astype(np.float32)
            buf = self.history[name]
            buf.append(current)
            while len(buf) < self.history_length:
                buf.append(current)
            per_term_flat.append(np.concatenate(list(buf), axis=0))
        return np.concatenate(per_term_flat, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Interactive control state — shared between key_callback and main loop
# ---------------------------------------------------------------------------

class CtrlState:
    """Mutable scalar commands set via keyboard.

    EE pos/orient commands come from the EE mocap bodies (mouse-controlled).
    Body height accepts BOTH mouse drag (on the green disk) AND keyboard:
    keyboard pushes accumulate in `body_height_delta` which the main loop
    applies to the disk's Z each step (additive, so it composes with any
    in-progress mouse drag instead of overwriting it).
    """

    def __init__(self):
        self.waist_alpha = DEFAULT_WAIST_ALPHA
        self.body_height_delta = 0.0  # pending delta (m) to apply to disk Z
        self.reset_requested = False
        self.snapshot_requested = False
        self.quit_requested = False

    def clamp_alpha(self):
        self.waist_alpha = float(np.clip(self.waist_alpha, WAIST_ALPHA_MIN, WAIST_ALPHA_MAX))


def make_key_callback(ctrl: CtrlState):
    """Build a callback bound to the shared CtrlState."""
    # GLFW key codes — `mujoco.viewer` passes them as ints. chr() gives the
    # printable character for ASCII keys (uppercase for letters).
    def cb(keycode):
        try:
            ch = chr(keycode)
        except ValueError:
            ch = ""

        # waist_alpha controls (',' '.' '[' ']')
        if ch == ",":
            ctrl.waist_alpha -= 0.1; ctrl.clamp_alpha()
        elif ch == ".":
            ctrl.waist_alpha += 0.1; ctrl.clamp_alpha()
        elif ch == "[":
            ctrl.waist_alpha -= 0.5; ctrl.clamp_alpha()
        elif ch == "]":
            ctrl.waist_alpha += 0.5; ctrl.clamp_alpha()
        # body_height controls ('-' '=' fine, '_' '+' coarse)
        elif ch == "-":
            ctrl.body_height_delta -= 0.01
        elif ch == "=":
            ctrl.body_height_delta += 0.01
        elif ch == "_":
            ctrl.body_height_delta -= 0.05
        elif ch == "+":
            ctrl.body_height_delta += 0.05
        # one-shot actions
        elif ch in ("R", "r"):
            ctrl.reset_requested = True
        elif ch in ("P", "p"):
            ctrl.snapshot_requested = True
        elif ch in ("Q", "q"):
            ctrl.quit_requested = True

    return cb


# ---------------------------------------------------------------------------
# Mocap target reader — world → pelvis frame transform
# ---------------------------------------------------------------------------

class MocapTargets:
    """Resolves the 3 mocap bodies and computes commands in pelvis frame."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
        self.m = m
        self.d = d
        self.pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY_NAME)
        if self.pelvis_id < 0:
            raise RuntimeError(f"Body '{PELVIS_BODY_NAME}' not found in MuJoCo model.")

        def _mocap_id(name: str) -> int:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise RuntimeError(
                    f"Body '{name}' not found. Are you using scene_interactive.xml?"
                )
            mid = int(m.body_mocapid[bid])
            if mid < 0:
                raise RuntimeError(f"Body '{name}' exists but is not a mocap body.")
            return mid

        self.left_mid = _mocap_id(MOCAP_LEFT)
        self.right_mid = _mocap_id(MOCAP_RIGHT)
        self.height_mid = _mocap_id(MOCAP_HEIGHT)

        self.left_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, LEFT_WRIST_BODY)
        self.right_wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, RIGHT_WRIST_BODY)
        if self.left_wrist_id < 0 or self.right_wrist_id < 0:
            raise RuntimeError(
                f"Wrist bodies '{LEFT_WRIST_BODY}'/'{RIGHT_WRIST_BODY}' not found."
            )

    def reset_to_defaults(self):
        self.d.mocap_pos[self.left_mid] = DEFAULT_MOCAP_LEFT
        self.d.mocap_pos[self.right_mid] = DEFAULT_MOCAP_RIGHT
        self.d.mocap_pos[self.height_mid] = np.array([0.0, 0.0, DEFAULT_BODY_HEIGHT])
        self.d.mocap_quat[self.left_mid] = np.array([1.0, 0.0, 0.0, 0.0])
        self.d.mocap_quat[self.right_mid] = np.array([1.0, 0.0, 0.0, 0.0])
        self.d.mocap_quat[self.height_mid] = np.array([1.0, 0.0, 0.0, 0.0])

    def pelvis_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.d.xpos[self.pelvis_id].astype(np.float32).copy(),
            self.d.xquat[self.pelvis_id].astype(np.float32).copy(),
        )

    def ee_target_in_pelvis(self, mocap_id: int) -> np.ndarray:
        """Return [px py pz qw qx qy qz] of the mocap, expressed in pelvis frame."""
        pelvis_pos, pelvis_quat = self.pelvis_pose()
        target_pos_w = self.d.mocap_pos[mocap_id].astype(np.float32)
        target_quat_w = self.d.mocap_quat[mocap_id].astype(np.float32)

        target_pos_pelvis = quat_rotate_inverse(pelvis_quat, target_pos_w - pelvis_pos)
        target_quat_pelvis = quat_mul(quat_conjugate(pelvis_quat), target_quat_w)
        return np.concatenate([target_pos_pelvis, target_quat_pelvis], dtype=np.float32)

    def actual_ee_in_pelvis(self, wrist_body_id: int) -> np.ndarray:
        pelvis_pos, pelvis_quat = self.pelvis_pose()
        wrist_pos_w = self.d.xpos[wrist_body_id].astype(np.float32)
        wrist_quat_w = self.d.xquat[wrist_body_id].astype(np.float32)
        wrist_pos_pelvis = quat_rotate_inverse(pelvis_quat, wrist_pos_w - pelvis_pos)
        wrist_quat_pelvis = quat_mul(quat_conjugate(pelvis_quat), wrist_quat_w)
        return np.concatenate([wrist_pos_pelvis, wrist_quat_pelvis], dtype=np.float32)

    def body_height_target(self) -> float:
        return float(self.d.mocap_pos[self.height_mid][2])


# ---------------------------------------------------------------------------
# Live console overlay — ANSI-cleared multi-line status block
# ---------------------------------------------------------------------------

OVERLAY_LINES = 8


def _format_vec3(v: np.ndarray) -> str:
    return f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"


class ConsoleOverlay:
    """Refresh a fixed-height status block in place using ANSI escapes."""

    def __init__(self):
        self._first = True
        # Pre-print blank lines so the first redraw has space to overwrite.
        sys.stdout.write("\n" * OVERLAY_LINES)
        sys.stdout.flush()

    def update(self, lines: List[str]):
        # Pad/truncate to fixed height so cursor math stays consistent.
        if len(lines) < OVERLAY_LINES:
            lines = lines + [""] * (OVERLAY_LINES - len(lines))
        elif len(lines) > OVERLAY_LINES:
            lines = lines[:OVERLAY_LINES]
        # Move cursor up to start of block, then rewrite each line clearing it.
        sys.stdout.write(f"\033[{OVERLAY_LINES}F")  # cursor to col0, N lines up
        for ln in lines:
            sys.stdout.write("\033[K" + ln + "\n")
        sys.stdout.flush()

    def snapshot(self, message: str):
        """Inject a permanent line into scrollback above the overlay."""
        sys.stdout.write(f"\033[{OVERLAY_LINES}F\033[K")
        sys.stdout.write(message + "\n")
        # Re-print blank lines so the overlay still has space below.
        sys.stdout.write("\n" * (OVERLAY_LINES - 1))
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive HV1 V3 deploy in MuJoCo.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument(
        "--xml", type=str,
        default="/home/rabisankar/IsaacLab/deploy/mujoco/scene_interactive.xml",
        help="MJCF path. Must contain mocap bodies left_ee_target / right_ee_target / body_height_target.",
    )
    parser.add_argument("--duration", type=float, default=3600.0)
    parser.add_argument("--base_height", type=float, default=None)
    parser.add_argument("--cmd_lin_x", type=float, default=0.0)
    parser.add_argument("--cmd_lin_y", type=float, default=0.0)
    parser.add_argument("--cmd_ang_z", type=float, default=0.0)
    parser.add_argument("--history_length", type=int, default=5)
    parser.add_argument("--realtime", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    xml_path = args.xml
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
    total_obs_dim_yaml = int(cfg["observation"]["total_dim"])
    obs_terms_yaml = cfg["observation"]["terms"]

    history_length = args.history_length
    single_frame_dims: Dict[str, int] = {}
    obs_term_order: List[str] = []
    for entry in obs_terms_yaml:
        name = entry["name"]
        dim_total = int(entry["dim"])
        if dim_total % history_length != 0:
            raise RuntimeError(
                f"Term '{name}' YAML dim {dim_total} not divisible by history "
                f"length {history_length}."
            )
        single_frame_dims[name] = dim_total // history_length
        obs_term_order.append(name)

    print(f"[interactive] xml         : {xml_path}")
    print(f"[interactive] policy      : {policy_path}")
    print(f"[interactive] sim_dt      : {sim_dt}  decimation={decimation}  policy_dt={sim_dt * decimation}")
    print(f"[interactive] action_dim  : {action_dim}    obs_total={total_obs_dim_yaml}")

    # MuJoCo model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    mj_joint_names = list_mujoco_joint_names(m)
    if len(mj_joint_names) != n_dof:
        raise RuntimeError(
            f"MuJoCo joint count {len(mj_joint_names)} != YAML num_dof_total {n_dof}. "
            f"Did the scene XML add extra joints? Mocap bodies are mocap-only "
            f"(no joint), they should not affect this count."
        )

    isaac_to_mj, mj_to_isaac, action_to_isaac = build_joint_maps(
        joint_names_isaac, joint_names_action, mj_joint_names
    )

    kp_mj = kp_isaac[mj_to_isaac]
    kd_mj = kd_isaac[mj_to_isaac]
    q_default_mj = q_default_isaac[mj_to_isaac]

    tau_limit_mj = np.zeros(n_dof, dtype=np.float32)
    for j_mj, name in enumerate(mj_joint_names):
        for a in range(m.nu):
            if m.actuator_trnid[a, 0] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name):
                tau_limit_mj[j_mj] = max(abs(m.actuator_ctrlrange[a, 0]), abs(m.actuator_ctrlrange[a, 1]))
                break
        else:
            tau_limit_mj[j_mj] = 1e6

    policy = torch.jit.load(policy_path, map_location="cpu").eval()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[interactive] policy      : {n_params:,} params")

    set_default_pose(m, d, q_default_isaac, isaac_to_mj, args.base_height)

    obs_builder = V3ObsBuilder(
        m=m, d=d,
        isaac_to_mj=isaac_to_mj,
        action_dim=action_dim,
        n_dof_total=n_dof,
        history_length=history_length,
        obs_term_order=obs_term_order,
        single_frame_dims=single_frame_dims,
    )
    obs_builder.q_default_isaac = q_default_isaac.copy()
    obs_builder.vel_cmd = np.array([args.cmd_lin_x, args.cmd_lin_y, args.cmd_ang_z], dtype=np.float32)

    targets = MocapTargets(m, d)
    targets.reset_to_defaults()
    mujoco.mj_forward(m, d)  # ensure xpos/xquat reflect mocap reset

    ctrl = CtrlState()
    key_cb = make_key_callback(ctrl)

    # Verify first obs dim before launching viewer
    obs_builder.body_height_cmd = np.array([targets.body_height_target()], dtype=np.float32)
    obs_builder.waist_alpha_cmd = np.array([ctrl.waist_alpha], dtype=np.float32)
    obs_builder.left_ee_cmd = targets.ee_target_in_pelvis(targets.left_mid)
    obs_builder.right_ee_cmd = targets.ee_target_in_pelvis(targets.right_mid)
    first_obs = obs_builder.step()
    if first_obs.shape[0] != total_obs_dim_yaml:
        raise RuntimeError(
            f"First obs dim {first_obs.shape[0]} != expected {total_obs_dim_yaml}."
        )
    print(f"[interactive] first obs   : {first_obs.shape[0]} ✓ (matches YAML)")
    obs_builder.reset_history()

    print()
    print("Controls:")
    print("  Mouse (EE targets, red/blue spheres):  double-click to select, then")
    print("          Ctrl + right-drag = translate in screen plane")
    print("          Ctrl + Shift + right-drag = translate along camera depth")
    print("          Ctrl + left-drag = rotate")
    print("  Keys:   = / -   body_height +/- 0.01 m       + / _   body_height +/- 0.05 m")
    print("          , / .   waist_alpha   +/- 0.1        [ / ]   waist_alpha   +/- 0.5")
    print("          r       reset all targets+pose       p       print snapshot to scrollback")
    print("          ESC     quit")
    print()
    print("  Note: the green disk shows the current body-height target.  Use '=' / '-'")
    print("        to nudge it vertically reliably (mouse drag direction depends on")
    print("        camera angle).")
    print()

    target_dof_pos_mj = q_default_mj.copy()
    overlay = ConsoleOverlay()
    counter = 0
    last_overlay_time = 0.0
    OVERLAY_PERIOD_S = 0.20  # 5 Hz

    with mujoco.viewer.launch_passive(m, d, key_callback=key_cb) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < args.duration:
            step_start = time.time()

            if ctrl.quit_requested:
                break
            if ctrl.reset_requested:
                ctrl.reset_requested = False
                ctrl.waist_alpha = DEFAULT_WAIST_ALPHA
                ctrl.body_height_delta = 0.0
                targets.reset_to_defaults()
                obs_builder.reset_history()
                set_default_pose(m, d, q_default_isaac, isaac_to_mj, args.base_height)
                target_dof_pos_mj = q_default_mj.copy()
                overlay.snapshot(f"[t={d.time:6.2f}s]  ** RESET **  targets and pose restored to defaults")

            # Apply any keyboard-accumulated body-height delta to the mocap
            # disk's Z. Additive so it composes with mouse drag in the same
            # session. Clamped to the training-distribution range plus a
            # little headroom so the slider can't go absurdly out of bounds.
            if abs(ctrl.body_height_delta) > 1e-9:
                current_z = float(d.mocap_pos[targets.height_mid][2])
                new_z = float(np.clip(current_z + ctrl.body_height_delta,
                                      BODY_HEIGHT_MIN, BODY_HEIGHT_MAX))
                d.mocap_pos[targets.height_mid][2] = new_z
                ctrl.body_height_delta = 0.0

            # PD control every physics step
            q_mj = d.qpos[7:]
            dq_mj = d.qvel[6:]
            tau = kp_mj * (target_dof_pos_mj - q_mj) - kd_mj * dq_mj
            tau = np.clip(tau, -tau_limit_mj, tau_limit_mj)
            d.qfrc_applied[6:] = tau
            mujoco.mj_step(m, d)
            counter += 1

            if not np.all(np.isfinite(d.qpos)):
                overlay.snapshot(f"[t={d.time:6.2f}s]  !! qpos NaN at step {counter} — stopping")
                break

            # Policy every `decimation` physics steps
            if counter % decimation == 0:
                # Refresh command channels from current mocap state
                obs_builder.left_ee_cmd = targets.ee_target_in_pelvis(targets.left_mid)
                obs_builder.right_ee_cmd = targets.ee_target_in_pelvis(targets.right_mid)
                obs_builder.body_height_cmd = np.array([targets.body_height_target()], dtype=np.float32)
                obs_builder.waist_alpha_cmd = np.array([ctrl.waist_alpha], dtype=np.float32)

                obs = obs_builder.step()
                with torch.no_grad():
                    action_t = policy(torch.from_numpy(obs).unsqueeze(0))
                action = action_t.detach().numpy().squeeze(0).astype(np.float32)
                obs_builder.last_action = action.copy()

                target_dof_pos_isaac = q_default_isaac.copy()
                if use_default_offset:
                    target_dof_pos_isaac[action_to_isaac] = q_default_isaac[action_to_isaac] + action_scale * action
                else:
                    target_dof_pos_isaac[action_to_isaac] = action_scale * action
                target_dof_pos_mj = target_dof_pos_isaac[mj_to_isaac]

            viewer.sync()

            # ---- Live overlay --------------------------------------------
            now = time.time()
            if now - last_overlay_time > OVERLAY_PERIOD_S:
                last_overlay_time = now

                body_h_cmd = targets.body_height_target()
                body_h_actual = float(d.qpos[2])
                body_h_err_cm = abs(body_h_actual - body_h_cmd) * 100

                left_cmd = obs_builder.left_ee_cmd
                right_cmd = obs_builder.right_ee_cmd
                left_act = targets.actual_ee_in_pelvis(targets.left_wrist_id)
                right_act = targets.actual_ee_in_pelvis(targets.right_wrist_id)

                left_pos_err = float(np.linalg.norm(left_act[:3] - left_cmd[:3])) * 100
                right_pos_err = float(np.linalg.norm(right_act[:3] - right_cmd[:3])) * 100
                left_orient_err = np.degrees(quat_angle_error(left_act[3:7], left_cmd[3:7]))
                right_orient_err = np.degrees(quat_angle_error(right_act[3:7], right_cmd[3:7]))

                # base pitch from pelvis projected gravity
                pg = projected_gravity_in_frame(d.qpos[3:7].astype(np.float32))
                base_pitch_rad = float(np.arctan2(pg[0], -pg[2]))

                # action / tau magnitudes
                act_max = float(np.abs(obs_builder.last_action).max())
                tau_max = float(np.abs(tau).max())

                lines = [
                    f"\033[1m[interactive t={d.time:7.2f}s  steps={counter:>7d}]\033[0m   "
                    f"|action|={act_max:.2f}  |tau|={tau_max:>5.1f}N⋅m  "
                    f"base_pitch={np.degrees(base_pitch_rad):+5.1f}°",
                    f"  body_height : cmd={body_h_cmd:.3f}m  actual={body_h_actual:.3f}m  "
                    f"err={body_h_err_cm:5.2f}cm  ('=' / '-' ±1cm  '+'/'_' ±5cm)    "
                    f"waist_alpha={ctrl.waist_alpha:+.2f}  (',/.' ±0.1   '[/]' ±0.5)",
                    f"  L_EE pelvis  cmd={_format_vec3(left_cmd[:3])} m   "
                    f"actual={_format_vec3(left_act[:3])} m",
                    f"               pos_err = \033[33m{left_pos_err:6.2f} cm\033[0m   "
                    f"orient_err = \033[33m{left_orient_err:6.2f}°\033[0m",
                    f"  R_EE pelvis  cmd={_format_vec3(right_cmd[:3])} m   "
                    f"actual={_format_vec3(right_act[:3])} m",
                    f"               pos_err = \033[33m{right_pos_err:6.2f} cm\033[0m   "
                    f"orient_err = \033[33m{right_orient_err:6.2f}°\033[0m",
                    f"  vel_cmd     = [{obs_builder.vel_cmd[0]:+.2f}, {obs_builder.vel_cmd[1]:+.2f}, "
                    f"{obs_builder.vel_cmd[2]:+.2f}]  (lin_x, lin_y, ang_z m/s, rad/s)",
                    f"  drag EE spheres in viewer · keys '=/-' for height · 'r' reset · 'p' snapshot · ESC quit",
                ]
                overlay.update(lines)

                if ctrl.snapshot_requested:
                    ctrl.snapshot_requested = False
                    overlay.snapshot(
                        f"[t={d.time:6.2f}s]  SNAPSHOT  "
                        f"L_err {left_pos_err:.1f}cm/{left_orient_err:.1f}°  "
                        f"R_err {right_pos_err:.1f}cm/{right_orient_err:.1f}°  "
                        f"body_h {body_h_actual:.3f}/{body_h_cmd:.3f}  "
                        f"alpha {ctrl.waist_alpha:.2f}"
                    )

            if args.realtime:
                slack = sim_dt - (time.time() - step_start)
                if slack > 0:
                    time.sleep(slack)

    print(f"\n[interactive] simulation ended after {time.time() - start:.1f}s, {counter} physics steps")


if __name__ == "__main__":
    sys.exit(main())
