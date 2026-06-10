# Copyright (c) 2026, Loco-Manip HV1 Sim-to-Sim
# SPDX-License-Identifier: BSD-3-Clause
"""
Interactive HV1 V4 deploy (KMP-residual action) — drag EE targets with the mouse.

This is `deploy_mujoco_hv1_v3_interactive.py` adapted to V4's action class.

V4 differences (everything else is identical to V3 interactive):
  * Action wiring: target = KMP(cmd_16d) + residual * per_joint_scale,
    NOT target = q_default + scalar_scale * action. We load the frozen KMP
    MLP at startup and run it every policy step.
  * Per-joint residual scale (legs 0.25, arms/waist 0.10) instead of a
    scalar action_scale.
  * KMP_OUTPUT_ORDER → action_joint_names permutation. KMP was trained on
    LEG+ARM+WAIST in a fixed order; the V4 env may resolve action joints
    in a slightly different order (preserve_order=True keeps it stable
    but the deploy script must still permute to be safe).
  * KMP expects quaternion as [x,y,z,w] (xyzw). IsaacLab pose commands are
    [px,py,pz,qw,qx,qy,qz] (w first). We swap before forward.
  * use_default_offset is meaningless in V4 — the offset is q_prior from
    KMP, which is recomputed each policy step.

V4 reuses the V3 obs builder verbatim (V4 inherits V3 observations).

Interactive controls (unchanged from V3 interactive):
  * MOCAP bodies in scene_interactive.xml provide EE targets.
        - left_ee_target, right_ee_target → EE pose commands (mouse-drag)
        - body_height_target              → green disk for height (mouse + keys)
    Drag EE spheres with Ctrl + right-mouse-drag (translate) or
    Ctrl + left-mouse-drag (rotate).
  * Body height keyboard: "=" / "-" ±0.01 m,  "+" / "_" ±0.05 m
  * Waist alpha keyboard: ","/"." ±0.1,  "["/"]" ±0.5
  * "r" reset, "p" snapshot, ESC quit
  * Live console overlay (5 Hz) — same layout as V3 interactive,
    plus a `|q_prior|` field so you can see KMP is being called.

Usage:
    python deploy_mujoco_hv1_v4_interactive.py \\
        --config logs/rsl_rl/<v4_run>/exported/mujoco_config_v4.yaml \\
        --xml    /home/rabisankar/IsaacLab/deploy/mujoco/scene_interactive.xml
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


# Make the KMP class importable for torch.load (the checkpoint dict references
# scripts/hv1/kmp/kmp_model.py:KMP). Same trick kmp_action.py uses.
_KMP_SCRIPTS_DIR = "/home/rabisankar/IsaacLab/scripts/hv1/kmp"
if _KMP_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _KMP_SCRIPTS_DIR)
from kmp_model import KMP  # noqa: E402


# ---------------------------------------------------------------------------
# Quaternion helpers — Isaac convention (w, x, y, z)
# ---------------------------------------------------------------------------

def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qw = q[0]
    qvec = q[1:4]
    a = v * (2.0 * qw * qw - 1.0)
    b = np.cross(qvec, v) * (2.0 * qw)
    c = qvec * (2.0 * np.dot(qvec, v))
    return a - b + c


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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

DEFAULT_MOCAP_LEFT = np.array([0.20, 0.25, 1.00], dtype=np.float32)
DEFAULT_MOCAP_RIGHT = np.array([0.20, -0.25, 1.00], dtype=np.float32)
DEFAULT_BODY_HEIGHT = 0.90
DEFAULT_WAIST_ALPHA = 1.0

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
# Obs builder — UNCHANGED from V3 interactive (V4 inherits the V3 obs group)
# ---------------------------------------------------------------------------

class V3ObsBuilder:
    """Builds V3/V4 single-frame observation and maintains per-term history."""

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
# Interactive control state (UNCHANGED from V3)
# ---------------------------------------------------------------------------

class CtrlState:
    def __init__(self):
        self.waist_alpha = DEFAULT_WAIST_ALPHA
        self.body_height_delta = 0.0
        self.reset_requested = False
        self.snapshot_requested = False
        self.quit_requested = False

    def clamp_alpha(self):
        self.waist_alpha = float(np.clip(self.waist_alpha, WAIST_ALPHA_MIN, WAIST_ALPHA_MAX))


def make_key_callback(ctrl: CtrlState):
    def cb(keycode):
        try:
            ch = chr(keycode)
        except ValueError:
            ch = ""
        if ch == ",":
            ctrl.waist_alpha -= 0.1; ctrl.clamp_alpha()
        elif ch == ".":
            ctrl.waist_alpha += 0.1; ctrl.clamp_alpha()
        elif ch == "[":
            ctrl.waist_alpha -= 0.5; ctrl.clamp_alpha()
        elif ch == "]":
            ctrl.waist_alpha += 0.5; ctrl.clamp_alpha()
        elif ch == "-":
            ctrl.body_height_delta -= 0.01
        elif ch == "=":
            ctrl.body_height_delta += 0.01
        elif ch == "_":
            ctrl.body_height_delta -= 0.05
        elif ch == "+":
            ctrl.body_height_delta += 0.05
        elif ch in ("R", "r"):
            ctrl.reset_requested = True
        elif ch in ("P", "p"):
            ctrl.snapshot_requested = True
        elif ch in ("Q", "q"):
            ctrl.quit_requested = True
    return cb


# ---------------------------------------------------------------------------
# Mocap target reader (UNCHANGED from V3)
# ---------------------------------------------------------------------------

class MocapTargets:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
        self.m = m
        self.d = d
        self.pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY_NAME)
        if self.pelvis_id < 0:
            raise RuntimeError(f"Body '{PELVIS_BODY_NAME}' not found in MuJoCo model.")

        def _mocap_id(name: str) -> int:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise RuntimeError(f"Body '{name}' not found. Are you using scene_interactive.xml?")
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
# KMP runtime — V4-specific
# ---------------------------------------------------------------------------

class KMPRuntime:
    """Wraps the frozen KMP MLP + the joint-order permutation.

    Reproduces the runtime side of `kmp_action.py:process_actions` so the
    sim2sim behaviour matches Isaac exactly:

      1. Pack 16-D KMP input from current commands (with qw<->qz tail swap).
      2. Forward KMP -> 28-D q_prior in KMP_OUTPUT_ORDER.
      3. Permute to action joint order so q_prior_action[action_slot] is
         the KMP output for the joint at that action slot.

    The deploy script then computes:
      q_target_action[i] = q_prior_action[i] + scale_per_joint[i] * action[i]
    """

    def __init__(self, kmp_checkpoint: str, kmp_output_order: List[str],
                 action_joint_names: List[str], device: str = "cpu"):
        kmp_set = set(kmp_output_order)
        act_set = set(action_joint_names)
        missing = kmp_set - act_set
        if missing:
            raise RuntimeError(
                f"KMP outputs joints not in action set: {sorted(missing)}. "
                f"Check the YAML's action.joint_names_action_order vs action.kmp_output_order."
            )
        extra = act_set - kmp_set
        if extra:
            # OK as long as KMP covers ALL 28 action joints. If extra joints
            # exist in action but not KMP, q_prior leaves them at 0 → policy
            # residual is solely responsible. Warn loudly.
            print(f"[deploy] WARNING: action joints not in KMP outputs: {sorted(extra)}. "
                  f"q_prior for these joints will be ZERO; residual fully drives them.")

        self.device = device
        self.kmp = KMP.load(kmp_checkpoint, map_location=device).eval()
        for p in self.kmp.parameters():
            p.requires_grad_(False)
        n_params = sum(p.numel() for p in self.kmp.parameters())
        print(f"[deploy] KMP loaded from {kmp_checkpoint}  ({n_params:,} params)")

        # kmp_to_action[i] = action-slot index for the i-th KMP output joint.
        # Permutation: q_prior_action[kmp_to_action] = q_prior_kmp
        self.kmp_to_action = np.array(
            [action_joint_names.index(n) for n in kmp_output_order],
            dtype=np.int64,
        )
        self.action_dim = len(action_joint_names)
        self._last_q_prior_action = np.zeros(self.action_dim, dtype=np.float32)

    def q_prior(self, body_height_cmd: float, left_ee: np.ndarray, right_ee: np.ndarray,
                waist_alpha_cmd: float) -> np.ndarray:
        """One forward pass. Returns 28-D q_prior in ACTION joint order.

        Inputs:
          body_height_cmd: scalar (m)
          left_ee:  [px, py, pz, qw, qx, qy, qz]   (Isaac wxyz convention)
          right_ee: same
          waist_alpha_cmd: scalar

        Packs to KMP layout (matches kmp_action.py:139-147):
          [h, lxyz, l_qxqyqz, l_qw, rxyz, r_qxqyqz, r_qw, alpha]
          i.e. quaternion as xyzw (KMP training convention).
        """
        kmp_in = np.empty(16, dtype=np.float32)
        kmp_in[0]    = body_height_cmd
        kmp_in[1:4]  = left_ee[0:3]
        kmp_in[4:7]  = left_ee[4:7]   # qx, qy, qz
        kmp_in[7]    = left_ee[3]     # qw (Isaac qw is index 3 -> KMP tail slot)
        kmp_in[8:11] = right_ee[0:3]
        kmp_in[11:14]= right_ee[4:7]
        kmp_in[14]   = right_ee[3]
        kmp_in[15]   = waist_alpha_cmd

        with torch.no_grad():
            t = torch.from_numpy(kmp_in).unsqueeze(0)
            out = self.kmp(t).squeeze(0).cpu().numpy().astype(np.float32)

        q_prior_action = np.zeros(self.action_dim, dtype=np.float32)
        q_prior_action[self.kmp_to_action] = out
        self._last_q_prior_action = q_prior_action
        return q_prior_action

    @property
    def last_q_prior_action(self) -> np.ndarray:
        return self._last_q_prior_action


# ---------------------------------------------------------------------------
# Console overlay (UNCHANGED from V3)
# ---------------------------------------------------------------------------

OVERLAY_LINES = 9   # +1 line vs V3 to display q_prior magnitude


def _format_vec3(v: np.ndarray) -> str:
    return f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"


class ConsoleOverlay:
    def __init__(self):
        self._first = True
        sys.stdout.write("\n" * OVERLAY_LINES)
        sys.stdout.flush()

    def update(self, lines: List[str]):
        if len(lines) < OVERLAY_LINES:
            lines = lines + [""] * (OVERLAY_LINES - len(lines))
        elif len(lines) > OVERLAY_LINES:
            lines = lines[:OVERLAY_LINES]
        sys.stdout.write(f"\033[{OVERLAY_LINES}F")
        for ln in lines:
            sys.stdout.write("\033[K" + ln + "\n")
        sys.stdout.flush()

    def snapshot(self, message: str):
        sys.stdout.write(f"\033[{OVERLAY_LINES}F\033[K")
        sys.stdout.write(message + "\n")
        sys.stdout.write("\n" * (OVERLAY_LINES - 1))
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive HV1 V4 (KMP-residual) deploy in MuJoCo.")
    parser.add_argument("--config", type=str, required=True, help="Path to mujoco_config_v4.yaml.")
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--kmp", type=str, default=None,
                        help="Override path to KMP checkpoint (default: YAML action.kmp_checkpoint).")
    parser.add_argument(
        "--xml", type=str,
        default="/home/rabisankar/IsaacLab/deploy/mujoco/scene_interactive.xml",
        help="MJCF path. Must contain mocap bodies left_ee_target/right_ee_target/body_height_target.",
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

    schema = cfg.get("meta", {}).get("schema", "v3")
    if schema != "v4":
        print(f"[deploy] WARNING: YAML schema is '{schema}', expected 'v4'. "
              "Did you mean to use deploy_mujoco_hv1_v3_interactive.py?")

    xml_path = args.xml
    policy_path = args.policy or cfg["policy"]["jit_path"]
    sim_dt = float(cfg["control"]["sim_dt"])
    decimation = int(cfg["control"]["decimation"])
    joint_names_isaac = cfg["robot"]["joint_names_isaac_order"]
    joint_names_action = cfg["action"]["joint_names_action_order"]
    q_default_isaac = np.array(cfg["robot"]["default_joint_pos"], dtype=np.float32)
    kp_isaac = np.array(cfg["robot"]["kp"], dtype=np.float32)
    kd_isaac = np.array(cfg["robot"]["kd"], dtype=np.float32)

    # V4-specific action fields
    scale_per_joint = np.array(cfg["action"]["scale_per_joint"], dtype=np.float32)
    if scale_per_joint.shape[0] != len(joint_names_action):
        raise RuntimeError(
            f"scale_per_joint len {scale_per_joint.shape[0]} != "
            f"joint_names_action len {len(joint_names_action)}."
        )
    kmp_ckpt = args.kmp or cfg["action"]["kmp_checkpoint"]
    kmp_output_order = cfg["action"]["kmp_output_order"]
    residual_scale = cfg["action"].get("residual_scale", None)
    if residual_scale is not None:
        print(f"[deploy] residual_scale (uniform) = {residual_scale} — overrides per-joint")
        scale_per_joint = np.full_like(scale_per_joint, float(residual_scale))

    n_dof = int(cfg["robot"]["num_dof_total"])
    action_dim = int(cfg["action"]["dim"])
    if action_dim != 28:
        raise RuntimeError(f"V4 expects action_dim=28; YAML says {action_dim}.")
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

    print(f"[v4-interactive] xml         : {xml_path}")
    print(f"[v4-interactive] policy      : {policy_path}")
    print(f"[v4-interactive] kmp         : {kmp_ckpt}")
    print(f"[v4-interactive] sim_dt      : {sim_dt}  decimation={decimation}  policy_dt={sim_dt * decimation}")
    print(f"[v4-interactive] action_dim  : {action_dim}    obs_total={total_obs_dim_yaml}")
    print(f"[v4-interactive] scale range : [{scale_per_joint.min():.3f}, {scale_per_joint.max():.3f}]")

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    mj_joint_names = list_mujoco_joint_names(m)
    if len(mj_joint_names) != n_dof:
        raise RuntimeError(
            f"MuJoCo joint count {len(mj_joint_names)} != YAML num_dof_total {n_dof}."
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
    print(f"[v4-interactive] policy      : {n_params:,} params")

    # ---- V4: load the KMP MLP -------------------------------------------
    kmp_rt = KMPRuntime(
        kmp_checkpoint=kmp_ckpt,
        kmp_output_order=kmp_output_order,
        action_joint_names=joint_names_action,
        device="cpu",
    )

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
    mujoco.mj_forward(m, d)

    ctrl = CtrlState()
    key_cb = make_key_callback(ctrl)

    # First-obs dim verification (also primes obs_builder commands)
    obs_builder.body_height_cmd = np.array([targets.body_height_target()], dtype=np.float32)
    obs_builder.waist_alpha_cmd = np.array([ctrl.waist_alpha], dtype=np.float32)
    obs_builder.left_ee_cmd = targets.ee_target_in_pelvis(targets.left_mid)
    obs_builder.right_ee_cmd = targets.ee_target_in_pelvis(targets.right_mid)
    first_obs = obs_builder.step()
    if first_obs.shape[0] != total_obs_dim_yaml:
        raise RuntimeError(
            f"First obs dim {first_obs.shape[0]} != expected {total_obs_dim_yaml}."
        )
    print(f"[v4-interactive] first obs   : {first_obs.shape[0]} ✓ (matches YAML)")
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

    # First policy target uses q_prior at the initial command set.
    initial_q_prior = kmp_rt.q_prior(
        targets.body_height_target(),
        obs_builder.left_ee_cmd,
        obs_builder.right_ee_cmd,
        ctrl.waist_alpha,
    )
    # No residual yet → target = q_prior in action slots, q_default elsewhere.
    target_dof_pos_isaac = q_default_isaac.copy()
    target_dof_pos_isaac[action_to_isaac] = initial_q_prior
    target_dof_pos_mj = target_dof_pos_isaac[mj_to_isaac]

    overlay = ConsoleOverlay()
    counter = 0
    last_overlay_time = 0.0
    OVERLAY_PERIOD_S = 0.20

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
                # Re-prime target with KMP at default commands so we don't lurch.
                obs_builder.left_ee_cmd = targets.ee_target_in_pelvis(targets.left_mid)
                obs_builder.right_ee_cmd = targets.ee_target_in_pelvis(targets.right_mid)
                qp = kmp_rt.q_prior(
                    targets.body_height_target(),
                    obs_builder.left_ee_cmd,
                    obs_builder.right_ee_cmd,
                    ctrl.waist_alpha,
                )
                target_dof_pos_isaac = q_default_isaac.copy()
                target_dof_pos_isaac[action_to_isaac] = qp
                target_dof_pos_mj = target_dof_pos_isaac[mj_to_isaac]
                overlay.snapshot(f"[t={d.time:6.2f}s]  ** RESET **  targets and pose restored to defaults")

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
                obs_builder.left_ee_cmd = targets.ee_target_in_pelvis(targets.left_mid)
                obs_builder.right_ee_cmd = targets.ee_target_in_pelvis(targets.right_mid)
                obs_builder.body_height_cmd = np.array([targets.body_height_target()], dtype=np.float32)
                obs_builder.waist_alpha_cmd = np.array([ctrl.waist_alpha], dtype=np.float32)

                obs = obs_builder.step()
                with torch.no_grad():
                    action_t = policy(torch.from_numpy(obs).unsqueeze(0))
                action = action_t.detach().numpy().squeeze(0).astype(np.float32)
                obs_builder.last_action = action.copy()

                # === V4: q_target = KMP(cmd) + residual * per_joint_scale ===
                q_prior_action = kmp_rt.q_prior(
                    float(obs_builder.body_height_cmd[0]),
                    obs_builder.left_ee_cmd,
                    obs_builder.right_ee_cmd,
                    float(obs_builder.waist_alpha_cmd[0]),
                )
                target_dof_pos_isaac = q_default_isaac.copy()
                target_dof_pos_isaac[action_to_isaac] = q_prior_action + scale_per_joint * action
                target_dof_pos_mj = target_dof_pos_isaac[mj_to_isaac]

            viewer.sync()

            # ---- Live overlay ---------------------------------------------
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

                pg = projected_gravity_in_frame(d.qpos[3:7].astype(np.float32))
                base_pitch_rad = float(np.arctan2(pg[0], -pg[2]))

                act_max = float(np.abs(obs_builder.last_action).max())
                tau_max = float(np.abs(tau).max())
                q_prior_max = float(np.abs(kmp_rt.last_q_prior_action).max())

                lines = [
                    f"\033[1m[v4-interactive t={d.time:7.2f}s  steps={counter:>7d}]\033[0m   "
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
                    f"  KMP         |q_prior|={q_prior_max:.3f} rad  (V4: target = q_prior + residual * scale)",
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
                        f"alpha {ctrl.waist_alpha:.2f}  "
                        f"|q_prior|={q_prior_max:.3f}"
                    )

            if args.realtime:
                slack = sim_dt - (time.time() - step_start)
                if slack > 0:
                    time.sleep(slack)

    print(f"\n[v4-interactive] simulation ended after {time.time() - start:.1f}s, {counter} physics steps")


if __name__ == "__main__":
    sys.exit(main())
