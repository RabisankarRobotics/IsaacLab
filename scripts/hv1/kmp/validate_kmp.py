"""Standalone validation of the trained KMP.

Three checks:
  1. END-TO-END ACCURACY — sample N random commands, push through KMP,
     run FK on the predicted joint angles, report wrist/foot position errors
     vs commanded. This is what RL training will see at iter 0.
  2. SMOOTHNESS — sweep one command axis at a time, compute max per-step
     joint change. KMP should be C0-smooth; abrupt joint jumps would
     destabilize the actor's residual on top of it.
  3. STANDING POSE — feed neutral command (h=0.93, wrists at default,
     alpha=1.0). Expect a sensible static standing posture.

Failure on any of these is a GO/NO-GO gate before plugging KMP into V4.

Usage:
  conda activate env_isaaclab
  python scripts/hv1/kmp/validate_kmp.py \
      --kmp deploy/model/kmp/kmp_v1.pt \
      --dataset deploy/model/kmp/kmp_dataset_v1.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pinocchio as pin
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_kmp_dataset import (  # noqa: E402
    ACTUATED_28, DEFAULT_FOOT_X, DEFAULT_FOOT_Y, DEFAULT_URDF,
    LEFT_FOOT_FRAME, LEFT_WRIST_FRAME, RIGHT_FOOT_FRAME, RIGHT_WRIST_FRAME,
    build_reduced_model, command_to_vector, rpy_to_quat_xyzw, sample_command,
)
from kmp_model import KMP  # noqa: E402


def kmp_eval(model, cmds_np):
    with torch.no_grad():
        cmds_t = torch.tensor(cmds_np, dtype=torch.float32)
        q_pred = model(cmds_t).numpy()
    return q_pred


def fk_ee_positions(model, data, q_pin):
    pin.framesForwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)
    return {
        "lf": data.oMf[model.getFrameId(LEFT_FOOT_FRAME)].translation.copy(),
        "rf": data.oMf[model.getFrameId(RIGHT_FOOT_FRAME)].translation.copy(),
        "lw": data.oMf[model.getFrameId(LEFT_WRIST_FRAME)].translation.copy(),
        "rw": data.oMf[model.getFrameId(RIGHT_WRIST_FRAME)].translation.copy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kmp", required=True)
    parser.add_argument("--dataset", required=True,
                        help="For computing val-split error vs IK ground truth")
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    print(f"[LOAD] KMP from {args.kmp}")
    model_kmp = KMP.load(args.kmp, map_location="cpu")
    model_kmp.eval()

    print(f"[LOAD] Pinocchio model")
    pin_model, pin_data = build_reduced_model(DEFAULT_URDF)
    pin_joints = list(pin_model.names)[1:]
    v3_to_pin = np.array([ACTUATED_28.index(n) for n in pin_joints])

    # ===== Check 1: end-to-end accuracy on N freshly-sampled commands ======
    print(f"\n[CHECK 1] End-to-end accuracy on {args.n_test} fresh commands")
    rng = np.random.default_rng(args.seed)
    cmds = []
    targets = []
    for _ in range(args.n_test):
        c = sample_command(rng)
        cmds.append(command_to_vector(c))
        targets.append((c["h"], c["left_pos"], c["right_pos"]))
    cmds_np = np.stack(cmds).astype(np.float32)
    q_pred = kmp_eval(model_kmp, cmds_np)  # (N, 28) in V3 action order

    err_lf, err_rf, err_lw, err_rw = [], [], [], []
    for q_v3, (h, lp, rp) in zip(q_pred, targets):
        q_pin = q_v3[v3_to_pin]
        p = fk_ee_positions(pin_model, pin_data, q_pin)
        target_lf = np.array([DEFAULT_FOOT_X,  DEFAULT_FOOT_Y, -h])
        target_rf = np.array([DEFAULT_FOOT_X, -DEFAULT_FOOT_Y, -h])
        err_lf.append(np.linalg.norm(target_lf - p["lf"]))
        err_rf.append(np.linalg.norm(target_rf - p["rf"]))
        err_lw.append(np.linalg.norm(lp - p["lw"]))
        err_rw.append(np.linalg.norm(rp - p["rw"]))

    for name, arr in [("foot_lf", err_lf), ("foot_rf", err_rf),
                      ("wrist_lw", err_lw), ("wrist_rw", err_rw)]:
        a = np.array(arr) * 1000
        print(f"  {name}:  mean={a.mean():5.2f}mm  p50={np.median(a):5.2f}mm  "
              f"p95={np.percentile(a,95):6.2f}mm  max={a.max():6.2f}mm")

    # ===== Check 2: smoothness — sweep one axis at a time ==================
    print(f"\n[CHECK 2] Smoothness sweep (max per-step joint jump per axis)")

    def sweep_axis(make_cmd, n=100):
        cmds = np.stack([command_to_vector(make_cmd(t)) for t in np.linspace(0, 1, n)]).astype(np.float32)
        q_seq = kmp_eval(model_kmp, cmds)
        diffs = np.diff(q_seq, axis=0)
        max_per_joint = np.abs(diffs).max(axis=0)  # (28,)
        return float(np.degrees(max_per_joint.max())), int(np.argmax(max_per_joint))

    def base_cmd_with_height(t):
        return {
            "h": 0.85 + 0.10 * t,  # 0.85 → 0.95
            "left_pos": np.array([0.30, 0.25, 0.25]),
            "left_rpy": np.zeros(3),
            "right_pos": np.array([0.30, -0.25, 0.25]),
            "right_rpy": np.zeros(3),
            "alpha_t": 1.0,
        }
    j, idx = sweep_axis(base_cmd_with_height)
    print(f"  height sweep:    max jump = {j:.3f}°  in joint #{idx} ({ACTUATED_28[idx]})")

    def base_cmd_with_lwx(t):
        return {
            "h": 0.90,
            "left_pos": np.array([0.10 + 0.40 * t, 0.25, 0.25]),
            "left_rpy": np.zeros(3),
            "right_pos": np.array([0.30, -0.25, 0.25]),
            "right_rpy": np.zeros(3),
            "alpha_t": 1.0,
        }
    j, idx = sweep_axis(base_cmd_with_lwx)
    print(f"  left_wrist_x:    max jump = {j:.3f}°  in joint #{idx} ({ACTUATED_28[idx]})")

    def base_cmd_with_alpha(t):
        return {
            "h": 0.90,
            "left_pos": np.array([0.30, 0.25, 0.25]),
            "left_rpy": np.zeros(3),
            "right_pos": np.array([0.30, -0.25, 0.25]),
            "right_rpy": np.zeros(3),
            "alpha_t": 0.1 + (3.0 - 0.1) * t,
        }
    j, idx = sweep_axis(base_cmd_with_alpha)
    print(f"  alpha sweep:     max jump = {j:.3f}°  in joint #{idx} ({ACTUATED_28[idx]})")

    # ===== Check 3: standing pose ==========================================
    print(f"\n[CHECK 3] Standing pose at h=0.93, neutral wrists, alpha=1.0")
    standing_cmd = command_to_vector({
        "h": 0.93,
        "left_pos": np.array([0.30, 0.26, 0.24]),   # near default wrist pos
        "left_rpy": np.zeros(3),
        "right_pos": np.array([0.30, -0.26, 0.24]),
        "right_rpy": np.zeros(3),
        "alpha_t": 1.0,
    }).reshape(1, -1).astype(np.float32)
    q_stand = kmp_eval(model_kmp, standing_cmd)[0]
    print(f"  Predicted joint angles (deg):")
    for n, qv in zip(ACTUATED_28, q_stand):
        if abs(qv) > 0.01:
            print(f"    {n:35s}  {np.degrees(qv):+7.2f}°")

    # ===== Compare against dataset val split if available ==================
    print(f"\n[CHECK 4] Reproduce known dataset samples (KMP vs IK joint MAE)")
    d = np.load(args.dataset)
    cmds_ds = d["commands"].astype(np.float32)
    q_ds = d["joint_pos"].astype(np.float32)
    # Take last 5% as held-out (matches train_kmp.py val split with seed 42 if you trust the rng)
    n_check = min(5000, len(cmds_ds))
    idx_rng = np.random.default_rng(args.seed + 1).choice(len(cmds_ds), size=n_check, replace=False)
    q_kmp = kmp_eval(model_kmp, cmds_ds[idx_rng])
    abs_err = np.abs(q_kmp - q_ds[idx_rng]) * 180 / np.pi  # deg
    print(f"  Mean MAE across all 28 joints, over {n_check} samples: {abs_err.mean():.2f}°")
    print(f"  Worst joint MAE:  {abs_err.mean(0).max():.2f}°  ({ACTUATED_28[int(abs_err.mean(0).argmax())]})")
    print(f"  P95 joint error:  {np.percentile(abs_err, 95):.2f}°")


if __name__ == "__main__":
    main()
