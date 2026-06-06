"""Validate a saved KMP IK dataset by re-running FK on the stored joint
positions and comparing to the stored commands.

Catches bugs in the IK pipeline (sign errors, joint reordering, etc.) that
would otherwise quietly corrupt the dataset and the KMP that trains on it.

Usage:
  conda activate env_isaaclab
  python scripts/hv1/kmp/validate_ik.py /tmp/kmp_test.npz
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, "/home/rabisankar/IsaacLab/scripts/hv1/kmp")
from generate_kmp_dataset import (  # noqa: E402
    ACTUATED_28, DEFAULT_FOOT_X, DEFAULT_FOOT_Y, DEFAULT_URDF,
    LEFT_FOOT_FRAME, LEFT_WRIST_FRAME, RIGHT_FOOT_FRAME, RIGHT_WRIST_FRAME,
    build_reduced_model,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path")
    parser.add_argument("--num", type=int, default=20, help="Random samples to inspect")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"[LOAD] {args.npz_path}")
    data_npz = np.load(args.npz_path)
    commands = data_npz["commands"]
    qpos = data_npz["joint_pos"]
    layout = list(data_npz["command_layout"])
    saved_joint_names = list(data_npz["actuated_joint_names"])

    print(f"[INFO] N={len(qpos)}, command dim={commands.shape[1]}, joint dim={qpos.shape[1]}")
    print(f"[INFO] command layout: {layout}")
    assert saved_joint_names == ACTUATED_28, "Saved joint order doesn't match V3 action order"

    # Load model
    model, data = build_reduced_model(DEFAULT_URDF)
    pin_joints = list(model.names)[1:]
    # Map V3-action-order → pin-order
    v3_to_pin = np.array([ACTUATED_28.index(n) for n in pin_joints])

    fids = {
        "lf": model.getFrameId(LEFT_FOOT_FRAME),
        "rf": model.getFrameId(RIGHT_FOOT_FRAME),
        "lw": model.getFrameId(LEFT_WRIST_FRAME),
        "rw": model.getFrameId(RIGHT_WRIST_FRAME),
    }

    # Pick random samples
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(qpos), size=min(args.num, len(qpos)), replace=False)

    foot_err = []
    wrist_err = []
    for i in sample_idx:
        cmd = commands[i]
        q_v3 = qpos[i]
        q_pin = q_v3[v3_to_pin]

        pin.framesForwardKinematics(model, data, q_pin)
        pin.updateFramePlacements(model, data)

        # Commanded targets
        h = cmd[layout.index("h")]
        lx, ly, lz = cmd[1:4]
        rx, ry, rz = cmd[8:11]

        # Actual via FK
        p_lf = data.oMf[fids["lf"]].translation
        p_rf = data.oMf[fids["rf"]].translation
        p_lw = data.oMf[fids["lw"]].translation
        p_rw = data.oMf[fids["rw"]].translation

        target_lf = np.array([DEFAULT_FOOT_X,  DEFAULT_FOOT_Y, -h])
        target_rf = np.array([DEFAULT_FOOT_X, -DEFAULT_FOOT_Y, -h])
        target_lw = np.array([lx, ly, lz])
        target_rw = np.array([rx, ry, rz])

        e_lf = float(np.linalg.norm(target_lf - p_lf))
        e_rf = float(np.linalg.norm(target_rf - p_rf))
        e_lw = float(np.linalg.norm(target_lw - p_lw))
        e_rw = float(np.linalg.norm(target_rw - p_rw))

        foot_err.append(max(e_lf, e_rf))
        wrist_err.append(max(e_lw, e_rw))

        if args.verbose:
            print(f"  [{i:5d}] h={h:.3f} α={cmd[15]:.2f}  "
                  f"foot err={max(e_lf,e_rf)*1000:5.2f}mm  wrist err={max(e_lw,e_rw)*1000:6.2f}mm")

    foot_err = np.array(foot_err)
    wrist_err = np.array(wrist_err)
    print(f"\n[FK CHECK on {len(sample_idx)} samples]")
    print(f"  foot err:  mean={foot_err.mean()*1000:5.2f}mm  max={foot_err.max()*1000:5.2f}mm")
    print(f"  wrist err: mean={wrist_err.mean()*1000:5.2f}mm  max={wrist_err.max()*1000:5.2f}mm")

    # Cross-check with stored ik_err
    if "ik_err_foot" in data_npz.files:
        stored_f = data_npz["ik_err_foot"][sample_idx]
        stored_w = data_npz["ik_err_wrist"][sample_idx]
        df = np.abs(foot_err - stored_f).max()
        dw = np.abs(wrist_err - stored_w).max()
        print(f"  vs stored: |df|max={df*1000:.3f}mm  |dw|max={dw*1000:.3f}mm  (should be ~0)")

    # Sanity check: knee should be positive (bent) for crouch
    knee_l_idx = ACTUATED_28.index("left_knee_joint")
    knee_r_idx = ACTUATED_28.index("right_knee_joint")
    print(f"\n[SANITY] Knee bending (positive = crouch):")
    print(f"  left knee:  mean={np.degrees(qpos[:, knee_l_idx].mean()):.1f}°  "
          f"min={np.degrees(qpos[:, knee_l_idx].min()):.1f}°  "
          f"max={np.degrees(qpos[:, knee_l_idx].max()):.1f}°")
    print(f"  right knee: mean={np.degrees(qpos[:, knee_r_idx].mean()):.1f}°  "
          f"min={np.degrees(qpos[:, knee_r_idx].min()):.1f}°  "
          f"max={np.degrees(qpos[:, knee_r_idx].max()):.1f}°")
    # Quick reachability check on the 5 deepest-crouch and 5 shallowest-crouch samples
    h_col = commands[:, layout.index("h")]
    print(f"\n[BY HEIGHT]")
    for label, mask in [("h<0.87", h_col < 0.87), ("h>0.93", h_col > 0.93)]:
        if mask.sum() == 0:
            continue
        knees = qpos[mask][:, knee_l_idx]
        print(f"  {label} (n={mask.sum()}): knee mean={np.degrees(knees.mean()):.1f}°")

    return 0


if __name__ == "__main__":
    sys.exit(main())
