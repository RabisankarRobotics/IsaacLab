"""Walk through ONE IK solve step-by-step with verbose logging.

Pedagogical: shows what generate_kmp_dataset.py actually computes per iteration
for a single command, so the IK loop is concrete instead of abstract.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_kmp_dataset import (  # noqa: E402
    ACTUATED_28, ARM_JOINTS, DEFAULT_FOOT_X, DEFAULT_FOOT_Y, DEFAULT_URDF,
    LEFT_FOOT_FRAME, LEFT_WRIST_FRAME, LEG_JOINTS, RIGHT_FOOT_FRAME,
    RIGHT_WRIST_FRAME, WAIST_ACTUATED, build_reduced_model, make_q_default,
    rpy_to_rot,
)


def degs(q):
    return np.degrees(q)


def main():
    # ---- Define ONE concrete command ----------------------------------------
    cmd = {
        "h": 0.88,
        "left_pos": np.array([0.35, 0.22, 0.30]),
        "left_rpy": np.array([0.0, 0.0, 0.0]),
        "right_pos": np.array([0.35, -0.22, 0.30]),
        "right_rpy": np.array([0.0, 0.0, 0.0]),
        "alpha_t": 1.0,
    }
    print("=" * 70)
    print("ONE COMMAND -> staged IK -> joint vector")
    print("=" * 70)
    print(f"  body height h     = {cmd['h']} m  (pelvis 88 cm off the ground)")
    print(f"  left  wrist pos   = {cmd['left_pos']}")
    print(f"  right wrist pos   = {cmd['right_pos']}")
    print(f"  alpha_t           = {cmd['alpha_t']}")

    # ---- Build Pinocchio model ----------------------------------------------
    model, data = build_reduced_model(DEFAULT_URDF)
    joints = list(model.names)[1:]
    leg_idx = np.array([joints.index(n) for n in LEG_JOINTS])
    arm_idx = np.array([joints.index(n) for n in ARM_JOINTS])
    waist_idx = np.array([joints.index(n) for n in WAIST_ACTUATED])
    upper_idx = np.concatenate([arm_idx, waist_idx])

    fid_lf = model.getFrameId(LEFT_FOOT_FRAME)
    fid_rf = model.getFrameId(RIGHT_FOOT_FRAME)
    fid_lw = model.getFrameId(LEFT_WRIST_FRAME)
    fid_rw = model.getFrameId(RIGHT_WRIST_FRAME)

    q_default = make_q_default(model)
    knee_l_pin = joints.index("left_knee_joint")
    hip_l_pin = joints.index("left_hip_pitch_joint")
    sh_l_pin = joints.index("left_shoulder_pitch_joint")
    el_l_pin = joints.index("left_elbow_joint")

    # ---- Initial pose check -------------------------------------------------
    pin.framesForwardKinematics(model, data, q_default)
    pin.updateFramePlacements(model, data)
    print(f"\n--- INITIAL POSE (V3 crouch: knee={degs(q_default[knee_l_pin]):.1f}°, "
          f"hip_pitch={degs(q_default[hip_l_pin]):.1f}°) ---")
    print(f"  FK left  foot  z = {data.oMf[fid_lf].translation[2]:+.4f} m   "
          f"(target {-cmd['h']:+.4f})")
    print(f"  FK left  wrist   = {data.oMf[fid_lw].translation.round(3)}    "
          f"(target {cmd['left_pos'].round(3)})")

    # =========================================================================
    # STAGE 1 — solve LEGS for foot position at z = -h
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1 — legs only (12 DOF) → put feet on ground at z = -0.88")
    print("=" * 70)
    p_lf_t = np.array([DEFAULT_FOOT_X,  DEFAULT_FOOT_Y, -cmd["h"]])
    p_rf_t = np.array([DEFAULT_FOOT_X, -DEFAULT_FOOT_Y, -cmd["h"]])

    q = q_default.copy()
    n_leg = len(leg_idx)
    WORLD = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    print(f"\n  {'iter':>4}  {'foot_err_mm':>11}  {'knee_deg':>8}  "
          f"{'hip_pitch_deg':>13}  {'max_dq_deg':>10}")
    for it in range(20):
        pin.framesForwardKinematics(model, data, q)
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)

        H = np.zeros((n_leg, n_leg))
        g = np.zeros(n_leg)
        for fid, p_t, w in [(fid_lf, p_lf_t, 5.0), (fid_rf, p_rf_t, 5.0)]:
            e = p_t - data.oMf[fid].translation
            J3 = pin.getFrameJacobian(model, data, fid, WORLD)[0:3, leg_idx]
            H += w * (J3.T @ J3)
            g += w * (J3.T @ e)
        H += 1e-3 * np.eye(n_leg)
        g += -1e-3 * (q[leg_idx] - q_default[leg_idx])
        dq_leg = np.linalg.solve(H + 1e-4 * np.eye(n_leg), g)
        dq = np.zeros(model.nv); dq[leg_idx] = dq_leg
        q = pin.integrate(model, q, dq)
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

        err_lf = float(np.linalg.norm(p_lf_t - data.oMf[fid_lf].translation))
        if it < 6 or err_lf < 1e-4 or it == 19:
            print(f"  {it:>4d}  {err_lf*1000:>10.3f}  "
                  f"{degs(q[knee_l_pin]):>+7.2f}  "
                  f"{degs(q[hip_l_pin]):>+12.2f}  "
                  f"{degs(np.abs(dq_leg).max()):>9.3f}")
        if err_lf < 1e-4:
            break

    q_after_stage1 = q.copy()
    pin.framesForwardKinematics(model, data, q_after_stage1)
    pin.updateFramePlacements(model, data)
    print(f"\n  STAGE 1 RESULT (legs only):")
    print(f"    left  foot z = {data.oMf[fid_lf].translation[2]:+.4f} m  "
          f"(target {-cmd['h']:+.4f})  err = "
          f"{abs(data.oMf[fid_lf].translation[2] + cmd['h'])*1000:.2f} mm")
    print(f"    left  knee   = {degs(q[knee_l_pin]):+.2f}°  "
          f"(crouches more when h is lower)")
    print(f"    legs are now solved; wrists still untouched")

    # =========================================================================
    # STAGE 2 — solve ARMS+WAIST for wrist SE(3), legs frozen
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2 — arms + waist (16 DOF) → put wrists at commanded poses")
    print("=" * 70)
    T_lw_t = pin.SE3(rpy_to_rot(cmd["left_rpy"]), cmd["left_pos"])
    T_rw_t = pin.SE3(rpy_to_rot(cmd["right_rpy"]), cmd["right_pos"])

    n_up = len(upper_idx)
    posture_diag = np.full(n_up, 1e-3)
    for j, idx in enumerate(upper_idx):
        if idx in waist_idx:
            posture_diag[j] = 1e-3 * cmd["alpha_t"]

    print(f"\n  {'iter':>4}  {'wrist_err_mm':>12}  {'sh_pitch_deg':>12}  "
          f"{'elbow_deg':>9}  {'max_dq_deg':>10}")
    for it in range(40):
        pin.framesForwardKinematics(model, data, q)
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)

        H = np.zeros((n_up, n_up))
        g = np.zeros(n_up)
        for fid, T_t in [(fid_lw, T_lw_t), (fid_rw, T_rw_t)]:
            iMd = data.oMf[fid].inverse() * T_t
            err6 = pin.log(iMd).vector
            J6 = pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL)[:, upper_idx]
            W = np.diag([1.0]*3 + [0.3]*3)
            H += J6.T @ W @ J6
            g += J6.T @ W @ err6
        H += np.diag(posture_diag)
        g += -posture_diag * (q[upper_idx] - q_default[upper_idx])
        dq_upper = np.linalg.solve(H + 1e-4 * np.eye(n_up), g)
        dq = np.zeros(model.nv); dq[upper_idx] = dq_upper
        q = pin.integrate(model, q, dq)
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

        err_lw = float(np.linalg.norm(T_lw_t.translation - data.oMf[fid_lw].translation))
        if it < 6 or err_lw < 1e-4 or it == 39 or it % 5 == 0:
            print(f"  {it:>4d}  {err_lw*1000:>11.3f}  "
                  f"{degs(q[sh_l_pin]):>+11.2f}  "
                  f"{degs(q[el_l_pin]):>+8.2f}  "
                  f"{degs(np.abs(dq_upper).max()):>9.3f}")
        if err_lw < 1e-4:
            break

    # =========================================================================
    # FINAL FK CHECK + the 28-D joint vector this command MAPS TO
    # =========================================================================
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    print("\n" + "=" * 70)
    print("FINAL FK CHECK (was this q the right answer?)")
    print("=" * 70)
    print(f"  left  foot  z = {data.oMf[fid_lf].translation[2]:+.4f} m   "
          f"(target {-cmd['h']:+.4f})  err = "
          f"{abs(data.oMf[fid_lf].translation[2] + cmd['h'])*1000:.2f} mm")
    print(f"  left  wrist   = {data.oMf[fid_lw].translation.round(4)}   "
          f"(target {cmd['left_pos']})  err = "
          f"{np.linalg.norm(cmd['left_pos'] - data.oMf[fid_lw].translation)*1000:.2f} mm")
    print(f"  right wrist   = {data.oMf[fid_rw].translation.round(4)}   "
          f"(target {cmd['right_pos']})  err = "
          f"{np.linalg.norm(cmd['right_pos'] - data.oMf[fid_rw].translation)*1000:.2f} mm")

    # Map pin order -> V3 action order, the format the dataset stores
    pin_to_v3 = np.array([joints.index(n) for n in ACTUATED_28])
    q_v3 = q[pin_to_v3]
    print("\n" + "=" * 70)
    print("THE 28-D JOINT VECTOR THIS COMMAND MAPS TO (V3 action order, degrees)")
    print("=" * 70)
    for n, qv in zip(ACTUATED_28, q_v3):
        marker = "  <-- nonzero" if abs(qv) > 0.05 else ""
        print(f"  {n:35s} {degs(qv):+8.3f}°{marker}")

    # =========================================================================
    # Now check: the KMP MLP, given the same command, should produce ~same q
    # =========================================================================
    print("\n" + "=" * 70)
    print("KMP MLP CHECK — does the trained network reproduce this answer?")
    print("=" * 70)
    try:
        import torch
        from kmp_model import KMP

        kmp_path = "/home/rabisankar/IsaacLab/deploy/model/kmp/kmp_v1.pt"
        kmp = KMP.load(kmp_path, map_location="cpu")
        kmp.eval()
        from generate_kmp_dataset import command_to_vector
        cmd_vec = torch.tensor(command_to_vector(cmd), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_kmp_v3 = kmp(cmd_vec).numpy()[0]
        diff_deg = degs(np.abs(q_kmp_v3 - q_v3))
        print(f"  Mean |IK - KMP| across 28 joints: {diff_deg.mean():.2f}°")
        print(f"  Max  |IK - KMP|:                  {diff_deg.max():.2f}°  "
              f"({ACTUATED_28[int(diff_deg.argmax())]})")
        print(f"  -> KMP is replacing seconds of IK with a single forward pass.")
    except Exception as e:
        print(f"  (KMP check skipped: {e})")


if __name__ == "__main__":
    main()
