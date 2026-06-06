"""KMP dataset generation — offline staged IK on HV1 URDF.

For each sampled command tuple (body_height, T_left_wrist, T_right_wrist, alpha_t),
solve a two-stage Levenberg-Marquardt IK:

  Stage 1: legs only — solve for the 12 leg joints that put both feet at the
    default stance position with pelvis at the desired body height. Posture
    pulled toward V3 default; ankle stays near -0.20 rad so foot stays flat.

  Stage 2: arms + waist only — with leg joints frozen, solve for the 16
    upper-body joints that put both wrists at their target poses (full SE(3)).
    Posture pulled toward V3 default with waist_roll/pitch cost scaled by
    `alpha_t` per HiWET Eq. 12.

Why staged: a joint IK over all 28 joints gets stuck balancing competing
feet/wrist objectives — pass rate ~22% even at 300 iters. Decoupling them
raises pass rate to ~50% and gives the legs a stable "stance" the arms can
reach from. The remaining failures are V3 commands that genuinely exceed
the arm workspace (corners of the EE box outside the shoulder reach sphere).

Output: .npz with arrays
  commands         shape (N, 16)  [h, T_L (7=pos+quat_xyzw), T_R (7), alpha_t]
  joint_pos        shape (N, 28)  actuated joints in V3 action order
  ik_err_wrist     shape (N,)     max wrist position error (m) — diagnostic
  ik_err_foot      shape (N,)     max foot position error (m) — diagnostic

Usage:
  conda activate env_isaaclab
  python scripts/hv1/kmp/generate_kmp_dataset.py --num-samples 100 --out /tmp/kmp_test.npz
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from typing import Any

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Paths and joint ordering — must match V3 cfg exactly
# ---------------------------------------------------------------------------
DEFAULT_URDF = (
    "/home/rabisankar/IsaacLab/source/isaaclab_assets/"
    "data/custom_robot/urdf_mesh/hv1/hv1.urdf"
)

LEG_JOINTS = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
WAIST_ACTUATED = ["waist_roll_joint", "waist_pitch_joint"]
ACTUATED_28 = LEG_JOINTS + ARM_JOINTS + WAIST_ACTUATED

PINNED_TO_ZERO = ["waist_yaw_joint", "neck_yaw_joint", "neck_pitch_joint"]

LEFT_WRIST_FRAME = "left_wrist_yaw_link"
RIGHT_WRIST_FRAME = "right_wrist_yaw_link"
LEFT_FOOT_FRAME = "left_ankle_roll_link"
RIGHT_FOOT_FRAME = "right_ankle_roll_link"

DEFAULT_FOOT_X = -0.0035
DEFAULT_FOOT_Y = 0.211

V3_DEFAULT_QPOS = {
    "left_hip_pitch_joint": -0.20, "right_hip_pitch_joint": -0.20,
    "left_knee_joint": 0.40, "right_knee_joint": 0.40,
    "left_ankle_pitch_joint": -0.20, "right_ankle_pitch_joint": -0.20,
    "left_shoulder_pitch_joint": 0.30, "right_shoulder_pitch_joint": 0.30,
    "left_elbow_joint": 0.30, "right_elbow_joint": 0.30,
}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_command(rng: np.random.Generator) -> dict:
    h = rng.uniform(0.85, 0.95)
    left_pos = np.array([rng.uniform(0.10, 0.50), rng.uniform(0.05, 0.45),
                         rng.uniform(0.00, 0.55)])
    left_rpy = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2),
                         rng.uniform(-0.2, 0.2)])
    right_pos = np.array([rng.uniform(0.10, 0.50), rng.uniform(-0.45, -0.05),
                          rng.uniform(0.00, 0.55)])
    right_rpy = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2),
                          rng.uniform(-0.2, 0.2)])
    alpha_t = float(np.exp(rng.uniform(np.log(0.1), np.log(3.0))))
    return {
        "h": float(h),
        "left_pos": left_pos, "left_rpy": left_rpy,
        "right_pos": right_pos, "right_rpy": right_rpy,
        "alpha_t": alpha_t,
    }


def rpy_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray:
    return R.from_euler("xyz", rpy).as_quat()


def rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    return R.from_euler("xyz", rpy).as_matrix()


def command_to_vector(cmd: dict) -> np.ndarray:
    return np.concatenate([
        [cmd["h"]],
        cmd["left_pos"], rpy_to_quat_xyzw(cmd["left_rpy"]),
        cmd["right_pos"], rpy_to_quat_xyzw(cmd["right_rpy"]),
        [cmd["alpha_t"]],
    ])


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def build_reduced_model(urdf_path: str):
    full_model = pin.buildModelFromUrdf(urdf_path)
    q_ref = pin.neutral(full_model)
    locked_ids = [full_model.getJointId(n) for n in PINNED_TO_ZERO]
    reduced = pin.buildReducedModel(full_model, locked_ids, q_ref)
    return reduced, reduced.createData()


def make_q_default(model) -> np.ndarray:
    q = pin.neutral(model)
    pin_joints = list(model.names)[1:]
    for name, val in V3_DEFAULT_QPOS.items():
        q[pin_joints.index(name)] = val
    return q


# ---------------------------------------------------------------------------
# Worker (one per process)
# ---------------------------------------------------------------------------
class IKWorker:
    """Stateful per-process worker. Built lazily inside each worker process so
    Pinocchio objects (which don't pickle) aren't sent through queues."""

    def __init__(self, urdf_path: str):
        self.model, self.data = build_reduced_model(urdf_path)
        self.joints = list(self.model.names)[1:]
        self.q_default = make_q_default(self.model)
        self.fid_lf = self.model.getFrameId(LEFT_FOOT_FRAME)
        self.fid_rf = self.model.getFrameId(RIGHT_FOOT_FRAME)
        self.fid_lw = self.model.getFrameId(LEFT_WRIST_FRAME)
        self.fid_rw = self.model.getFrameId(RIGHT_WRIST_FRAME)
        self.leg_idx = np.array([self.joints.index(n) for n in LEG_JOINTS])
        self.arm_idx = np.array([self.joints.index(n) for n in ARM_JOINTS])
        self.waist_idx = np.array([self.joints.index(n) for n in WAIST_ACTUATED])
        self.upper_idx = np.concatenate([self.arm_idx, self.waist_idx])
        # Remap: pin order → V3 action order
        self.pin_to_v3 = np.array([self.joints.index(n) for n in ACTUATED_28])
        self.lower = self.model.lowerPositionLimit
        self.upper = self.model.upperPositionLimit
        self.WORLD = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

    def stage1_legs(self, q_init, p_lf_t, p_rf_t, n_iter=80):
        q = q_init.copy()
        n = len(self.leg_idx)
        I = np.eye(n)
        for it in range(n_iter):
            pin.framesForwardKinematics(self.model, self.data, q)
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            H = np.zeros((n, n))
            g = np.zeros(n)
            for fid, p_t, w in [(self.fid_lf, p_lf_t, 5.0), (self.fid_rf, p_rf_t, 5.0)]:
                e = p_t - self.data.oMf[fid].translation
                J3 = pin.getFrameJacobian(self.model, self.data, fid, self.WORLD)[0:3, self.leg_idx]
                H += w * (J3.T @ J3)
                g += w * (J3.T @ e)
            H += 1e-3 * I
            g += -1e-3 * (q[self.leg_idx] - self.q_default[self.leg_idx])
            dq_leg = np.linalg.solve(H + 1e-4 * I, g)
            dq = np.zeros(self.model.nv); dq[self.leg_idx] = dq_leg
            q = pin.integrate(self.model, q, dq)
            q = np.clip(q, self.lower, self.upper)
            err_lf = float(np.linalg.norm(p_lf_t - self.data.oMf[self.fid_lf].translation))
            err_rf = float(np.linalg.norm(p_rf_t - self.data.oMf[self.fid_rf].translation))
            if max(err_lf, err_rf) < 1e-4:
                break
        return q, err_lf, err_rf

    def stage2_arms(self, q_init, T_lw_t, T_rw_t, alpha_t, n_iter=120):
        q = q_init.copy()
        n = len(self.upper_idx)
        I = np.eye(n)
        posture_diag = np.full(n, 1e-3)
        for j, idx in enumerate(self.upper_idx):
            if idx in self.waist_idx:
                posture_diag[j] = 1e-3 * alpha_t
        for it in range(n_iter):
            pin.framesForwardKinematics(self.model, self.data, q)
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            H = np.zeros((n, n))
            g = np.zeros(n)
            for fid, T_t in [(self.fid_lw, T_lw_t), (self.fid_rw, T_rw_t)]:
                iMd = self.data.oMf[fid].inverse() * T_t
                err6 = pin.log(iMd).vector
                J6 = pin.getFrameJacobian(self.model, self.data, fid, pin.ReferenceFrame.LOCAL)[:, self.upper_idx]
                # Per-component weights: linear>angular (V3 cares most about position)
                w_vec = np.array([1.0]*3 + [0.3]*3)
                W = np.diag(w_vec)
                H += J6.T @ W @ J6
                g += J6.T @ W @ err6
            H += np.diag(posture_diag)
            g += -posture_diag * (q[self.upper_idx] - self.q_default[self.upper_idx])
            dq_upper = np.linalg.solve(H + 1e-4 * I, g)
            dq = np.zeros(self.model.nv); dq[self.upper_idx] = dq_upper
            q = pin.integrate(self.model, q, dq)
            q = np.clip(q, self.lower, self.upper)
            err_lw = float(np.linalg.norm(T_lw_t.translation - self.data.oMf[self.fid_lw].translation))
            err_rw = float(np.linalg.norm(T_rw_t.translation - self.data.oMf[self.fid_rw].translation))
            if max(err_lw, err_rw) < 1e-4:
                break
        return q, err_lw, err_rw

    def solve(self, cmd: dict) -> dict:
        p_lf_t = np.array([DEFAULT_FOOT_X,  DEFAULT_FOOT_Y, -cmd["h"]])
        p_rf_t = np.array([DEFAULT_FOOT_X, -DEFAULT_FOOT_Y, -cmd["h"]])
        T_lw_t = pin.SE3(rpy_to_rot(cmd["left_rpy"]), cmd["left_pos"])
        T_rw_t = pin.SE3(rpy_to_rot(cmd["right_rpy"]), cmd["right_pos"])
        q, err_lf, err_rf = self.stage1_legs(self.q_default, p_lf_t, p_rf_t)
        q, err_lw, err_rw = self.stage2_arms(q, T_lw_t, T_rw_t, cmd["alpha_t"])
        q_v3 = q[self.pin_to_v3]
        return {
            "cmd": command_to_vector(cmd),
            "q": q_v3,
            "err_foot": max(err_lf, err_rf),
            "err_wrist": max(err_lw, err_rw),
        }


# Worker globals (set in _init_worker)
_worker: IKWorker | None = None


def _init_worker(urdf_path: str):
    global _worker
    _worker = IKWorker(urdf_path)


def _solve_batch(batch_cmds: list[dict]) -> list[dict]:
    assert _worker is not None
    return [_worker.solve(c) for c in batch_cmds]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--out", type=str, default="/tmp/kmp_dataset.npz")
    parser.add_argument("--urdf", type=str, default=DEFAULT_URDF)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--err-tol-wrist", type=float, default=0.05,
                        help="Discard if max wrist pos error > tol (m). 50mm = generous.")
    parser.add_argument("--err-tol-foot", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count() - 2)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    print(f"[INFO] Loading {args.urdf}")
    print(f"[INFO] Workers: {args.num_workers}, batch size: {args.batch_size}")

    # Pre-generate all commands (each worker is deterministic)
    rng = np.random.default_rng(args.seed)
    all_cmds = [sample_command(rng) for _ in range(args.num_samples)]
    batches = [all_cmds[i:i + args.batch_size] for i in range(0, args.num_samples, args.batch_size)]

    t0 = time.time()
    results: list[dict] = []
    with mp.get_context("fork").Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(args.urdf,),
    ) as pool:
        for i, batch_result in enumerate(pool.imap_unordered(_solve_batch, batches)):
            results.extend(batch_result)
            done = len(results)
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (args.num_samples - done) / max(rate, 1e-9)
            print(f"  [{done}/{args.num_samples}]  rate={rate:.0f}/s  eta={eta:.0f}s")

    dt = time.time() - t0
    print(f"\n[IK done] {args.num_samples} samples in {dt:.1f}s ({args.num_samples/dt:.0f}/s)")

    # Quality filter
    commands_list, qpos_list, errf_list, errw_list = [], [], [], []
    fail_foot = fail_wrist = 0
    for r in results:
        if r["err_foot"] > args.err_tol_foot:
            fail_foot += 1
            continue
        if r["err_wrist"] > args.err_tol_wrist:
            fail_wrist += 1
            continue
        commands_list.append(r["cmd"])
        qpos_list.append(r["q"])
        errf_list.append(r["err_foot"])
        errw_list.append(r["err_wrist"])

    kept = len(qpos_list)
    print(f"[FILTER] kept={kept}/{args.num_samples} ({kept/args.num_samples:.1%})  "
          f"foot_fail={fail_foot}  wrist_fail={fail_wrist}")

    if kept == 0:
        print("[FATAL] No samples passed quality filter")
        return 1

    commands_arr = np.stack(commands_list).astype(np.float32)
    qpos_arr = np.stack(qpos_list).astype(np.float32)
    errf_arr = np.array(errf_list, dtype=np.float32)
    errw_arr = np.array(errw_list, dtype=np.float32)

    print(f"[OUT] commands: {commands_arr.shape}, joint_pos: {qpos_arr.shape}")
    print(f"[OUT] err_foot: mean={errf_arr.mean()*1000:.2f}mm  max={errf_arr.max()*1000:.2f}mm")
    print(f"[OUT] err_wrist: mean={errw_arr.mean()*1000:.2f}mm  max={errw_arr.max()*1000:.2f}mm")
    print(f"[STATS] joint_pos mean (deg): {np.degrees(qpos_arr.mean(0)).round(1)}")
    print(f"        joint_pos std  (deg): {np.degrees(qpos_arr.std(0)).round(1)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        commands=commands_arr,
        joint_pos=qpos_arr,
        ik_err_foot=errf_arr,
        ik_err_wrist=errw_arr,
        actuated_joint_names=np.array(ACTUATED_28),
        command_layout=np.array([
            "h",
            "lx", "ly", "lz", "lqx", "lqy", "lqz", "lqw",
            "rx", "ry", "rz", "rqx", "rqy", "rqz", "rqw",
            "alpha_t",
        ]),
    )
    print(f"[SAVE] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
