"""Smoke test V4 env: spawn 4 envs, step 5 times, confirm KMP action works.

Catches the dumb stuff before launching real training:
  * KMP checkpoint loads at the right device
  * Command names resolve in env.command_manager
  * q_prior + residual shape matches the 28-D action_dim
  * Reward terms compute without dimension errors

Usage:
  conda activate env_isaaclab
  python scripts/hv1/kmp/smoke_v4.py
"""

from __future__ import annotations

import argparse

# Isaac Sim must be launched before any isaaclab imports.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True   # always headless for smoke

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402  - registers Isaac-* envs
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    print("[SMOKE] Creating V4 env...")
    env_cfg = parse_env_cfg(
        "Isaac-Tracking-LocoManipV4-HV1-v0",
        device="cuda:0",
        num_envs=args.num_envs,
    )
    env = gym.make("Isaac-Tracking-LocoManipV4-HV1-v0", cfg=env_cfg).unwrapped

    print(f"[SMOKE] Env created. action_space={env.action_space.shape}  "
          f"obs_space(policy)={env.observation_space['policy'].shape}")

    # Confirm action term is the KMP one.
    term = env.action_manager.get_term("joint_pos")
    print(f"[SMOKE] Action term class: {type(term).__name__}")
    assert hasattr(term, "_kmp"), "joint_pos is not a KMPResidualJointPositionAction"
    print(f"[SMOKE] KMP loaded on device {next(term._kmp.parameters()).device}")
    print(f"[SMOKE] Residual scale shape: {getattr(term._scale, 'shape', 'scalar')}")
    # KMP-to-action remap should cover all 28 actuated joints. The remap is
    # built at __init__; if any KMP joint was missing it would have raised.
    assert term._joint_names is not None and len(term._joint_names) == 28
    print(f"[SMOKE] Action term resolved {len(term._joint_names)} joints; "
          f"KMP remap kmp_to_action.shape={tuple(term._kmp_to_action.shape)}")
    # Spot-check scale lands at the right joint NAME (not index) in the term.
    sc = term._scale[0] if hasattr(term._scale, 'shape') and term._scale.ndim > 1 else None
    if sc is not None:
        for name, want in [("left_knee_joint", 0.25), ("left_elbow_joint", 0.10),
                           ("waist_pitch_joint", 0.10)]:
            slot = term._joint_names.index(name)
            got = float(sc[slot])
            ok = "OK" if abs(got - want) < 1e-4 else "!! MISMATCH !!"
            print(f"  scale[{name:25s}] = {got:.3f}  (expected {want:.3f})  {ok}")

    print(f"[SMOKE] Resetting + stepping {args.steps} times...")
    obs_dict, _ = env.reset()
    for i in range(args.steps):
        actions = torch.zeros(args.num_envs, env.action_space.shape[1],
                              device=env.device)
        obs_dict, rew, term_flag, trunc, info = env.step(actions)
        # Inspect q_prior to confirm KMP is firing
        q_prior = env.action_manager.get_term("joint_pos").last_q_prior
        print(f"  step {i}: rew_mean={rew.mean().item():+.4f}  "
              f"q_prior_norm={q_prior.norm(dim=1).mean().item():.3f}  "
              f"terminated={term_flag.sum().item()}  truncated={trunc.sum().item()}")

    print("[SMOKE] PASS — V4 env runs cleanly.")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
