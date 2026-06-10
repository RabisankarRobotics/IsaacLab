# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Dump MuJoCo deploy config for an HV1 V4 (KMP-residual) policy.

V4 differs from V3 in three deploy-relevant ways:
  1. Action scale is a per-joint DICT (legs 0.25, arms 0.10, waist 0.10)
     instead of a scalar. The V3 dumper does `float(scale)` and crashes.
  2. Action class needs a frozen KMP MLP loaded at deploy time. We emit
     the checkpoint path + the 28-joint KMP output order so the deploy
     script doesn't need to import anything from scripts/hv1/kmp/.
  3. q_target = KMP(commands) + residual * per_joint_scale, NOT
     q_target = q_default + scale * action. The MuJoCo runner must
     handle this; use_default_offset is meaningless for V4.

The YAML schema is a superset of V3:
  - action.scale (dict) replaces V3's scalar action.scale
  - action.scale_per_joint (list, in joint_names_action_order) for fast lookup
  - action.kmp_checkpoint (path)
  - action.kmp_output_order (list of 28 joint names)
  - action.residual_scale (float or null)
Everything else identical to V3 — observation, robot, control, policy blocks.

Usage:
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/dump_mujoco_config_v4.py \\
      --task Isaac-Tracking-LocoManipV4-HV1-Play-v0 --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Dump V4 MuJoCo deploy config from a trained RSL-RL policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (1 is fine for dump).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--output_yaml",
    type=str,
    default=None,
    help="Path to write the YAML. Default: <run_dir>/exported/mujoco_config_v4.yaml",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the imports."""

import os
from collections import OrderedDict

import gymnasium as gym
import torch  # noqa: F401  (kept for parity with V3 dump)
import yaml

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import importlib.metadata as metadata

installed_version = metadata.version("rsl-rl-lib")

# Add the KMP scripts dir to sys.path so we can import ACTUATED_28 (the canonical
# 28-joint KMP output order) without copy-pasting it.
_KMP_SCRIPTS_DIR = "/home/rabisankar/IsaacLab/scripts/hv1/kmp"
if _KMP_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _KMP_SCRIPTS_DIR)
from generate_kmp_dataset import ACTUATED_28 as KMP_OUTPUT_ORDER  # noqa: E402


def _to_pylist(t):
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    if hasattr(t, "tolist"):
        return t.tolist()
    return list(t)


def _resolve_xml_path(robot_cfg):
    try:
        return robot_cfg.spawn.asset_path
    except AttributeError:
        return None


def _resolve_scale_per_joint(scale_field, action_joint_names):
    """Convert V4's dict-or-float scale into a per-joint list aligned with
    `action_joint_names`. Raises if a joint is missing from the dict."""
    if isinstance(scale_field, dict):
        missing = [n for n in action_joint_names if n not in scale_field]
        if missing:
            raise RuntimeError(
                f"action.scale dict missing entries for: {missing}. "
                f"V4 expects an entry for every action joint."
            )
        return [float(scale_field[n]) for n in action_joint_names]
    if isinstance(scale_field, (int, float)):
        return [float(scale_field)] * len(action_joint_names)
    raise RuntimeError(
        f"Unsupported action.scale type {type(scale_field)} for V4 dump. "
        f"Expected dict (per-joint) or float (uniform)."
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        try:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        except Exception:
            resume_path = None

    run_dir = os.path.dirname(resume_path) if resume_path else log_root_path

    env_cfg.log_dir = run_dir
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    sim_dt = unwrapped.physics_dt
    decimation = unwrapped.cfg.decimation
    policy_dt = sim_dt * decimation

    joint_names_isaac = list(robot.data.joint_names)
    default_joint_pos = _to_pylist(robot.data.default_joint_pos[0])
    joint_stiffness = _to_pylist(robot.data.joint_stiffness[0])
    joint_damping = _to_pylist(robot.data.joint_damping[0])
    joint_effort_limit = (
        _to_pylist(robot.data.joint_effort_limits[0]) if hasattr(robot.data, "joint_effort_limits") else None
    )

    # ---- V4 action term: KMP-residual ------------------------------------
    action_term = unwrapped.action_manager.get_term("joint_pos")
    action_joint_names = list(action_term._joint_names) if hasattr(action_term, "_joint_names") else None
    if action_joint_names is None:
        action_joint_ids = getattr(action_term, "_joint_ids", None)
        if action_joint_ids is not None:
            action_joint_names = [joint_names_isaac[i] for i in action_joint_ids]

    scale_field = getattr(action_term.cfg, "scale", None)
    scale_per_joint = _resolve_scale_per_joint(scale_field, action_joint_names)
    kmp_checkpoint = getattr(action_term.cfg, "kmp_checkpoint", None)
    residual_scale = getattr(action_term.cfg, "residual_scale", None)
    use_default_offset = getattr(action_term.cfg, "use_default_offset", None)
    action_dim = unwrapped.action_manager.total_action_dim

    if not action_joint_names or len(action_joint_names) != 28:
        raise RuntimeError(
            f"V4 expects 28 action joints; got {len(action_joint_names) if action_joint_names else 0}."
        )
    if kmp_checkpoint is None:
        raise RuntimeError(
            "Action term has no kmp_checkpoint field. Are you sure this is a V4 task? "
            f"Got cfg type {type(action_term.cfg).__name__}."
        )

    # Sanity: KMP_OUTPUT_ORDER must be a subset of action joint names.
    missing_kmp = [n for n in KMP_OUTPUT_ORDER if n not in action_joint_names]
    if missing_kmp:
        raise RuntimeError(
            f"KMP output joints not in action joint set: {missing_kmp}. "
            "Action joint_names regex doesn't cover all KMP outputs — fix the env cfg."
        )

    # ---- Observations (unchanged from V3 schema) -------------------------
    obs_manager = unwrapped.observation_manager
    obs_group = "policy"
    obs_terms = OrderedDict()
    active_term_names = obs_manager.active_terms[obs_group]
    if hasattr(obs_manager, "group_obs_term_dim"):
        per_term_dims = obs_manager.group_obs_term_dim[obs_group]
    else:
        per_term_dims = obs_manager.group_obs_dim[obs_group]
    for term_name, term_dim in zip(active_term_names, per_term_dims):
        obs_terms[term_name] = int(term_dim[0]) if isinstance(term_dim, (tuple, list)) else int(term_dim)
    full_dim = obs_manager.group_obs_dim[obs_group]
    if isinstance(full_dim, (tuple, list)) and len(full_dim) > 0:
        total_obs_dim = int(full_dim[0]) if isinstance(full_dim[0], int) else int(full_dim[0][0])
    else:
        total_obs_dim = int(full_dim)

    command_terms = []
    if hasattr(unwrapped, "command_manager") and unwrapped.command_manager is not None:
        for cname in unwrapped.command_manager.active_terms:
            cterm = unwrapped.command_manager.get_term(cname)
            cmd_dim = int(cterm.command.shape[-1])
            command_terms.append({"name": cname, "dim": cmd_dim})

    xml_path = _resolve_xml_path(robot.cfg)

    exported_dir = os.path.join(run_dir, "exported")
    policy_path = os.path.join(exported_dir, "policy.pt")
    policy_onnx_path = os.path.join(exported_dir, "policy.onnx")

    policy_status = {"exists": False, "stale": None, "size_mb": None, "mtime": None}
    if os.path.isfile(policy_path):
        policy_status["exists"] = True
        policy_status["size_mb"] = round(os.path.getsize(policy_path) / 1024 / 1024, 2)
        policy_status["mtime"] = os.path.getmtime(policy_path)
        if resume_path and os.path.isfile(resume_path):
            policy_status["stale"] = os.path.getmtime(resume_path) > policy_status["mtime"]

    # ---- Assemble YAML ----------------------------------------------------
    cfg_out = {
        "meta": {
            "task": train_task_name,
            "play_task": task_name,
            "experiment_name": agent_cfg.experiment_name,
            "run_dir": run_dir,
            "checkpoint": resume_path,
            "schema": "v4",
            "schema_notes": (
                "V4 KMP-residual: target = KMP(cmd_16d) + residual * scale_per_joint. "
                "Deploy script must load action.kmp_checkpoint and apply the "
                "kmp_output_order → action_joint_names permutation."
            ),
        },
        "robot": {
            "xml_path": xml_path,
            "num_dof_total": len(joint_names_isaac),
            "joint_names_isaac_order": joint_names_isaac,
            "default_joint_pos": default_joint_pos,
            "kp": joint_stiffness,
            "kd": joint_damping,
            "effort_limit": joint_effort_limit,
        },
        "action": {
            "dim": int(action_dim),
            "joint_names_action_order": action_joint_names,
            # V4 changes vs V3 dump:
            "scale": (scale_field if isinstance(scale_field, dict)
                      else float(scale_field) if scale_field is not None else None),
            "scale_per_joint": scale_per_joint,  # aligned with joint_names_action_order
            "kmp_checkpoint": kmp_checkpoint,
            "kmp_output_order": list(KMP_OUTPUT_ORDER),
            "residual_scale": float(residual_scale) if residual_scale is not None else None,
            "use_default_offset": bool(use_default_offset) if use_default_offset is not None else None,
        },
        "observation": {
            "total_dim": int(total_obs_dim),
            "terms": [{"name": k, "dim": v} for k, v in obs_terms.items()],
            "obs_scales": {
                "lin_vel": 1.0,
                "ang_vel": 1.0,
                "joint_pos": 1.0,
                "joint_vel": 1.0,
                "_comment": "Isaac Lab manager-based env returns RAW values (no scaling).",
            },
        },
        "control": {
            "sim_dt": float(sim_dt),
            "decimation": int(decimation),
            "policy_dt": float(policy_dt),
        },
        "commands": {
            "active_terms": command_terms,
            "init_velocity": [0.0, 0.0, 0.0],
            "_comment": (
                "EE pose + body_height + waist_alpha commands fed into KMP each step. "
                "Quat layout in IsaacLab is [x,y,z,qw,qx,qy,qz] — KMP wants xyzw. "
                "Deploy script must swap qw <-> qz tail before forward."
            ),
        },
        "policy": {
            "jit_path": policy_path,
            "onnx_path": policy_onnx_path if os.path.isfile(policy_onnx_path) else None,
            "exists": policy_status["exists"],
            "size_mb": policy_status["size_mb"],
            "actor_obs_normalization": getattr(agent_cfg.policy, "actor_obs_normalization", None),
            "actor_hidden_dims": getattr(agent_cfg.policy, "actor_hidden_dims", None),
            "activation": getattr(agent_cfg.policy, "activation", None),
        },
    }

    out_yaml = args_cli.output_yaml or os.path.join(exported_dir, "mujoco_config_v4.yaml")
    os.makedirs(os.path.dirname(out_yaml), exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.dump(cfg_out, f, default_flow_style=False, sort_keys=False)

    # ---- Console summary --------------------------------------------------
    print("\n" + "=" * 78)
    print(f"V4 MuJoCo deploy config written to:\n  {out_yaml}")
    print("=" * 78)
    print(f"  task                : {task_name}")
    print(f"  checkpoint          : {resume_path}")
    print(f"  policy (will export): {policy_path}")
    print(f"  xml_path            : {xml_path}")
    print(f"  sim_dt              : {sim_dt}  (policy_dt = {policy_dt})")
    print(f"  decimation          : {decimation}")
    print(f"  num_dof_total       : {len(joint_names_isaac)}")
    print(f"  action_dim          : {action_dim}  (28 expected for V4)")
    print(f"  kmp_checkpoint      : {kmp_checkpoint}")
    print(f"  residual_scale      : {residual_scale}  (null = use scale_per_joint)")
    print(f"  scale_per_joint     : min={min(scale_per_joint)}  max={max(scale_per_joint)}")
    print(f"  observation_dim     : {total_obs_dim}")
    print("-" * 78)
    print("  observation terms (declared order):")
    for name, dim in obs_terms.items():
        print(f"    {name:<28s} {dim:>4d}")
    print("-" * 78)
    print("  command terms:")
    for c in command_terms:
        print(f"    {c['name']:<28s} {c['dim']:>4d}")
    print("-" * 78)
    print("  action joint order vs per-joint scale:")
    for i, n in enumerate(action_joint_names):
        print(f"    [{i:2d}] {n:<32s} scale={scale_per_joint[i]:.3f}")
    print("=" * 78)
    if policy_status["exists"]:
        from datetime import datetime
        mtime_str = datetime.fromtimestamp(policy_status["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  policy.pt           : EXISTS  ({policy_status['size_mb']} MB, mtime {mtime_str})")
        if policy_status["stale"]:
            print("  ⚠  WARNING: policy.pt is OLDER than the selected checkpoint.")
            print("     Re-run play.py with the same --checkpoint to refresh:")
            ckpt_arg = f"--checkpoint {resume_path}" if resume_path else ""
            print(f"       ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\")
            print(f"           --task {task_name} --num_envs 1 {ckpt_arg}")
        else:
            print("  → policy.pt is current. Ready for MuJoCo V4 deployment.")
    else:
        print("  policy.pt           : MISSING")
        print("     Run play.py first to export the policy:")
        ckpt_arg = f"--checkpoint {resume_path}" if resume_path else ""
        print(f"       ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\")
        print(f"           --task {task_name} --num_envs 1 {ckpt_arg}")
    print("=" * 78)
    print()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
