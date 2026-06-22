# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Dump everything needed to deploy a trained policy in MuJoCo.

Mirrors `play.py`'s env construction so the metadata reflects the *actual*
runtime state (joint order from the articulation, default joint positions,
PD gains, observation layout, etc.). Writes:

  logs/rsl_rl/<exp>/<run>/exported/mujoco_config.yaml

That YAML is the single source of truth for the MuJoCo deploy runner.

Usage:
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/dump_mujoco_config.py \\
      --task Isaac-Tracking-LocoManip-HV1-Play-v0 --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Dump MuJoCo deploy config from a trained RSL-RL policy.")
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
    help="Path to write the YAML. Default: <run_dir>/exported/mujoco_config.yaml",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# headless dump — no need for a viewer
args_cli.headless = True

# Hydra wants its own argv
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the imports."""

import os
from collections import OrderedDict

import gymnasium as gym
import torch
import yaml

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import importlib.metadata as metadata

installed_version = metadata.version("rsl-rl-lib")


def _to_pylist(t):
    """Convert torch / numpy to a plain Python list of floats."""
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    if hasattr(t, "tolist"):
        return t.tolist()
    return list(t)


def _resolve_xml_path(robot_cfg):
    """Pull the source asset path out of the articulation cfg.

    Note: this is whatever Isaac used to spawn the robot (USD / URDF). The
    MuJoCo deploy runner needs an MJCF instead, so the user typically
    overrides this value in the YAML to point at their `scene.xml`.
    """
    try:
        return robot_cfg.spawn.asset_path
    except AttributeError:
        return None


def _collect_obs_layout(env_cfg):
    """Walk the policy observation group and record term name + nominal dim hint.

    Actual per-term dims come from env.unwrapped.observation_manager once the
    env is built; this is just for human-readable structure.
    """
    layout = []
    try:
        policy_grp = env_cfg.observations.policy
        for name in policy_grp.__dataclass_fields__:
            term = getattr(policy_grp, name)
            if hasattr(term, "func"):
                layout.append(name)
    except Exception:
        pass
    return layout


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Build env, extract MuJoCo deploy metadata, write YAML."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed

    # locate the run dir (same logic as play.py)
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

    # ---- Pull everything out of the live env ------------------------------
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    sim_dt = unwrapped.physics_dt
    decimation = unwrapped.cfg.decimation
    policy_dt = sim_dt * decimation

    # joint order Isaac actually uses (this is the canonical order for obs)
    joint_names_isaac = list(robot.data.joint_names)
    default_joint_pos = _to_pylist(robot.data.default_joint_pos[0])
    joint_stiffness = _to_pylist(robot.data.joint_stiffness[0])
    joint_damping = _to_pylist(robot.data.joint_damping[0])
    joint_effort_limit = _to_pylist(robot.data.joint_effort_limits[0]) if hasattr(robot.data, "joint_effort_limits") else None
    # Motor physics params — needed to hand-edit the MJCF (joint armature /
    # frictionloss / range) and for actuator velocity / position safety clamps
    # in the deploy script. All hasattr-guarded so older IsaacLab versions that
    # don't expose these attributes still produce a valid (partial) YAML.
    joint_velocity_limit = (
        _to_pylist(robot.data.joint_velocity_limits[0]) if hasattr(robot.data, "joint_velocity_limits") else None
    )
    joint_armature = _to_pylist(robot.data.joint_armature[0]) if hasattr(robot.data, "joint_armature") else None
    joint_friction = _to_pylist(robot.data.joint_friction[0]) if hasattr(robot.data, "joint_friction") else None
    if hasattr(robot.data, "joint_pos_limits"):
        # shape: (Nenv, Nj, 2) where [:, :, 0] is lower, [:, :, 1] is upper
        jpl = robot.data.joint_pos_limits[0]
        joint_pos_limit_lower = _to_pylist(jpl[:, 0])
        joint_pos_limit_upper = _to_pylist(jpl[:, 1])
    else:
        joint_pos_limit_lower = None
        joint_pos_limit_upper = None
    # Viscous friction lives on each ActuatorBase instance (not on robot.data),
    # as per-actuator-group tensors. Walk robot.actuators and project the values
    # back into the full Isaac joint order. Recorded in the YAML for sim2real
    # reference and applied as MuJoCo passive damping via <joint damping="..."/>
    # in the hand-edited MJCF. NOT subtracted from Python PD kd — the small
    # double-count is consistent with how the real motor behaves.
    joint_viscous_friction = [0.0] * len(joint_names_isaac)
    # Walk robot.actuators to override per-joint motor params with the NOMINAL
    # configured values from each actuator's .cfg (not the runtime tensors).
    # Why .cfg and not the runtime tensor:
    #   * robot.data.joint_stiffness/damping are ZERO for explicit actuators
    #     (DelayedPDActuatorCfg / IdealPDActuatorCfg) — they run the PD in
    #     Python and leave the underlying USD drive in effort-mode with no gains.
    #   * act.stiffness / act.damping / act.armature / act.friction at runtime
    #     are the per-env tensors AFTER startup-mode DR events have been applied
    #     (actuator_gains_randomize ±20%, joint_params_randomize ±20%). Reading
    #     act.stiffness[0] therefore reports env 0's RANDOMIZED scaling, not the
    #     nominal motor spec.
    #   * Same story for effort_limit: robot.data.joint_effort_limits is the
    #     USD-level limit (set to ~1e9 / infinity for explicit actuators because
    #     they clamp in Python), while the real per-group limit lives on
    #     act.cfg.effort_limit.
    # Deploy needs nominal: the real robot has one fixed set of motors, not a
    # randomized one, so the YAML must record the configured spec.
    #
    # _resolve_cfg_val handles the two cfg formats actuators accept:
    #   * float (single value applied to all joints in the group) — the common case
    #   * dict[str, float] (regex → value mapping) — used in some configs
    import re as _re
    def _resolve_cfg_val(cfg_val, joint_names_in_group, fallback=None):
        if cfg_val is None:
            return [fallback] * len(joint_names_in_group)
        if isinstance(cfg_val, (int, float)):
            return [float(cfg_val)] * len(joint_names_in_group)
        if isinstance(cfg_val, dict):
            out = []
            for jname in joint_names_in_group:
                matched = fallback
                for pattern, val in cfg_val.items():
                    if _re.fullmatch(pattern, jname) is not None:
                        matched = float(val)
                        break
                out.append(matched)
            return out
        # Unknown format — fall back to per-joint repetition of stringified value.
        return [fallback] * len(joint_names_in_group)

    if hasattr(robot, "actuators") and robot.actuators:
        for act_name, act in robot.actuators.items():
            try:
                jids = act.joint_indices if hasattr(act, "joint_indices") else act.joint_ids
                if isinstance(jids, slice):
                    ids_list = list(range(len(joint_names_isaac)))[jids]
                elif hasattr(jids, "tolist"):
                    ids_list = jids.tolist()
                else:
                    ids_list = list(jids)
                joint_names_in_group = [joint_names_isaac[i] for i in ids_list]
                cfg = act.cfg if hasattr(act, "cfg") else None

                def _set(arr, attr_name, fallback_runtime_tensor=None):
                    """Write resolved cfg value for this group into arr at ids_list positions."""
                    cfg_val = getattr(cfg, attr_name, None) if cfg is not None else None
                    if cfg_val is not None:
                        vals = _resolve_cfg_val(cfg_val, joint_names_in_group, fallback=None)
                        for local_idx, isaac_idx in enumerate(ids_list):
                            if vals[local_idx] is not None:
                                arr[isaac_idx] = float(vals[local_idx])
                        return
                    # Fall back to runtime tensor only if cfg is unavailable.
                    if fallback_runtime_tensor is not None:
                        vals = fallback_runtime_tensor[0].detach().cpu().numpy()
                        for local_idx, isaac_idx in enumerate(ids_list):
                            arr[isaac_idx] = float(vals[local_idx])

                # Kp / Kd — the policy PD gains. Real deploy uses these directly.
                _set(joint_stiffness, "stiffness",
                     fallback_runtime_tensor=getattr(act, "stiffness", None))
                _set(joint_damping, "damping",
                     fallback_runtime_tensor=getattr(act, "damping", None))
                # Motor physics — written into MJCF; deploy uses for safety clamps.
                if joint_armature is not None:
                    _set(joint_armature, "armature",
                         fallback_runtime_tensor=getattr(act, "armature", None))
                if joint_friction is not None:
                    _set(joint_friction, "friction",
                         fallback_runtime_tensor=getattr(act, "friction", None))
                _set(joint_viscous_friction, "viscous_friction",
                     fallback_runtime_tensor=getattr(act, "viscous_friction", None))
                # Effort limit — robot.data reports the USD limit (huge / inf for
                # explicit actuators). The per-group spec is on act.cfg.
                if joint_effort_limit is not None:
                    _set(joint_effort_limit, "effort_limit")
                # Velocity limit — usually matches between cfg and robot.data, but
                # override to be consistent (and correct if a cfg dict was used).
                if joint_velocity_limit is not None:
                    _set(joint_velocity_limit, "velocity_limit")
            except Exception as e:
                print(f"  WARNING: actuator '{act_name}' extraction failed: {e}")

    # action — what subset of joints the policy outputs
    action_term = unwrapped.action_manager.get_term("joint_pos")
    action_joint_names = list(action_term._joint_names) if hasattr(action_term, "_joint_names") else None
    if action_joint_names is None:
        # fallback: introspect via the resolved joint_ids
        action_joint_ids = getattr(action_term, "_joint_ids", None)
        if action_joint_ids is not None:
            action_joint_names = [joint_names_isaac[i] for i in action_joint_ids]
    action_scale = getattr(action_term.cfg, "scale", None)
    use_default_offset = getattr(action_term.cfg, "use_default_offset", None)
    action_dim = unwrapped.action_manager.total_action_dim

    # observation — per-term dims
    # Note: when concatenate_terms=True, `group_obs_dim` returns a single combined
    # tuple. Use `group_obs_term_dim` for the always-per-term breakdown.
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
    # Total: prefer manager's reported dim (handles concat correctly).
    full_dim = obs_manager.group_obs_dim[obs_group]
    if isinstance(full_dim, (tuple, list)) and len(full_dim) > 0:
        total_obs_dim = int(full_dim[0]) if isinstance(full_dim[0], int) else int(full_dim[0][0])
    else:
        total_obs_dim = int(full_dim)

    # command terms (so the deploy runner can publish them)
    command_terms = []
    if hasattr(unwrapped, "command_manager") and unwrapped.command_manager is not None:
        for cname in unwrapped.command_manager.active_terms:
            cterm = unwrapped.command_manager.get_term(cname)
            cmd_dim = int(cterm.command.shape[-1])
            command_terms.append({"name": cname, "dim": cmd_dim})

    # robot asset path — Isaac's spawn source. User overrides to MJCF/XML
    # in the deploy YAML before running MuJoCo.
    xml_path = _resolve_xml_path(robot.cfg)

    # exported policy path (play.py writes here; we mirror the location)
    exported_dir = os.path.join(run_dir, "exported")
    policy_path = os.path.join(exported_dir, "policy.pt")
    policy_onnx_path = os.path.join(exported_dir, "policy.onnx")

    # Verify the exported policy exists and is fresh relative to the checkpoint.
    policy_status = {"exists": False, "stale": None, "size_mb": None, "mtime": None}
    if os.path.isfile(policy_path):
        policy_status["exists"] = True
        policy_status["size_mb"] = round(os.path.getsize(policy_path) / 1024 / 1024, 2)
        policy_status["mtime"] = os.path.getmtime(policy_path)
        if resume_path and os.path.isfile(resume_path):
            policy_status["stale"] = os.path.getmtime(resume_path) > policy_status["mtime"]

    # ---- Assemble the YAML ------------------------------------------------
    cfg_out = {
        "meta": {
            "task": train_task_name,
            "play_task": task_name,
            "experiment_name": agent_cfg.experiment_name,
            "run_dir": run_dir,
            "checkpoint": resume_path,
        },
        "robot": {
            "xml_path": xml_path,
            "num_dof_total": len(joint_names_isaac),
            # CRITICAL: this is the order Isaac uses for joint_pos/joint_vel observations
            "joint_names_isaac_order": joint_names_isaac,
            "default_joint_pos": default_joint_pos,
            # Controller params (live in the Python deploy PD loop)
            "kp": joint_stiffness,
            "kd": joint_damping,
            "effort_limit": joint_effort_limit,
            "velocity_limit": joint_velocity_limit,
            # MJCF joint physics params (hand-edit into the model XML to match Isaac)
            #   armature           -> <joint armature="..."/>
            #   friction           -> <joint frictionloss="..."/>  (Coulomb)
            #   viscous_friction   -> <joint damping="..."/>       (velocity-proportional)
            #   pos_limit_*        -> <joint range="lower upper"/>
            "armature": joint_armature,
            "friction": joint_friction,
            "viscous_friction": joint_viscous_friction,
            "pos_limit_lower": joint_pos_limit_lower,
            "pos_limit_upper": joint_pos_limit_upper,
        },
        "action": {
            "dim": int(action_dim),
            # CRITICAL: this is the subset of joints (and their order) that the policy outputs
            "joint_names_action_order": action_joint_names,
            "scale": float(action_scale) if action_scale is not None else None,
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
            "_comment": "Set EE pose commands at runtime (per-hand) in the MuJoCo runner.",
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

    # ---- Write -------------------------------------------------------------
    out_yaml = args_cli.output_yaml or os.path.join(exported_dir, "mujoco_config.yaml")
    os.makedirs(os.path.dirname(out_yaml), exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.dump(cfg_out, f, default_flow_style=False, sort_keys=False)

    # ---- Console summary ---------------------------------------------------
    print("\n" + "=" * 78)
    print(f"MuJoCo deploy config written to:\n  {out_yaml}")
    print("=" * 78)
    print(f"  task                : {task_name}")
    print(f"  checkpoint          : {resume_path}")
    print(f"  policy (will export): {policy_path}")
    print(f"  xml_path            : {xml_path}")
    print(f"  sim_dt              : {sim_dt}  (policy_dt = {policy_dt})")
    print(f"  decimation          : {decimation}")
    print(f"  num_dof_total       : {len(joint_names_isaac)}")
    print(f"  action_dim          : {action_dim}  (joints below)")
    print(f"  action_scale        : {action_scale}    use_default_offset: {use_default_offset}")
    print(f"  observation_dim     : {total_obs_dim}")
    print("-" * 78)
    print("  observation terms (in Isaac order):")
    for name, dim in obs_terms.items():
        print(f"    {name:<28s} {dim:>4d}")
    print("-" * 78)
    print("  command terms:")
    for c in command_terms:
        print(f"    {c['name']:<28s} {c['dim']:>4d}")
    print("-" * 78)
    print(f"  Isaac joint order (first 10):")
    for i, n in enumerate(joint_names_isaac[:10]):
        # Compose optional fields without crashing if a value is None
        arm_s = f"{joint_armature[i]:>6.4f}" if joint_armature is not None else "  n/a"
        fric_s = f"{joint_friction[i]:>5.3f}" if joint_friction is not None else " n/a"
        vlim_s = f"{joint_velocity_limit[i]:>5.2f}" if joint_velocity_limit is not None else " n/a"
        print(
            f"    [{i:2d}] {n:<32s} q_default={default_joint_pos[i]:+.4f}  "
            f"kp={joint_stiffness[i]:>6.1f}  kd={joint_damping[i]:>5.2f}  "
            f"arm={arm_s}  fric={fric_s}  vlim={vlim_s}"
        )
    if len(joint_names_isaac) > 10:
        print(f"    ... ({len(joint_names_isaac) - 10} more — see YAML)")
    if action_joint_names:
        print("-" * 78)
        print(f"  Action joint order (policy outputs in THIS order):")
        for i, n in enumerate(action_joint_names):
            print(f"    [{i:2d}] {n}")
    print("=" * 78)
    # Policy file status
    if policy_status["exists"]:
        from datetime import datetime
        mtime_str = datetime.fromtimestamp(policy_status["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  policy.pt           : EXISTS  ({policy_status['size_mb']} MB, mtime {mtime_str})")
        if policy_status["stale"]:
            print("  ⚠  WARNING: policy.pt is OLDER than the selected checkpoint.")
            print("     The exported policy may not match this checkpoint.")
            print("     Re-run play.py with the same --checkpoint to refresh:")
            ckpt_arg = f"--checkpoint {resume_path}" if resume_path else ""
            print(f"       ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\")
            print(f"           --task {task_name} --num_envs 1 {ckpt_arg}")
        else:
            print("  → policy.pt is current. Ready for MuJoCo deployment.")
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
