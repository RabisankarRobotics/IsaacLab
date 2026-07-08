# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Dump the MuJoCo/real-robot deploy config for the G1 29-DOF legs-only clean walk.

Single source of truth for `run_policy_mujoco_cfg.py` (and the real-robot node):
everything the deploy runner needs is read from the *live* trained env so it can
never drift from what was trained — joint order, per-joint PD gains, default pose,
action scale, observation layout, control dt, and the frozen gait-clock params.

Everything is keyed by joint NAME. The deploy runner maps names -> MuJoCo/DDS
indices itself, so there are NO hand-typed index arrays anywhere.

Schema (superset of dump_mujoco_config.py's v3 schema):
  * robot.leg_joint_names / robot.upper_body_joint_names  -> the two obs groups,
    in the exact order the ObservationManager emits them (resolved from the live
    obs terms, not guessed).
  * gait.{period, freeze_cmd_threshold, phase_obs_term}   -> lets the runner
    reproduce custom_mdp._gait_phase_scalar (the frozen-at-idle clock).

Usage:
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/dump_mujoco_config_g1_clean.py \\
      --task Isaac-Velocity-Flat-Legs-G1-29Dof-Clean-Play-v0 --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Dump G1 legs-only clean MuJoCo deploy config.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (1 is fine for a dump).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--output_yaml",
    type=str,
    default=None,
    help="Path to write the YAML. Default: <run_dir>/exported/mujoco_config_g1_clean.yaml",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.headless = True  # headless dump — no viewer needed
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the imports."""

import os
import re as _re
from collections import OrderedDict

import gymnasium as gym
import yaml

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import importlib.metadata as metadata

installed_version = metadata.version("rsl-rl-lib")

# Mirror of the default cmd_threshold in custom_mdp._gait_phase_scalar. The gait
# clock advances only while ||base_velocity|| exceeds this (frozen at idle).
GAIT_FREEZE_CMD_THRESHOLD = 0.1


def _to_pylist(t):
    """torch/numpy -> plain Python list of floats."""
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    if hasattr(t, "tolist"):
        return t.tolist()
    return list(t)


def _resolve_xml_path(robot_cfg):
    """Isaac's spawn asset (USD). The MuJoCo runner uses its own MJCF via --xml,
    so this is informational only."""
    try:
        return robot_cfg.spawn.asset_path
    except AttributeError:
        return None


def _resolve_obs_term_joint_names(env_cfg, term_name, joint_names_isaac):
    """Return the ordered joint names an obs term emits. Matches the term's OWN
    SceneEntityCfg joint_names regex list against the Isaac-order joint list, which
    reproduces IsaacLab's default resolution (ascending index, preserve_order=False)
    — the exact order the ObservationManager concatenates. No scene dependency."""
    patterns = getattr(env_cfg.observations.policy, term_name).params["asset_cfg"].joint_names
    if isinstance(patterns, str):
        patterns = [patterns]
    return [n for n in joint_names_isaac if any(_re.fullmatch(p, n) for p in patterns)]


def _resolve_cfg_val(cfg_val, joint_names_in_group, fallback=None):
    """Resolve an actuator cfg field (float or regex->value dict) to a per-joint list."""
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
    return [fallback] * len(joint_names_in_group)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
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

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    sim_dt = unwrapped.physics_dt
    decimation = unwrapped.cfg.decimation
    policy_dt = sim_dt * decimation

    # --- joint order Isaac actually uses (canonical order for obs/gains) --------
    joint_names_isaac = list(robot.data.joint_names)
    default_joint_pos = _to_pylist(robot.data.default_joint_pos[0])
    joint_stiffness = _to_pylist(robot.data.joint_stiffness[0])
    joint_damping = _to_pylist(robot.data.joint_damping[0])
    joint_effort_limit = (
        _to_pylist(robot.data.joint_effort_limits[0]) if hasattr(robot.data, "joint_effort_limits") else None
    )

    # Override kp/kd/effort with the NOMINAL configured actuator values (robust to
    # explicit actuators reporting 0 in robot.data and to startup DR scaling). The
    # G1 clean uses ImplicitActuatorCfg so robot.data is already correct; this just
    # reaffirms it from the cfg, which is the authoritative motor spec for deploy.
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
                jn_group = [joint_names_isaac[i] for i in ids_list]
                cfg = act.cfg if hasattr(act, "cfg") else None

                def _set(arr, attr_name):
                    cfg_val = getattr(cfg, attr_name, None) if cfg is not None else None
                    if cfg_val is None:
                        return
                    vals = _resolve_cfg_val(cfg_val, jn_group, fallback=None)
                    for local_idx, isaac_idx in enumerate(ids_list):
                        if vals[local_idx] is not None:
                            arr[isaac_idx] = float(vals[local_idx])

                _set(joint_stiffness, "stiffness")
                _set(joint_damping, "damping")
                if joint_effort_limit is not None:
                    _set(joint_effort_limit, "effort_limit")
            except Exception as e:
                print(f"  WARNING: actuator '{act_name}' gain extraction failed: {e}")

    # --- action term: the legs the policy outputs ------------------------------
    action_term = unwrapped.action_manager.get_term("joint_pos")
    action_joint_names = list(action_term._joint_names) if hasattr(action_term, "_joint_names") else None
    if action_joint_names is None:
        action_joint_ids = getattr(action_term, "_joint_ids", None)
        if action_joint_ids is not None:
            action_joint_names = [joint_names_isaac[i] for i in action_joint_ids]
    action_scale = getattr(action_term.cfg, "scale", None)
    use_default_offset = getattr(action_term.cfg, "use_default_offset", None)
    action_dim = unwrapped.action_manager.total_action_dim

    # --- the two obs joint groups, resolved from their OWN obs terms ------------
    leg_joint_names = _resolve_obs_term_joint_names(env_cfg, "joint_pos", joint_names_isaac)
    upper_body_joint_names = _resolve_obs_term_joint_names(env_cfg, "upper_body_joint_pos", joint_names_isaac)
    # legs obs order MUST equal the action order (both resolve LEG_JOINT_NAMES the
    # same way) — assert it so a config mismatch fails loud, not silently.
    if action_joint_names is not None and list(action_joint_names) != list(leg_joint_names):
        raise RuntimeError(
            "Action joint order != leg obs joint order:\n"
            f"  action: {action_joint_names}\n  obs   : {leg_joint_names}\n"
            "The deploy contract assumes they match."
        )

    # --- observation: per-term dims in declared order --------------------------
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

    # --- gait clock (frozen-at-idle) params ------------------------------------
    gait = None
    if hasattr(env_cfg.observations.policy, "gait_phase"):
        gait_term = env_cfg.observations.policy.gait_phase
        gait = {
            "phase_obs_term": "gait_phase",
            "period": float(gait_term.params["period"]),
            "freeze_cmd_threshold": float(GAIT_FREEZE_CMD_THRESHOLD),
            "_comment": (
                "phase = (count * policy_dt) % period / period; count advances only while "
                "||base_velocity|| > freeze_cmd_threshold. Emit obs as [sin, cos] of 2*pi*phase. "
                "Mirrors custom_mdp._gait_phase_scalar."
            ),
        }

    # --- command terms ---------------------------------------------------------
    command_terms = []
    if hasattr(unwrapped, "command_manager") and unwrapped.command_manager is not None:
        for cname in unwrapped.command_manager.active_terms:
            cterm = unwrapped.command_manager.get_term(cname)
            command_terms.append({"name": cname, "dim": int(cterm.command.shape[-1])})

    xml_path = _resolve_xml_path(robot.cfg)
    exported_dir = os.path.join(run_dir, "exported")
    policy_pt = os.path.join(exported_dir, "policy.pt")
    policy_onnx = os.path.join(exported_dir, "policy.onnx")

    policy_status = {"exists": os.path.isfile(policy_pt), "size_mb": None, "mtime": None, "stale": None}
    if policy_status["exists"]:
        policy_status["size_mb"] = round(os.path.getsize(policy_pt) / 1024 / 1024, 2)
        policy_status["mtime"] = os.path.getmtime(policy_pt)
        if resume_path and os.path.isfile(resume_path):
            policy_status["stale"] = os.path.getmtime(resume_path) > policy_status["mtime"]

    # --- assemble YAML ---------------------------------------------------------
    cfg_out = {
        "meta": {
            "task": train_task_name,
            "play_task": task_name,
            "experiment_name": agent_cfg.experiment_name,
            "run_dir": run_dir,
            "checkpoint": resume_path,
            "schema": "g1_clean",
        },
        "robot": {
            "xml_path_isaac": xml_path,  # USD spawn source (informational)
            "num_dof_total": len(joint_names_isaac),
            "joint_names_isaac_order": joint_names_isaac,  # gains/default are in THIS order
            "default_joint_pos": default_joint_pos,
            "kp": joint_stiffness,
            "kd": joint_damping,
            "effort_limit": joint_effort_limit,
            # the two obs groups, each in ObservationManager emit order:
            "leg_joint_names": leg_joint_names,  # policy-actioned (12)
            "upper_body_joint_names": upper_body_joint_names,  # held at default (17)
        },
        "action": {
            "dim": int(action_dim),
            "joint_names_action_order": action_joint_names,
            "scale": float(action_scale) if isinstance(action_scale, (int, float)) else action_scale,
            "use_default_offset": bool(use_default_offset) if use_default_offset is not None else None,
            "_comment": "target_q[leg] = default_q[leg] + scale * raw_action (use_default_offset=True).",
        },
        "observation": {
            "total_dim": int(total_obs_dim),
            "terms": [{"name": k, "dim": v} for k, v in obs_terms.items()],
            "obs_scales_comment": "Isaac Lab manager-based env returns RAW values (no obs scaling).",
        },
        "gait": gait,
        "control": {
            "sim_dt": float(sim_dt),
            "decimation": int(decimation),
            "policy_dt": float(policy_dt),
        },
        "commands": {
            "active_terms": command_terms,
            "init_velocity": [0.0, 0.0, 0.0],
        },
        "policy": {
            "jit_path": policy_pt,
            "onnx_path": policy_onnx if os.path.isfile(policy_onnx) else None,
            "exists": policy_status["exists"],
            "size_mb": policy_status["size_mb"],
            "actor_hidden_dims": getattr(agent_cfg.policy, "actor_hidden_dims", None),
            "activation": getattr(agent_cfg.policy, "activation", None),
        },
    }

    out_yaml = args_cli.output_yaml or os.path.join(exported_dir, "mujoco_config_g1_clean.yaml")
    os.makedirs(os.path.dirname(out_yaml), exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.dump(cfg_out, f, default_flow_style=False, sort_keys=False)

    # --- console summary -------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"G1 clean MuJoCo deploy config written to:\n  {out_yaml}")
    print("=" * 78)
    print(f"  task            : {task_name}")
    print(f"  checkpoint      : {resume_path}")
    print(f"  sim_dt          : {sim_dt}   policy_dt: {policy_dt}   decimation: {decimation}")
    print(f"  num_dof_total   : {len(joint_names_isaac)}")
    print(f"  action_dim      : {action_dim}   scale: {action_scale}   use_default_offset: {use_default_offset}")
    print(f"  observation_dim : {total_obs_dim}")
    if gait is not None:
        print(f"  gait            : period {gait['period']}s   freeze<{gait['freeze_cmd_threshold']}   "
              f"term '{gait['phase_obs_term']}'")
    print("-" * 78)
    print("  observation terms (declared order):")
    for name, dim in obs_terms.items():
        print(f"    {name:<28s} {dim:>4d}")
    print("-" * 78)
    print("  leg joints (policy action / obs order):")
    for i, n in enumerate(leg_joint_names):
        j = joint_names_isaac.index(n)
        print(f"    [{i:2d}] {n:<26s} q0={default_joint_pos[j]:+.3f}  kp={joint_stiffness[j]:>6.1f}  kd={joint_damping[j]:>5.2f}")
    print("-" * 78)
    print("  upper-body joints (held at default, obs order):")
    for i, n in enumerate(upper_body_joint_names):
        j = joint_names_isaac.index(n)
        print(f"    [{i:2d}] {n:<26s} q0={default_joint_pos[j]:+.3f}  kp={joint_stiffness[j]:>6.1f}  kd={joint_damping[j]:>5.2f}")
    print("=" * 78)
    if policy_status["exists"]:
        from datetime import datetime
        mt = datetime.fromtimestamp(policy_status["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  policy.pt       : EXISTS ({policy_status['size_mb']} MB, {mt})")
        if policy_status["stale"]:
            print("  ⚠  policy.pt is OLDER than the checkpoint — re-run play.py to refresh the export.")
        else:
            print("  → policy.pt current. Export the ONNX (play.py) and run run_policy_mujoco_cfg.py.")
    else:
        print("  policy.pt       : MISSING — run play.py to export policy.pt / policy.onnx first.")
    print("=" * 78 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
