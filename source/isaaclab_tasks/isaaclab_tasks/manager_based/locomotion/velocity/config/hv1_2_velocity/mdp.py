"""Custom MDP terms for the HV1.2 velocity-tracking (walking) task.

Same as the standing task — re-exported here so this package is self-contained
and we don't cross-import between sibling task folders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def air_time_variance_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize variance in air/contact time across feet (asymmetric-gait penalty).

    Adapted from Boston Dynamics Spot's MDP. If both feet spend the same time
    in the air and the same time in contact, variance is 0. If one foot stays
    up much longer than the other (yoga-walk), variance is high.

    Returns var(last_air_time, clipped at 0.5) + var(last_contact_time, clipped at 0.5).
    The clip prevents runaway penalty during very long stance phases.
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return (
        torch.var(torch.clip(last_air_time, max=0.5), dim=1)
        + torch.var(torch.clip(last_contact_time, max=0.5), dim=1)
    )


def foot_clearance_reward(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swinging feet for clearing `target_height` off the ground.

    Adapted from Boston Dynamics Spot's MDP. The tanh on foot horizontal velocity
    ensures the reward only kicks in while the foot is actually moving (swing
    phase), not while it's planted.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_err = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_xy_speed = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    reward = foot_z_err * torch.tanh(tanh_mult * foot_xy_speed)
    return torch.exp(-torch.sum(reward, dim=1) / std)


def base_height_below_target_l1(
    env: "ManagerBasedRLEnv",
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-sided L1 penalty when the base sags BELOW `target_height`.

    Returns max(0, target - actual_height) per env.
    * Zero if the pelvis is at or above the target — free to stand tall.
    * Linear in shortfall — well-behaved. (Earlier squared variant caused
      `value_loss=inf` because fall events at shortfall ~ 0.8 m produced
      ~0.64 raw penalty, blowing up PPO value targets at weight -50.)
    Use with a NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    shortfall = torch.clamp(target_height - base_height, min=0.0)
    return shortfall


def knee_too_straight_penalty(
    env: "ManagerBasedRLEnv",
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-sided L1 penalty when a knee is straighter than `threshold` rad.

    Returns sum_over_knees(max(0, threshold - knee_angle)) per env.
    * Swing knee (heavily bent, e.g. 0.8 rad) → 0 contribution.
    * Stance knee at the default bend (0.36 rad) → small contribution.
    * Locked-straight stance knee (0.0 rad) → full threshold contribution.
    Use with a NEGATIVE weight.

    Pair with `base_height_below_target_l1` — height-below removes the wall
    that forced rigid stance; this term adds the positive pressure to dip.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    knee_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    shortfall = torch.clamp(threshold - knee_pos, min=0.0)
    return shortfall.sum(dim=1)


def stand_still_joint_deviation_l1(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 deviation from default joint pose, gated to standing commands only.

    Returns sum_over_joints(|q - q_default|) per env, multiplied by a mask
    that is 1.0 when ||cmd_vel|| < command_threshold and 0.0 otherwise.

    * Standing (v_cmd ≈ 0): the term fires, forcing the listed joints toward
      their default values → kills foot cycling / parade-march at standstill.
    * Walking (|v_cmd| ≥ threshold): the term is zero, so it does not fight
      the swing motion the velocity-tracking reward needs.

    Use with a NEGATIVE weight. Target only the swing-relevant joints
    (hip_pitch, knee, ankle_pitch) so hip_roll / ankle_roll remain free for
    static balance compensation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(command[:, :3], dim=1)
    standing_mask = (cmd_norm < command_threshold).float()

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(joint_pos - default_pos), dim=1)
    return deviation * standing_mask


def joint_deviation_no_turn_l1(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 joint deviation from default, gated to non-turning commands only.

    Returns sum_over_joints(|q - q_default|) per env, multiplied by a mask that
    is 1.0 when |ang_vel_z_cmd| < command_threshold and 0.0 otherwise.

    Use for hip_yaw: keeps legs forward during stand / forward / lateral
    motion, but releases the constraint during turn commands so the policy
    can use hip_yaw to pivot in place instead of arc-walking.
    Use with a NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    no_turn_mask = (torch.abs(command[:, 2]) < command_threshold).float()

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(joint_pos - default_pos), dim=1)
    return deviation * no_turn_mask


def joint_vel_no_turn_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 joint-velocity penalty, gated to non-turning commands only.

    Returns sum_over_joints(q_dot^2) per env, multiplied by a mask that is
    1.0 when |ang_vel_z_cmd| < command_threshold and 0.0 otherwise.

    Companion to `joint_deviation_no_turn_l1` — kills the back-and-forth
    hip_yaw cycling during stand / forward / lateral motion, but allows
    hip_yaw velocity during turn commands so in-place pivot is not crushed.
    Use with a NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    no_turn_mask = (torch.abs(command[:, 2]) < command_threshold).float()

    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    vel_sq = torch.sum(torch.square(joint_vel), dim=1)
    return vel_sq * no_turn_mask


def feet_lateral_distance_clearance(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_distance: float = 0.18,
) -> torch.Tensor:
    """One-sided clearance penalty: positive when the two feet are LATERALLY
    closer than `min_distance` (measured in the robot's yaw frame).

    * `asset_cfg.body_ids` must resolve to exactly two bodies (the left and
      right foot links — e.g. body_names=[".*_ankle_roll_link"]).
    * Returns `max(0, min_distance - actual_lateral_distance)` per env, so
      it's zero when the feet have enough lateral clearance and grows as
      they come together. Use with a NEGATIVE weight in RewardsCfg.

    Lateral = Y component in the yaw-aligned base frame, so forward stride
    motion (X separation) doesn't trigger the penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    rel_pos_w = feet_pos_w[:, 1] - feet_pos_w[:, 0]               # (N, 3)
    rel_pos_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), rel_pos_w)
    lateral_distance = torch.abs(rel_pos_yaw[:, 1])               # (N,)
    return torch.clamp(min_distance - lateral_distance, min=0.0)


def randomize_arm_joint_targets(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample joint position targets uniformly and write them as PD targets."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_names = list(asset.data.joint_names)
        joint_ids_list = list(range(len(joint_names)))
    else:
        joint_ids_list = list(joint_ids)
        joint_names = [asset.data.joint_names[i] for i in joint_ids_list]

    device = asset.device
    n_envs = env_ids.numel() if torch.is_tensor(env_ids) else len(env_ids)

    targets = torch.empty(n_envs, len(joint_ids_list), device=device)
    for i, name in enumerate(joint_names):
        if name in position_range:
            lo, hi = position_range[name]
            targets[:, i].uniform_(lo, hi)
        else:
            targets[:, i] = asset.data.joint_pos_target[env_ids, joint_ids_list[i]]

    asset.set_joint_position_target(targets, joint_ids=joint_ids_list, env_ids=env_ids)
