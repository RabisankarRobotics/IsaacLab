# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP terms for the G1 velocity-tracking (walking) task.

Ported from the HV1.2 velocity task's ``mdp.py``. Copied here (instead of
cross-importing between sibling config folders) so this package stays
self-contained, matching the convention used by the other robot configs.

Gait-shaping terms used by ``flat_legs_29dof_env_cfg.py``:

* ``air_time_variance_penalty``    — symmetric stepping (equal L/R cadence).
* ``foot_clearance_reward``        — crisp, visible swing-foot lift.
* ``knee_too_straight_penalty``    — keep a knee bend during stance (stops the
                                     stiff-legged forward step).
* ``feet_lateral_distance_clearance`` — keep a minimum lateral gap between feet.
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
    up much longer than the other (limping / yoga-walk), variance is high.

    Returns var(last_air_time, clipped at 0.5) + var(last_contact_time, clipped at 0.5).
    The clip prevents runaway penalty during very long stance phases.
    Use with a NEGATIVE weight.
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
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
    phase), not while it's planted. Use with a POSITIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_err = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_xy_speed = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    reward = foot_z_err * torch.tanh(tanh_mult * foot_xy_speed)
    return torch.exp(-torch.sum(reward, dim=1) / std)


def knee_too_straight_penalty(
    env: "ManagerBasedRLEnv",
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-sided L1 penalty when a knee is straighter than `threshold` rad.

    Returns sum_over_knees(max(0, threshold - knee_angle)) per env.
    * Swing knee (heavily bent) → 0 contribution.
    * Locked-straight stance knee (≈0 rad) → full threshold contribution.
    Use with a NEGATIVE weight. Adds positive pressure to keep a knee bend
    during the stance/forward step instead of walking stiff-legged.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    knee_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    shortfall = torch.clamp(threshold - knee_pos, min=0.0)
    return shortfall.sum(dim=1)


def stand_still_joint_deviation_l1(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize leg-joint deviation from the default stance when NO velocity is
    commanded — stops the robot stepping/shuffling in place while idle.

    Unlike the stock ``stand_still_joint_deviation_l1`` (which gates on the
    linear command ``[:, :2]`` only, and therefore wrongly fires during a
    turn-in-place yaw command), this gates on the FULL command
    ``[lin_x, lin_y, ang_z]``. So it is active only when the robot is told to
    do nothing, and never penalizes a commanded turn.

    Returns ``sum_j |q_j - q_default_j|`` over ``asset_cfg.joint_ids``, masked to
    the envs whose total commanded velocity is below ``command_threshold``.
    Use with a NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    is_standing = (torch.norm(command[:, :3], dim=1) < command_threshold).float()
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_def = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(q - q_def), dim=1)
    return deviation * is_standing


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
      they come together. Use with a NEGATIVE weight.

    Lateral = Y component in the yaw-aligned base frame, so forward stride
    motion (X separation) doesn't trigger the penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    rel_pos_w = feet_pos_w[:, 1] - feet_pos_w[:, 0]  # (N, 3)
    rel_pos_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), rel_pos_w)
    lateral_distance = torch.abs(rel_pos_yaw[:, 1])  # (N,)
    return torch.clamp(min_distance - lateral_distance, min=0.0)
