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


def knee_bent_stance_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.25,
) -> torch.Tensor:
    """Penalize a STANCE knee (its foot is on the ground) bent beyond
    ``threshold`` rad — pulls the load-bearing leg toward an extended, "tall"
    stance posture, i.e. the natural walk. The SWING knee is not in contact, so
    it is never penalized and stays free to flex for ground clearance.

    This is the inverse of :func:`knee_too_straight_penalty`. It targets the
    bent-knee "Groucho" crouch, where the pelvis rides low and the robot steps
    from the hip with a permanently flexed knee instead of extending the knee to
    support the body during stance.

    ``sensor_cfg`` (feet) and ``asset_cfg`` (knees) MUST be built with matching
    left-then-right order and ``preserve_order=True`` so that foot *i* gates knee
    *i*. A mismatch would gate the swing knee by the stance foot and penalize the
    necessary swing-phase flexion.

    Returns ``sum_legs( in_contact_leg * max(0, knee_leg - threshold) )``.
    Use with a NEGATIVE weight.
    """
    from isaaclab.sensors import ContactSensor

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # current_contact_time > 0 means the foot is currently on the ground (stance).
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0  # (N, L)
    knee = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (N, L)
    excess = torch.clamp(knee - threshold, min=0.0)  # only the "too bent" part
    return torch.sum(excess * in_contact.float(), dim=1)


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


def stand_still_joint_vel_l1(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize leg-joint *velocity* when NO velocity is commanded — damps the
    idle tremor/vibration that a pure position-deviation stand-still term misses.

    The companion :func:`stand_still_joint_deviation_l1` pulls the *position*
    back to the default stance, but a policy can hold the mean position while
    still oscillating around it (the standing "vibration") — that oscillation is
    exactly non-zero joint velocity. Penalizing ``sum_j |qd_j|`` while idle
    forces the legs to actually go quiet, and (since drift is slow motion) also
    helps stop the feet slowly creeping apart.

    Full-command gated ``[lin_x, lin_y, ang_z]`` like its companion, so a
    commanded turn-in-place is never penalized. Use with a NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    is_standing = (torch.norm(command[:, :3], dim=1) < command_threshold).float()
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qd), dim=1) * is_standing


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


# ===========================================================================
# Ported from unitree_rl_lab (the reference G1 velocity recipe).
#
# These implement the phase-based gait shaping + energy penalty + velocity
# command curriculum that give the unitree G1 walk its clean, rhythmic gait.
# Used by ``flat_legs_29dof_clean_env_cfg.py``. Kept verbatim (modulo docs) so
# the behavior matches the proven reference.
# ===========================================================================


def energy(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize mechanical power ``sum_j |joint_vel_j| * |applied_torque_j|``.

    Pushes the policy toward an efficient, natural gait (a stiff-legged or
    over-actuated gait burns more power). Use with a NEGATIVE weight. Scope
    ``asset_cfg.joint_ids`` to the actuated leg joints so it never charges the
    policy for torque the stiff PD spends holding the non-actioned arms/waist.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def feet_gait(
    env: "ManagerBasedRLEnv",
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str | None = None,
) -> torch.Tensor:
    """Phase-based periodic gait reward — the core of the natural walk.

    A fixed clock of length ``period`` seconds defines, per foot, when it SHOULD
    be in stance vs swing. ``offset`` gives each foot's phase shift; ``[0.0,
    0.5]`` puts the two feet in anti-phase (alternating steps). ``threshold`` is
    the stance duty (fraction of the cycle a foot should be planted). Each
    control step, every foot scores +1 when its actual contact matches the
    scheduled state (stance & in-contact, OR swing & airborne), else 0.

    This gives the policy a rhythmic scaffold; the knee/hip motion needed to
    satisfy the rhythm emerges on its own — no per-joint angle clamping (that
    was the crouch↔locked-knee trap). Use with a POSITIVE weight. The SAME phase
    clock is exposed to the policy via :func:`gait_phase` so it can phase-lock.
    Command-gated (``command_name``) so it does not force stepping while idle.

    ``sensor_cfg.body_ids`` order pairs with ``offset`` order.
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


def gait_phase(env: "ManagerBasedRLEnv", period: float) -> torch.Tensor:
    """2-value (sin, cos) clock of the gait phase — the deploy-safe replacement
    for an observation history.

    Exposes the SAME phase clock :func:`feet_gait` scores against, so the policy
    can time its steps without a history buffer. At deploy this is just a
    free-running timer: ``phase = (t / period) % 1``, then ``[sin, cos]`` of
    ``2*pi*phase`` — no encoder history required.
    """
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def lin_vel_cmd_levels(
    env: "ManagerBasedRLEnv",
    env_ids,
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """Command curriculum: widen the linear-velocity command range by 0.1 m/s
    per side once mean linear-tracking reward clears 80% of its weight.

    Starts the robot on tiny commands (easy, clean slow walking) and grows the
    range only as it earns it, up to ``cfg.limit_ranges``. Requires the command
    term to be a ``UniformLevelVelocityCommandCfg`` (has ``limit_ranges``). The
    reward term named ``reward_term_name`` must exist (default matches the
    ``track_lin_vel_xy`` term in the clean env cfg).
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: "ManagerBasedRLEnv",
    env_ids,
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    """Yaw-rate command curriculum (mirror of :func:`lin_vel_cmd_levels`), grown
    once yaw tracking clears 80% of its weight."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def hold_joint_targets_at_default(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Set the PD *position target* of the given joints to their default pose at reset.

    Required for joints NOT covered by an action term. IsaacLab initialises
    ``joint_pos_target`` to ZERO and only the action term overwrites it (here: the
    legs). Without this, the implicit PD would drive the un-actioned upper-body
    joints toward joint-angle 0 instead of the default (bent-arm) pose — and it
    would not match the deploy PD, which holds them at the default. The target
    persists across steps, so writing it once per reset holds the joints at their
    default for the whole episode (the soft PD lets them comply under gravity around
    that target).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    default_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    asset.set_joint_position_target(default_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def stand_still_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """L1 penalty on leg-joint deviation from the default stance while the robot is
    commanded to stand still (``|command| < command_threshold``).

    Stops the "march in place" reflex: the gait clock keeps ticking in the
    observation, so without this the policy keeps stepping to the beat even at zero
    command. Gated at the same threshold as ``feet_gait`` so walking is untouched.
    Use with a NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.abs(diff), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return penalty * (cmd_norm < command_threshold)


def stand_still_joint_vel_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """L1 penalty on leg-joint VELOCITY while the robot is commanded to stand still.

    Companion to :func:`stand_still_penalty` (which penalises POSITION deviation).
    A "march in place" cycles the legs back through the default between steps, so the
    position error is near-zero except at the mid-step extremes — a weak signal. The
    marching MOTION, however, shows up directly as joint velocity, so this term gives
    a much sharper gradient toward truly standing still. Gated at the same threshold
    as ``feet_gait``; self-extinguishes (→0) once the robot actually holds still. Use
    with a NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.abs(joint_vel), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return penalty * (cmd_norm < command_threshold)
