# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP terms for the H1_2 legs-only velocity-tracking (walking) task.

Ported verbatim (modulo docs) from ``config/g1/mdp.py`` — the proven G1 clean
legs-only walk recipe — so this package stays self-contained, matching the
convention used by the other robot configs. The gait/clock/stand-still/feet
terms are robot-agnostic (they key off contact sensors, joint groups, and the
command), so they transfer to H1_2 unchanged.

The one H1_2-specific addition is :func:`randomize_arm_joint_targets` (ported
from ``config/h1_2_stand/mdp.py``): the walker is trained with the arms driven
to per-episode-randomized poses, so the legs learn to keep balance under the
CoM shifts a loco-manipulation policy will later create.

Sign-convention note carried over from G1: H1_2's knee also flexes in the +q
direction (default ≈ +0.36 rad), so :func:`swing_knee_flexion_reward`'s
``clamp(knee, 0)`` remains the correct flexion amount here.
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


def swing_knee_flexion_reward(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    scale: float = 0.8,
) -> torch.Tensor:
    """Reward the SWING knee (its foot is airborne) for flexing — the clean way to
    clear the swing foot, instead of hip-roll circumduction (the "rounded step").

    The ``feet_clearance`` reward only asks for foot HEIGHT, which the policy can
    satisfy either by flexing the knee (clean) OR by rolling/abducting the hip so
    a straight leg swings up in an arc (circumduction). This term gives the knee an
    explicit reason to bend during swing, so height comes from the knee, not the hip.

    Mirror of :func:`knee_bent_stance_penalty`: the STANCE knee is pulled toward
    extension (tall support), the SWING knee toward flexion (foot clearance). The
    reward is a SATURATING ``tanh(flex / scale)`` so it wants "clearly bent" but
    never a fixed target angle — this avoids the crouch↔locked-knee oscillation that
    a hard knee-angle target produced. The stance knee is never rewarded here, so
    it stays free to extend and bear load.

    ``sensor_cfg`` (feet) and ``asset_cfg`` (knees) MUST be built with matching
    left-then-right order and ``preserve_order=True`` so foot *i* gates knee *i*.
    The H1_2 knee flexes in the +q direction (default ≈0.36 rad), so ``clamp(knee, 0)``
    is the flexion amount. Use with a POSITIVE weight.

    Returns ``sum_legs( swing_leg * tanh(max(0, knee_leg) / scale) )``.
    """
    from isaaclab.sensors import ContactSensor

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # current_contact_time <= 0 means the foot is airborne (swing).
    in_swing = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] <= 0.0  # (N, L)
    knee = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (N, L)
    flex = torch.clamp(knee, min=0.0)  # H1_2 knee flexes positive
    return torch.sum(torch.tanh(flex / scale) * in_swing.float(), dim=1)


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


def foot_contact_velocity_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalize |foot vertical velocity| while the foot is in contact — the
    SOFT-LANDING shaper (ported from the tuned tahiti_c1 recipe).

    This is the UPSTREAM half of the two-part contact-force design. Peak ground
    reaction force at touchdown scales with ``m * Δv_z / Δt``, so zeroing the
    foot's downward velocity just before it lands directly cuts the landing
    impulse — the physical cause of a loud, hard-stomping walk. It teaches the
    policy to decelerate the foot before contact rather than paying the impulse
    afterward (which is all the reactive ``mdp.contact_forces`` cap can do).

    Mirrors the ``feet_slide`` pattern but on the world-z axis instead of the xy
    plane. ``force_threshold`` (N) is only the contact detector (foot is "down"
    when |net force| exceeds it), NOT a force cap. Use with a NEGATIVE weight.

    Returns ``sum_feet( |foot_vz| * in_contact )``.
    """
    from isaaclab.sensors import ContactSensor

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = (contact_forces.norm(dim=-1).max(dim=1)[0] > force_threshold).float()

    asset: Articulation = env.scene[asset_cfg.name]
    foot_vz = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    return torch.sum(torch.abs(foot_vz) * in_contact, dim=1)


# ===========================================================================
# Ported from unitree_rl_lab (the reference G1 velocity recipe).
#
# These implement the phase-based gait shaping + energy penalty + velocity
# command curriculum that give the walk its clean, rhythmic gait. Kept verbatim
# (modulo docs) so the behavior matches the proven reference.
# ===========================================================================


def energy(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize mechanical power ``sum_j |joint_vel_j| * |applied_torque_j|``.

    Pushes the policy toward an efficient, natural gait (a stiff-legged or
    over-actuated gait burns more power). Use with a NEGATIVE weight. Scope
    ``asset_cfg.joint_ids`` to the actuated leg joints so it never charges the
    policy for torque the stiff PD spends holding the non-actioned arms/torso.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def _gait_phase_scalar(
    env: "ManagerBasedRLEnv",
    period: float,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Command-gated gait phase in ``[0, 1)`` — the FROZEN-AT-IDLE clock.

    A per-env integer step counter that advances ONLY while a base command is
    present (``||cmd|| > cmd_threshold``) and HOLDS while the robot is idle. The
    phase is ``(count * step_dt) % period / period``. Freezing at standstill is
    what kills the "parade" march: the :func:`gait_phase` observation stops
    cycling when idle, so the policy has nothing driving it to step in place.

    Advanced exactly ONCE per env step, keyed on ``common_step_counter`` (which
    bumps once per ``env.step``). Both :func:`feet_gait` (reward, computed first)
    and :func:`gait_phase` (observation, computed later in the same step) call
    this, so the reward clock and the observed clock are guaranteed identical.
    NOT reset per episode — a continuous accumulator that mirrors the deploy
    runner's single, never-resetting ``step_count`` exactly.
    """
    if not hasattr(env, "_gait_count"):
        env._gait_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._gait_last_step = -1
    if env.common_step_counter != env._gait_last_step:
        env._gait_last_step = env.common_step_counter
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        env._gait_count = env._gait_count + (cmd_norm > cmd_threshold).long()
    return (env._gait_count.float() * env.step_dt) % period / period


def feet_gait(
    env: "ManagerBasedRLEnv",
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str | None = None,
) -> torch.Tensor:
    """Phase-based periodic gait reward — the core of the natural walk.

    A clock of length ``period`` seconds (frozen while idle, see
    :func:`_gait_phase_scalar`) defines, per foot, when it SHOULD
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

    global_phase = _gait_phase_scalar(env, period).unsqueeze(1)  # frozen-at-idle clock
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

    Exposes the SAME frozen-at-idle phase clock :func:`feet_gait` scores against
    (:func:`_gait_phase_scalar`), so the policy can time its steps without a
    history buffer. The clock only advances while a base command is present, so
    at standstill this observation is CONSTANT — nothing drives an idle march.
    At deploy this is a command-gated step counter: ``phase = (count * dt) %
    period / period`` (count bumped only while moving), then ``[sin, cos]`` of
    ``2*pi*phase`` — no encoder history required.
    """
    global_phase = _gait_phase_scalar(env, period)
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def _tracking_quality(env: "ManagerBasedRLEnv", env_ids, reward_term_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a curriculum gate into its two independent questions:

    * ``quality`` — mean tracking reward **per second of episode actually lived**,
      i.e. in ``[0, weight]`` regardless of how long the robot survived.
    * ``alive_frac`` — mean episode length as a fraction of the full episode.

    The original gate divided the episode SUM by ``max_episode_length_s``, which
    silently multiplies the two together. That deadlocks a robot that falls early:
    with episodes ending at 32% of the horizon the term is capped at 0.32 and can
    NEVER clear a 0.8 gate no matter how perfectly it tracks — the command range
    stays at its start value forever, so the robot never gets a reason to walk,
    so it never survives longer. (This is exactly what pinned the H1_2 run at
    lin=0.1/ang=0.2 for 10 000 iterations.) Keeping the two signals separate lets
    the gate ask "does it track well?" AND "does it stay up?" independently.

    Must be called from a curriculum term: ``curriculum_manager.compute()`` runs
    first in ``_reset_idx``, so ``episode_length_buf`` and ``_episode_sums`` still
    hold the finished episode's values.
    """
    ep_len_s = (env.episode_length_buf[env_ids].float() * env.step_dt).clamp(min=env.step_dt)
    quality = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids] / ep_len_s)
    alive_frac = torch.mean(ep_len_s) / env.max_episode_length_s
    return quality, alive_frac


def lin_vel_cmd_levels(
    env: "ManagerBasedRLEnv",
    env_ids,
    reward_term_name: str = "track_lin_vel_xy",
    threshold: float = 0.75,
    min_alive_frac: float = 0.5,
) -> torch.Tensor:
    """Command curriculum: widen the linear-velocity command range by 0.1 m/s
    per side once the robot both TRACKS well and STAYS UP.

    Starts the robot on tiny commands (easy, clean slow walking) and grows the
    range only as it earns it, up to ``cfg.limit_ranges``. Requires the command
    term to be a ``UniformLevelVelocityCommandCfg`` (has ``limit_ranges``). The
    reward term named ``reward_term_name`` must exist (default matches the
    ``track_lin_vel_xy`` term in the env cfg).

    ``threshold`` is on the per-second tracking quality (fraction of the term's
    weight); ``min_alive_frac`` is the survival precondition, so the command range
    never widens while the robot is still falling over. See :func:`_tracking_quality`
    for why the two must be measured separately.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    quality, alive_frac = _tracking_quality(env, env_ids, reward_term_name)

    if env.common_step_counter % env.max_episode_length == 0:
        if quality > reward_term.weight * threshold and alive_frac > min_alive_frac:
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
    threshold: float = 0.5,
    min_alive_frac: float = 0.5,
) -> torch.Tensor:
    """Yaw-rate command curriculum (mirror of :func:`lin_vel_cmd_levels`).

    ``threshold`` defaults LOWER than the linear gate on purpose: a biped's yaw
    tracking reward plateaus well below its linear one (the base yaws constantly
    as a side effect of stepping), so a 0.8 gate is effectively unreachable and
    the yaw range would stay pinned at its start value forever.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    quality, alive_frac = _tracking_quality(env, env_ids, reward_term_name)

    if env.common_step_counter % env.max_episode_length == 0:
        if quality > reward_term.weight * threshold and alive_frac > min_alive_frac:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def push_velocity_levels(
    env: "ManagerBasedRLEnv",
    env_ids,
    term_name: str = "push_robot",
    step: float = 0.05,
    max_velocity: float = 0.5,
    min_alive_frac: float = 0.8,
    start_after_iters: int = 0,
) -> torch.Tensor:
    """Disturbance curriculum: grow the ``push_robot`` velocity kick only once the
    robot reliably survives a full episode AND the walk phase has started.

    ``push_by_setting_velocity`` writes the root velocity DIRECTLY, so a v m/s kick
    displaces the capture point by ``v / sqrt(g / h_com)``. For H1_2 (h_com ~0.85 m)
    a 0.5 m/s push moves it ~0.147 m — past the foot edge, so recovery REQUIRES a
    step. Applying that from iteration 0, before the command curriculum has given
    the robot any reason to learn stepping, made the first push (t = 5 s) fatal:
    100% of terminations were ``bad_orientation``, 79% of them in the 5-8 s window,
    mean pelvis tilt spiking from 0.10 to 0.27 rad within 0.8 s of every push.

    So: learn to walk first, harden second. Start ``velocity_range`` at ~0 in the
    env cfg and let this term raise it as survival is demonstrated.

    ``start_after_iters`` FLOORS the growth to the walk phase. When the command uses
    the fixed :func:`stand_to_walk_command_curriculum`, the robot trivially survives
    the pure-standing phase, so a survival-only gate would ramp the push UP before
    the robot can walk — reintroducing "too much at once". Set this to the schedule's
    ``stand_until_iters`` so push stays at 0 until slow-walking begins. Iterations are
    derived as ``common_step_counter // 24`` (``steps_per_iter``), matching the fixed
    schedule and rsl_rl ``num_steps_per_env=24`` — keep the two in sync.
    """
    term_cfg = env.event_manager.get_term_cfg(term_name)
    vel_range = term_cfg.params["velocity_range"]
    current = float(vel_range["x"][1])

    ep_len_s = (env.episode_length_buf[env_ids].float() * env.step_dt).clamp(min=env.step_dt)
    alive_frac = torch.mean(ep_len_s) / env.max_episode_length_s

    steps_per_iter = 24  # must match agents/rsl_rl_ppo_cfg.py num_steps_per_env
    iters = env.common_step_counter // steps_per_iter

    if (
        env.common_step_counter % env.max_episode_length == 0
        and alive_frac > min_alive_frac
        and iters >= start_after_iters
    ):
        current = min(current + step, max_velocity)
        vel_range["x"] = (-current, current)
        vel_range["y"] = (-current, current)
        term_cfg.params["velocity_range"] = vel_range

    return torch.tensor(current, device=env.device)


def payload_mass_levels(
    env: "ManagerBasedRLEnv",
    env_ids,
    term_name: str = "add_ee_payload",
    start_iters: int = 5000,
    full_iters: int = 9000,
    max_mass: float = 3.0,
) -> torch.Tensor:
    """EE-payload curriculum: grow the per-hand payload mass range 0 -> ``max_mass`` kg
    linearly between ``start_iters`` and ``full_iters``, so the robot learns a stable
    WALK before it must also balance a heavy manipulated object.

    Parallels :func:`push_velocity_levels` and the stand->walk schedule. The push
    disturbance was staged, but the +3 kg EE payload (and the arm-motion DR that swings
    it every 3-5 s) ran at full strength from iter 0, so the robot was learning to walk
    while balancing a heavy, arm-swung payload — too much at once, and a driver of the
    falls. Staging the PAYLOAD also stages the coupled arm x payload disturbance: the
    arm motion is full throughout, but early on it swings ~empty hands, so the
    destabilising moment ramps in with the payload.

    Requires the ``term_name`` event (the EE-payload mass DR) to run in ``mode="reset"``
    so the grown range is re-sampled each episode. Iterations = ``common_step_counter
    // 24`` (matches ``num_steps_per_env``). NOTE: ``common_step_counter`` is not
    restored on ``--resume``, so a resumed run restarts the payload ramp from 0.
    """
    steps_per_iter = 24
    iters = env.common_step_counter // steps_per_iter
    span = max(full_iters - start_iters, 1)
    frac = min(max(iters - start_iters, 0) / span, 1.0)
    current = frac * max_mass

    term_cfg = env.event_manager.get_term_cfg(term_name)
    lo = term_cfg.params["mass_distribution_params"][0]
    term_cfg.params["mass_distribution_params"] = (lo, current)
    return torch.tensor(current, device=env.device)


def stand_to_walk_command_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,
    stand_until_iters: int = 2000,
    slow_until_iters: int = 5000,
    slow_scale: float = 0.3,
    lin_vel_x_full: tuple[float, float] = (-0.5, 1.0),
    lin_vel_y_full: tuple[float, float] = (-0.5, 0.5),
    ang_vel_z_full: tuple[float, float] = (-0.5, 0.5),
    rel_standing_envs_phase1: float = 1.0,
    rel_standing_envs_phase2: float = 0.3,
    rel_standing_envs_phase3: float = 0.1,
) -> torch.Tensor:
    """Three-phase stand→slow-walk→full-walk command curriculum (ported verbatim from
    the tuned ``config/tahiti_c1_velocity`` recipe).

    Phase 1 (iter 0 .. stand_until_iters):     zero cmd,  100 % standing envs.
    Phase 2 (stand_until_iters .. slow):       cmd × slow_scale, 30 % standing.
    Phase 3 (>= slow_until_iters):             full ranges, 10 % standing.

    This REPLACES the performance-gated ``lin/ang_vel_cmd_levels`` for H1_2. Those
    gates deadlocked the 10 000-iteration run (a robot falling early can never clear
    a per-episode quality threshold, so the command range stayed pinned at its start
    value forever). A fixed schedule can't stall: the command simply grows on a
    known iteration timeline, so the robot is guaranteed a reason to start stepping.
    The default ``*_full`` ranges match ``commands.base_velocity.limit_ranges``.

    Iteration count derived from ``env.common_step_counter // 24`` (matches rsl_rl
    default ``num_steps_per_env=24`` in ``agents/rsl_rl_ppo_cfg.py`` — keep in sync).

    Auto-detects ``--resume`` / ``--checkpoint`` / ``--load_run`` from
    ``/proc/self/cmdline`` (train.py wipes ``sys.argv`` before this runs) and, if any
    is present, jumps straight to Phase 3. Rationale: ``env.common_step_counter`` is
    NOT restored on rsl_rl resume, so without this check the counter starts at 0 and
    forces a resumed policy back through the stand phase.
    """
    cmd_term = env.command_manager.get_term("base_velocity")
    cfg = cmd_term.cfg

    if not hasattr(env, "_curriculum_resume_detected"):
        import sys

        try:
            with open("/proc/self/cmdline", "rb") as fh:
                argv = fh.read().decode("utf-8", errors="replace").split("\x00")
        except OSError:
            argv = sys.argv
        env._curriculum_resume_detected = (
            "--resume" in argv
            or any(a == "--checkpoint" or a.startswith("--checkpoint=") for a in argv)
            or any(a == "--load_run" or a.startswith("--load_run=") for a in argv)
        )
        if env._curriculum_resume_detected:
            print("[curriculum] --resume detected → skipping stand→walk curriculum, jumping to Phase 3.")

    if env._curriculum_resume_detected:
        cfg.ranges.lin_vel_x = lin_vel_x_full
        cfg.ranges.lin_vel_y = lin_vel_y_full
        cfg.ranges.ang_vel_z = ang_vel_z_full
        cfg.rel_standing_envs = rel_standing_envs_phase3
        return torch.tensor(3.0, device=env.device)

    steps_per_iter = 24
    iters = env.common_step_counter // steps_per_iter

    if iters < stand_until_iters:
        cfg.ranges.lin_vel_x = (0.0, 0.0)
        cfg.ranges.lin_vel_y = (0.0, 0.0)
        cfg.ranges.ang_vel_z = (0.0, 0.0)
        cfg.rel_standing_envs = rel_standing_envs_phase1
        phase = 1.0
    elif iters < slow_until_iters:
        s = slow_scale
        cfg.ranges.lin_vel_x = (lin_vel_x_full[0] * s, lin_vel_x_full[1] * s)
        cfg.ranges.lin_vel_y = (lin_vel_y_full[0] * s, lin_vel_y_full[1] * s)
        cfg.ranges.ang_vel_z = (ang_vel_z_full[0] * s, ang_vel_z_full[1] * s)
        cfg.rel_standing_envs = rel_standing_envs_phase2
        phase = 2.0
    else:
        cfg.ranges.lin_vel_x = lin_vel_x_full
        cfg.ranges.lin_vel_y = lin_vel_y_full
        cfg.ranges.ang_vel_z = ang_vel_z_full
        cfg.rel_standing_envs = rel_standing_envs_phase3
        phase = 3.0

    return torch.tensor(phase, device=env.device)


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

    NOTE: for the arms, :func:`randomize_arm_joint_targets` runs AFTER this at reset
    and OVERWRITES the arm targets with a randomized pose — so this term effectively
    only pins the torso, while the arms are held at their per-episode random target.
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


def joint_deviation_l1_when_straight(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """L1 joint deviation from the default pose, applied ONLY when NOT commanded to
    turn (``|yaw command| < command_threshold``).

    For hip_yaw on a legs-only walk this keeps the feet pointing FORWARD during
    straight walking (kills the toe-in/out drift that curves the path) while leaving
    the joint completely free to rotate when a yaw command IS present, so turning is
    not blocked. The gate uses only the YAW command component (index 2 of the base
    velocity command), not the full command norm, so a straight-line walk at nonzero
    linear velocity is still penalised. Use with a NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.abs(diff), dim=1)
    yaw_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    return penalty * (yaw_cmd < command_threshold)


# ===========================================================================
# H1_2-specific: arm-target randomization (ported from config/h1_2_stand/mdp.py).
#
# The walker is trained with the arms actively driven to per-episode randomized
# poses (and re-randomized mid-episode on an interval), so the legs learn to keep
# balance under the CoM shifts a loco-manipulation policy will later create.
# ===========================================================================


def randomize_arm_joint_targets(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample arm joint position targets uniformly and apply them as PD targets.

    ``position_range`` is keyed by joint name (URDF name, no env prefix) and maps
    to (lo, hi) bounds. Joints not in the dict keep their existing target. Used in
    ``mode="reset"`` (fresh pose each episode) and ``mode="interval"`` (arm moves
    mid-episode so the legs must react to a live CoM shift, not just a static hold).
    """
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
        else:
            # fall back to the joint's existing target so we don't perturb it
            current = asset.data.joint_pos_target[env_ids, joint_ids_list[i]]
            targets[:, i] = current
            continue
        targets[:, i].uniform_(lo, hi)

    asset.set_joint_position_target(targets, joint_ids=joint_ids_list, env_ids=env_ids)


def arm_target_delta(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation: current arm PD target minus default arm pose, shape (N, |joints|).

    Not used in the deploy-safe policy group (the walker infers arm motion from the
    live upper-body encoder obs instead). Kept as an optional FEED-FORWARD term for
    experiments where the leg policy is told the arm's commanded target directly.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    return asset.data.joint_pos_target[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
