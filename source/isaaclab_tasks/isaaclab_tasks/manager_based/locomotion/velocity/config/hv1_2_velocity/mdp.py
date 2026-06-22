"""Custom MDP terms for the HV1.2 velocity-tracking (walking) task.

Same as the standing task — re-exported here so this package is self-contained
and we don't cross-import between sibling task folders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

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


def stand_still_base_ang_vel_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty on base angular velocity, gated to standing commands only.

    Returns ||base_ang_vel||^2 per env, multiplied by a 0/1 mask that fires
    only when ||cmd_vel||_xyz < command_threshold.

    Targets the visible standing sway directly. The always-on ang_vel_xy_l2
    (weight -0.12) damps angular velocity in general, but its weight has to
    stay small because walking gait naturally produces base angular velocity.
    This term fires only at standstill, so it can be much stronger without
    fighting the walking dynamics.

    Use with a NEGATIVE weight (typical -2 to -5).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(command[:, :3], dim=1)
    standing_mask = (cmd_norm < command_threshold).float()

    ang_vel = asset.data.root_ang_vel_b  # body-frame angular velocity (3-vec)
    sq = torch.sum(ang_vel * ang_vel, dim=1)
    return sq * standing_mask


def hip_yaw_symmetry_l1(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    turn_softness_std: float = 0.3,
) -> torch.Tensor:
    """L1 penalty on the SIGNED SUM of left/right hip_yaw — fires only when the
    commanded yaw rate is small (i.e. during forward / sideways walking).

    Returns |q_left_hip_yaw + q_right_hip_yaw| * exp(-(wz_cmd/std)²) per env.

    Why the softening (REQUIRED — penalty without it fights turning):
    * Straight walking (wz_cmd ≈ 0): the symmetric-mirror pose has q_L = -q_R,
      so |q_L + q_R| ≈ 0 ⇒ no penalty. An asymmetric drift (q_L ≈ q_R ≈ +δ)
      that produces unwanted body yaw IS penalized. This is exactly what we
      want to break the "walks in a circle when commanded forward" symptom.
    * Commanded turning (wz_cmd large): turning ALSO produces a same-sign
      hip_yaw bias (both yaw CCW for a left turn, both CW for right). Without
      softening, the penalty would actively fight the track_ang_vel_z_exp
      reward and the policy would turn awkwardly. The softness factor
      exp(-(wz_cmd/std)²) dies smoothly as |wz_cmd| grows, returning hip_yaw
      freedom to the policy precisely when the command needs it.

    With turn_softness_std=0.3:
      wz_cmd = 0.00 → softness 1.00 (full penalty — straight walking)
      wz_cmd = 0.15 → softness 0.78
      wz_cmd = 0.30 → softness 0.37
      wz_cmd = 0.50 → softness 0.06 (essentially off — full turn allowed)

    asset_cfg.joint_ids must point to exactly two joints (left, right) with
    preserve_order=True so the ordering is stable. Use with a NEGATIVE weight
    (typical -0.3 to -1.0).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]  # shape (N, 2)
    sym_violation = torch.abs(torch.sum(q, dim=1))

    command = env.command_manager.get_command(command_name)
    wz_cmd = command[:, 2]  # base_velocity command is [vx, vy, wz]
    softness = torch.exp(-(wz_cmd * wz_cmd) / (turn_softness_std ** 2))
    return sym_violation * softness


def joint_deviation_turn_softened_l1(
    env: "ManagerBasedRLEnv",
    command_name: str,
    turn_softness_std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 joint deviation from default, with smooth softening during turn commands.

    Returns sum_over_joints(|q - q_default|) per env, scaled by a softness
    factor exp(-(ang_vel_z_cmd^2) / turn_softness_std^2):
        ang_vel_z = 0.0  → softness 1.00 (full penalty)
        ang_vel_z = 0.25 → softness 0.78
        ang_vel_z = 0.50 → softness 0.37

    Replaces the earlier hard-mask `joint_deviation_no_turn_l1`. The hard mask
    released the penalty entirely above a threshold — and since ang_vel_z is
    sampled uniformly from (-0.5, 0.5), that left hip_yaw unpenalized in 72%
    of envs. hip_yaw drifted to its mechanical limit (one-leg 90° outward
    collapse). Soft scaling keeps a permanent restoring force on hip_yaw
    while still relaxing the constraint when the policy needs to turn.

    Use with a NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    softness = torch.exp(-(command[:, 2] ** 2) / (turn_softness_std ** 2))

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(joint_pos - default_pos), dim=1)
    return deviation * softness


def joint_vel_turn_softened_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    turn_softness_std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 joint-velocity penalty, with smooth softening during turn commands.

    Same softness factor as `joint_deviation_turn_softened_l1`. Always firing
    but reduced when the policy needs hip_yaw velocity to turn — kills the
    back-and-forth swing-arc cycling without crushing in-place pivot.
    Use with a NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    softness = torch.exp(-(command[:, 2] ** 2) / (turn_softness_std ** 2))

    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    vel_sq = torch.sum(torch.square(joint_vel), dim=1)
    return vel_sq * softness


def foot_yaw_misalignment_l1(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 penalty on foot forward-axis misalignment with body forward-axis (yaw plane).

    For each foot link in ``asset_cfg.body_ids``:
      1. Take the foot's +X axis (forward at default pose, verified from URDF
         inertial origin), transform to world.
      2. Take the body's +X axis (pelvis forward), transform to world.
      3. Project both to the horizontal plane (zero Z, normalize).
      4. Compute the signed yaw angle between them via ``atan2(cross_z, dot)``.

    Returns ``sum_over_feet(|yaw_misalignment_rad|)`` per env.
    Use with a NEGATIVE weight.

    Penalizes the OUTCOME (foot pointing sideways in body yaw frame) rather
    than the MEANS (hip_yaw joint angle). The HV1.2 URDF has a ±30° X-roll
    pre-rotation on hip_pitch (Cassie-style splayed-hip design): pure forward
    leg swing kinematically displaces the foot 3-6 cm laterally outward for
    a normal stride, with the policy's hip_yaw at exactly 0. The policy must
    use hip_yaw INWARD during swing to cancel that drift. A direct
    joint_deviation_hip_yaw penalty (pulling hip_yaw → 0) actively prevents
    this compensation. This reward decouples the goal from the means — the
    policy is free to use whatever combination of hip_yaw / hip_roll / foot
    orientation gets the foot pointing forward.

    Note: only the yaw component of misalignment is penalized. Foot pitch
    (toe-up during swing, toe-down at landing) is unconstrained.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    forward_local = torch.tensor([1.0, 0.0, 0.0], device=asset.device, dtype=torch.float32)

    # ---- Body (pelvis) forward axis in world, projected to horizontal -----
    body_quat_w = asset.data.root_quat_w  # (N, 4) wxyz
    N = body_quat_w.shape[0]
    body_forward_w = quat_apply(body_quat_w, forward_local.expand(N, 3))  # (N, 3)
    body_forward_yaw = body_forward_w.clone()
    body_forward_yaw[..., 2] = 0.0
    body_forward_yaw = body_forward_yaw / (
        torch.norm(body_forward_yaw, dim=-1, keepdim=True) + 1e-8
    )

    # ---- Each foot's forward axis in world, projected to horizontal ------
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # (N, K, 4)
    K = foot_quat_w.shape[1]
    foot_quat_flat = foot_quat_w.reshape(N * K, 4)
    foot_forward_w = quat_apply(
        foot_quat_flat, forward_local.expand(N * K, 3)
    ).reshape(N, K, 3)
    foot_forward_yaw = foot_forward_w.clone()
    foot_forward_yaw[..., 2] = 0.0
    foot_forward_yaw = foot_forward_yaw / (
        torch.norm(foot_forward_yaw, dim=-1, keepdim=True) + 1e-8
    )

    # ---- Signed yaw angle between body forward and each foot forward -----
    body_fwd = body_forward_yaw.unsqueeze(1)  # (N, 1, 3)
    cross_z = (
        body_fwd[..., 0] * foot_forward_yaw[..., 1]
        - body_fwd[..., 1] * foot_forward_yaw[..., 0]
    )  # (N, K)
    dot_val = (body_fwd * foot_forward_yaw).sum(dim=-1)  # (N, K)
    misalign = torch.abs(torch.atan2(cross_z, dot_val))  # (N, K)

    return misalign.sum(dim=-1)  # (N,)


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


def stand_to_walk_command_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,
    stand_until_iters: int = 2000,
    slow_until_iters: int = 5000,
    slow_scale: float = 0.3,
    lin_vel_x_full: tuple[float, float] = (-1.0, 1.0),
    lin_vel_y_full: tuple[float, float] = (-0.5, 0.5),
    ang_vel_z_full: tuple[float, float] = (-0.5, 0.5),
    rel_standing_envs_phase1: float = 1.0,
    rel_standing_envs_phase2: float = 0.3,
    rel_standing_envs_phase3: float = 0.1,
) -> torch.Tensor:
    """Three-phase command curriculum for sim-to-real DelayedPD training.

    Phase 1 (iter 0 .. stand_until_iters):
        Zero velocity commands in all axes. 100% standing envs. Policy learns
        to hold balance under 0-30 ms actuator delay with no locomotion task.

    Phase 2 (stand_until_iters .. slow_until_iters):
        Command ranges scaled to ``slow_scale * full`` in all axes. ~30% of
        envs still standing. Policy introduces slow walking while delay
        handling is preserved.

    Phase 3 (iter >= slow_until_iters):
        Full command ranges in all axes. 10% standing envs.

    Iteration count is derived from ``env.common_step_counter / 24`` (assumes
    rsl_rl's default ``num_steps_per_env = 24``). Adjust the thresholds if you
    use a different rollout length.

    Mutates ``env.command_manager.get_term("base_velocity").cfg.ranges`` and
    ``.rel_standing_envs`` in-place; the next ``_resample_command`` call picks
    them up automatically. Returns the current phase (1/2/3) for logging.

    Auto-skip on resume: when the training script was launched with --resume
    (or --checkpoint / --load_run), the curriculum jumps straight to Phase 3.
    Rationale: env.common_step_counter is NOT restored from the rsl_rl
    checkpoint, so on a fresh process it starts at 0 — that would force
    Phase 1 (zero commands, 100% standing envs) and wipe the resumed
    policy's walking ability over ~2000 iters of re-standing. The CLI
    flag is the canonical signal that "this process is continuing a
    trained policy", so we detect it once and pin Phase 3.
    """
    cmd_term = env.command_manager.get_term("base_velocity")
    cfg = cmd_term.cfg

    # One-time detection of --resume / --checkpoint / --load_run in the
    # original launch command. We CANNOT use sys.argv: train.py at
    # scripts/reinforcement_learning/rsl_rl/train.py line 48 wipes it to
    # `[script] + hydra_args` before this code runs, so the resume flags are
    # gone by the time the curriculum is called. Read /proc/self/cmdline
    # instead — the kernel keeps the original launch string intact regardless
    # of Python-level argv reassignment. Falls back to sys.argv on non-Linux.
    if not hasattr(env, "_curriculum_resume_detected"):
        import os, sys
        try:
            with open("/proc/self/cmdline", "rb") as fh:
                argv = fh.read().decode("utf-8", errors="replace").split("\x00")
        except OSError:
            argv = sys.argv  # non-Linux fallback
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

    steps_per_iter = 24  # rsl_rl PPO num_steps_per_env
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
