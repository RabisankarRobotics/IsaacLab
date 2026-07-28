"""Custom MDP terms for the Tahiti C1 velocity-tracking (walking) task.

Trimmed subset of the HV1.2 velocity MDP — Tahiti C1 has no upper-body joints,
so nothing here has to pin arms / waist / head. The URDF also has a plain
vertical hip-pitch axis (no Cassie-style ±30° splay), so we don't need the
foot-yaw-misalignment reward that HV1.2 uses to compensate for its splayed
hip geometry.
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
    """Penalize variance of left/right air+contact time — symmetric-gait shaper."""
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
    """Reward swing feet for clearing ``target_height``. tanh(velocity) gates
    the reward to swing phase only."""
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
    """One-sided L1: fires only when the base sags BELOW ``target_height``."""
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    return torch.clamp(target_height - base_height, min=0.0)


def knee_too_straight_penalty(
    env: "ManagerBasedRLEnv",
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-sided L1: fires when knee angle is less than ``threshold`` (straighter).

    Swing-phase knees bend well past the threshold (~0.7-1.0 rad) → 0 cost.
    Stance-phase knees near the 0.36 default → tiny cost. Rigid-straight
    stilt-walk stance → full cost. Use with a NEGATIVE weight.
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
    """L1 joint deviation from default, gated to standing commands only.

    Zero during walking (any commanded velocity above ``command_threshold``),
    active at standstill — kills foot cycling / parade march when the operator
    is not commanding motion. Use with a NEGATIVE weight.
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
    """L2 base angular velocity, gated to standing only. Kills standstill sway
    without interfering with walking dynamics. Use with a NEGATIVE weight."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(command[:, :3], dim=1)
    standing_mask = (cmd_norm < command_threshold).float()
    ang_vel = asset.data.root_ang_vel_b
    return torch.sum(ang_vel * ang_vel, dim=1) * standing_mask


def hip_yaw_symmetry_l1(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    turn_softness_std: float = 0.3,
) -> torch.Tensor:
    """|q_L_hip_yaw + q_R_hip_yaw| × exp(-(wz_cmd / std)²).

    Same-sign hip_yaw drift (both L and R rotate the same way) produces
    unwanted body yaw during commanded-straight walking — the source of the
    "walks in a circle" symptom. Symmetric mirror pose q_L = -q_R sums to 0,
    so the penalty is zero for the desired straight-gait posture. The
    exp(-(wz_cmd/std)²) softness releases the penalty when the operator
    commands a turn (both legs then naturally rotate the same way).

    ``asset_cfg.joint_ids`` must resolve to exactly two joints in
    (left, right) order (``preserve_order=True`` when building the cfg).
    Use with a NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (N, 2)
    sym_violation = torch.abs(torch.sum(q, dim=1))
    command = env.command_manager.get_command(command_name)
    wz_cmd = command[:, 2]
    softness = torch.exp(-(wz_cmd * wz_cmd) / (turn_softness_std ** 2))
    return sym_violation * softness


def feet_lateral_distance_clearance(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_distance: float = 0.12,
) -> torch.Tensor:
    """One-sided clearance penalty: positive when the two feet are LATERALLY
    closer than min_distance (measured in the robot's yaw frame). Use with
    NEGATIVE weight. asset_cfg.body_ids must resolve to exactly two feet."""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    rel_pos_w = feet_pos_w[:, 1] - feet_pos_w[:, 0]
    rel_pos_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), rel_pos_w)
    lateral_distance = torch.abs(rel_pos_yaw[:, 1])
    return torch.clamp(min_distance - lateral_distance, min=0.0)


def feet_stance_flat_ankle(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Per-foot L1 ankle_pitch deviation from default, gated to feet in contact.

    Encourages foot-flat stance. Kills the toe-up / heel-only pathology.
    sensor_cfg.body_ids must resolve to the 2 foot contact bodies (L, R);
    asset_cfg.joint_ids to the 2 ankle_pitch joints in the SAME order.
    Use with NEGATIVE weight.
    """
    from isaaclab.sensors import ContactSensor

    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_mag = torch.max(torch.norm(contact_forces, dim=-1), dim=1)[0]
    in_contact = (contact_mag > force_threshold).float()

    ankle_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    ankle_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.abs(ankle_pos - ankle_default)
    return torch.sum(deviation * in_contact, dim=1)


def foot_contact_velocity_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalize |foot vertical velocity| while the foot is in contact.

    Upstream of the reactive `contact_forces` penalty: peak GRF scales with
    Δv_z / Δt at touchdown, so zeroing foot z-velocity before the foot lands
    directly reduces landing impulse. Mirrors the `feet_slide` pattern but
    on the world-z axis instead of the xy plane. Use with NEGATIVE weight.
    """
    from isaaclab.sensors import ContactSensor

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    in_contact = (contact_forces.norm(dim=-1).max(dim=1)[0] > force_threshold).float()

    asset: Articulation = env.scene[asset_cfg.name]
    foot_vz = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    return torch.sum(torch.abs(foot_vz) * in_contact, dim=1)


def leg_symmetry_l1(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """L1 penalty on left/right joint-position asymmetry.

    Reads asset_cfg.joint_ids interleaved as (L, R, L, R, ...) and penalizes
    |L - R| per pair. Fights persistent left-dominance across gait cycles.
    Instantaneous — will oppose normal swing asymmetry, so use SMALL weight.
    Requires preserve_order=True on the SceneEntityCfg.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    n_pairs = joint_pos.shape[1] // 2
    pairs = joint_pos.view(joint_pos.shape[0], n_pairs, 2)
    diff = torch.abs(pairs[..., 0] - pairs[..., 1])
    return diff.sum(dim=1)


def hip_pitch_swing_amplitude(
    env: "ManagerBasedRLEnv",
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    max_amplitude: float = 0.6,
    cmd_threshold: float = 0.1,
    swing_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward |L_hip_pitch - R_hip_pitch| — horizontal-stride shaper.

    A real walking gait keeps one hip flexed forward while the other
    extends back, so the L-R hip-pitch differential tracks stride length
    directly. The vertical-parade failure mode flexes BOTH hips together
    to lift the knees, driving the differential toward zero — this term
    punishes that shape by paying only when hips are opposed.

    Gated to prevent farming:
    - ``cmd_threshold``: no reward when there is no commanded motion, so a
      standing robot cannot split its stance for free reward.
    - ``swing_threshold``: no reward unless at least one foot is currently
      in the air (``current_air_time`` above threshold), so a moving robot
      cannot hold a static split stance for free either.

    Capped at ``max_amplitude`` (rad) — prevents the policy from cashing
    more reward by driving hip pitches to their travel limits and getting
    stuck at the joint ends. 0.6 rad ≈ 34° per hip, matching a healthy
    walking stride's peak hip-pitch swing (Winter's gait data).

    ``asset_cfg.joint_ids`` MUST resolve to exactly two joint ids in the
    order (L_hip_pitch, R_hip_pitch). Set with ``preserve_order=True``.
    """
    from isaaclab.sensors import ContactSensor

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    diff = torch.clamp(torch.abs(joint_pos[:, 0] - joint_pos[:, 1]), max=max_amplitude)

    cmd = env.command_manager.get_command(command_name)
    moving = (torch.norm(cmd[:, :2], dim=1) > cmd_threshold).float()

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    swinging = (current_air.max(dim=1)[0] > swing_threshold).float()

    return diff * moving * swinging


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
    """Three-phase stand→slow-walk→full-walk command curriculum.

    Phase 1 (iter 0 .. stand_until_iters):     zero cmd,  100 % standing envs.
    Phase 2 (stand_until_iters .. slow):       cmd × slow_scale, 30 % standing.
    Phase 3 (>= slow_until_iters):             full ranges, 10 % standing.

    Iteration count derived from ``env.common_step_counter // 24`` (matches
    rsl_rl default ``num_steps_per_env=24``).

    Auto-detects ``--resume`` / ``--checkpoint`` / ``--load_run`` from
    ``/proc/self/cmdline`` (train.py wipes ``sys.argv`` before this runs) and,
    if any is present, jumps straight to Phase 3. Rationale:
    ``env.common_step_counter`` is NOT restored on rsl_rl resume, so without
    this check the counter starts at 0 and forces a resumed policy back
    through 2000 iterations of zero-command standing.
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


# ---------------------------------------------------------------------------
# 2026-07-28 Round Vibration (V1): action-noise wrapper for JointPositionAction.
# Real hardware jitters during walking in ways MuJoCo/Isaac don't reproduce
# (PID inner-loop chatter, gearbox backlash, comm-timing wobble). Injecting
# small uniform noise on the joint-position TARGET each step trains the policy
# to be robust to those sources without another reward tweak.
# ---------------------------------------------------------------------------
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class NoisyJointPositionAction(JointPositionAction):
    """JointPositionAction that adds uniform noise on the joint pos target.

    Noise is applied AFTER scale/offset/clip so ``noise_range`` is in radians
    on the target sent to the actuator — the same units as real hardware
    jitter.
    """

    cfg: "NoisyJointPositionActionCfg"

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        noise = torch.empty_like(self._processed_actions).uniform_(
            self.cfg.noise_range[0], self.cfg.noise_range[1]
        )
        self._processed_actions = self._processed_actions + noise


@configclass
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    """Config for :class:`NoisyJointPositionAction`."""

    class_type: type[ActionTerm] = NoisyJointPositionAction
    noise_range: tuple[float, float] = (-0.02, 0.02)
    """Uniform noise range in radians added to the joint pos target each step."""
