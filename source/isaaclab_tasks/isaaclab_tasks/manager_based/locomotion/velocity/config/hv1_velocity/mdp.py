"""Custom MDP terms for the HV1 unified standing+walking task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_below_target_l1(
    env: "ManagerBasedRLEnv",
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-sided L1 penalty when the base sags BELOW `target_height`.

    Returns max(0, target - actual_height). Use with a NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    return torch.clamp(target_height - base_height, min=0.0)


def feet_lateral_distance_clearance(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_distance: float = 0.18,
) -> torch.Tensor:
    """One-sided clearance penalty when the two feet are LATERALLY closer than
    `min_distance` in the robot's yaw frame.

    Returns max(0, min_distance - actual_lateral_distance). Use NEGATIVE weight.
    `asset_cfg.body_ids` must resolve to exactly two bodies (left, right foot).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    rel_pos_w = feet_pos_w[:, 1] - feet_pos_w[:, 0]
    rel_pos_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), rel_pos_w)
    lateral_distance = torch.abs(rel_pos_yaw[:, 1])
    return torch.clamp(min_distance - lateral_distance, min=0.0)


def projected_gravity_body(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gravity unit vector projected into the specified body's link frame.

    Mirrors `mdp.projected_gravity` (which reads root only) but for an
    arbitrary body. `asset_cfg.body_ids` must resolve to exactly one body.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    body_idx = body_ids[0] if not isinstance(body_ids, int) else body_ids
    body_quat = asset.data.body_link_quat_w[:, body_idx]
    return quat_apply_inverse(body_quat, asset.data.GRAVITY_VEC_W)


def body_ang_vel_b(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Angular velocity of a body expressed in its own link frame.

    `asset_cfg.body_ids` must resolve to exactly one body.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    body_idx = body_ids[0] if not isinstance(body_ids, int) else body_ids
    body_quat = asset.data.body_link_quat_w[:, body_idx]
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_idx]
    return quat_apply_inverse(body_quat, body_ang_vel_w)


def flat_orientation_l2_body(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 of the body-frame gravity xy components for a given link.

    Mirrors `mdp.flat_orientation_l2` (root only) but for an arbitrary body —
    used to keep the torso flat once the waist is policy-controlled.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    body_idx = body_ids[0] if not isinstance(body_ids, int) else body_ids
    body_quat = asset.data.body_link_quat_w[:, body_idx]
    g_b = quat_apply_inverse(body_quat, asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(g_b[:, :2]), dim=1)


def body_ang_vel_xy_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 of the body-frame xy angular velocity for a given link.

    Mirrors `mdp.ang_vel_xy_l2` (root only) but for an arbitrary body.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    body_idx = body_ids[0] if not isinstance(body_ids, int) else body_ids
    body_quat = asset.data.body_link_quat_w[:, body_idx]
    w_b = quat_apply_inverse(body_quat, asset.data.body_link_ang_vel_w[:, body_idx])
    return torch.sum(torch.square(w_b[:, :2]), dim=1)


# =============================================================================
# V3 additions — scalar commands (body height, waist regularization α_t),
# body-height tracking reward, and α-weighted waist-deviation reward.
# =============================================================================


class UniformScalarCommand(CommandTerm):
    """Per-env scalar command sampled uniformly from a (lo, hi) range.

    Either linear-uniform (default) or log-uniform (sample in log10 space then
    exponentiate — useful for the waist regularization weight α_t ∈ [0.1, 10]).

    If `metric_source` is set (e.g. "root_pos_z"), the abs error between the
    command and that source is published as `Metrics/<command_name>/error`
    every step so it shows up in tensorboard.

    Debug viz: when `metric_source == "root_pos_z"` and `debug_vis=True`, two
    flat horizontal plates are drawn at the robot's xy:
      * red  — commanded height
      * green — current pelvis height
    so you can see the gap during PLAY.
    """

    cfg: "UniformScalarCommandCfg"

    def __init__(self, cfg: "UniformScalarCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 1, device=self.device)
        if cfg.metric_source is not None:
            self.metrics["error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        lo, hi = self.cfg.range
        if self.cfg.log_uniform:
            import math
            log_lo = math.log10(lo)
            log_hi = math.log10(hi)
            u = torch.empty(len(env_ids), 1, device=self.device).uniform_(log_lo, log_hi)
            self._command[env_ids] = torch.pow(10.0, u)
        else:
            self._command[env_ids] = torch.empty(
                len(env_ids), 1, device=self.device
            ).uniform_(lo, hi)

    def _update_command(self):
        pass

    def _update_metrics(self):
        if self.cfg.metric_source is None:
            return
        asset = self._env.scene[self.cfg.asset_name]
        if self.cfg.metric_source == "root_pos_z":
            actual = asset.data.root_pos_w[:, 2]
        else:
            raise ValueError(f"Unknown metric_source: {self.cfg.metric_source}")
        self.metrics["error"] = torch.abs(actual - self._command.squeeze(-1))

    def _set_debug_vis_impl(self, debug_vis: bool):
        # Only meaningful for height (needs robot xy + a z value).
        if self.cfg.metric_source != "root_pos_z":
            return
        if debug_vis:
            if not hasattr(self, "goal_height_visualizer"):
                self.goal_height_visualizer = VisualizationMarkers(
                    self.cfg.goal_height_visualizer_cfg
                )
                self.current_height_visualizer = VisualizationMarkers(
                    self.cfg.current_height_visualizer_cfg
                )
            self.goal_height_visualizer.set_visibility(True)
            self.current_height_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_height_visualizer"):
                self.goal_height_visualizer.set_visibility(False)
                self.current_height_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self.cfg.metric_source != "root_pos_z":
            return
        asset = self._env.scene[self.cfg.asset_name]
        if not asset.is_initialized:
            return
        xy = asset.data.root_pos_w[:, :2]
        goal_z = self._command.squeeze(-1)
        cur_z = asset.data.root_pos_w[:, 2]
        # Plate at (root_x, root_y, z) — identity orientation, no scale change.
        goal_pos = torch.stack([xy[:, 0], xy[:, 1], goal_z], dim=-1)
        cur_pos = torch.stack([xy[:, 0], xy[:, 1], cur_z], dim=-1)
        self.goal_height_visualizer.visualize(translations=goal_pos)
        self.current_height_visualizer.visualize(translations=cur_pos)


# Flat 40x40 cm plate, 1 cm thick — horizontal disk-like marker.
_GOAL_HEIGHT_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "plate": sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
        ),
    }
)

_CURRENT_HEIGHT_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "plate": sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1)),
        ),
    }
)


@configclass
class UniformScalarCommandCfg(CommandTermCfg):
    """Cfg for a per-episode scalar command sampled uniformly (or log-uniformly).

    Used in V3 for the body-height target h^des and waist-regularization α_t.

    If `metric_source` is given, an abs-error metric is exposed each step
    against the asset's `asset_name` (default "robot"). Currently supported:
      * "root_pos_z" — for body-height tracking.

    When `metric_source == "root_pos_z"` and `debug_vis=True`, the height
    command and current pelvis z are visualized as two flat plates.
    """

    class_type: type = UniformScalarCommand
    range: tuple[float, float] = MISSING
    log_uniform: bool = False
    asset_name: str = "robot"
    metric_source: str | None = None

    goal_height_visualizer_cfg: VisualizationMarkersCfg = _GOAL_HEIGHT_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_height_goal"
    )
    current_height_visualizer_cfg: VisualizationMarkersCfg = _CURRENT_HEIGHT_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_height_current"
    )


def base_height_tracking_exp(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking a body-height command via exp(-err² / std²).

    Reads the commanded height from `command_name` (a UniformScalarCommand,
    so shape (N, 1)). Compares against `root_pos_w[:, 2]` (world-z of the
    pelvis since HV1 spawns on flat terrain).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name).squeeze(-1)
    err_sq = torch.square(asset.data.root_pos_w[:, 2] - target)
    return torch.exp(-err_sq / (std ** 2))


def base_height_above_command_l1(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-sided L1 penalty when the base stands TALLER than the commanded height.

    Symmetric counterpart to `base_height_below_target_l1`, but reads the
    threshold per-env from a `UniformScalarCommand` instead of a fixed scalar.
    Returns max(0, actual_height - commanded_height). Use NEGATIVE weight.

    Pairs with `base_height_tracking_exp(std=0.05)`: the exp reward gives the
    fine gradient near target; this L1 keeps pushing once exp saturates so the
    policy cannot park 5–10 cm above the commanded crouch and ignore it.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name).squeeze(-1)
    return torch.clamp(asset.data.root_pos_w[:, 2] - target, min=0.0)


def joint_deviation_l1_alpha_weighted(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L1 joint-position deviation from default, scaled by per-env α_t command.

    Used in V3 to make the waist regularization weight a sampled command rather
    than a fixed reward weight. The reward weight in the RewTerm should be -1.0
    (or similar); the per-env α_t in `command_name` modulates the strength.

    α_t is sampled per episode from a log-uniform [0.1, 10] range, so the
    effective per-episode penalty spans two decades. The policy gets α_t as an
    observation so it can adapt its waist usage.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    alpha = env.command_manager.get_command(command_name).squeeze(-1)  # (N,)
    joint_dev = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=-1,
    )
    return alpha * joint_dev


def action_rate_l2_joint_subset(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_term_name: str = "joint_pos",
) -> torch.Tensor:
    """L2 squared action rate over the action positions of a joint subset.

    The built-in ``action_rate_l2`` sums (a_t - a_{t-1})² across the full action
    vector. This variant lets us tune per-joint-group action smoothness — e.g.,
    extra arm-jitter penalty without touching legs.

    Maps ``asset_cfg.joint_ids`` (resolved from ``joint_names``) to positions in
    the action vector by reading the named action term's ``_joint_ids`` order.
    Cached on the env on first call.
    """
    if not hasattr(env, "_action_rate_subset_cache"):
        env._action_rate_subset_cache = {}
    cache_key = (action_term_name, tuple(int(j) for j in asset_cfg.joint_ids))
    positions = env._action_rate_subset_cache.get(cache_key)
    if positions is None:
        action_term = env.action_manager.get_term(action_term_name)
        term_joint_ids = action_term._joint_ids
        if isinstance(term_joint_ids, slice):
            term_joint_ids = list(range(env.scene[asset_cfg.name].num_joints))
        jid_to_pos = {int(jid): i for i, jid in enumerate(term_joint_ids)}
        positions = [jid_to_pos[int(jid)] for jid in asset_cfg.joint_ids]
        env._action_rate_subset_cache[cache_key] = positions
    delta = (
        env.action_manager.action[:, positions]
        - env.action_manager.prev_action[:, positions]
    )
    return torch.sum(torch.square(delta), dim=1)


def kmp_residual_l2(
    env: "ManagerBasedRLEnv",
    action_term_name: str = "joint_pos",
) -> torch.Tensor:
    """HiWET Eq. 12: penalty on the actor's residual on top of the KMP prior.

    When the action term is a `KMPResidualJointPositionAction`, the actor's
    raw output IS the residual the network adds to `q_prior`. Penalizing its
    L2 norm keeps the actor's correction small and anchored to the KMP's
    kinematically-feasible posture, while still allowing deviation when
    walking dynamics demand it.

    Equivalent to action_l2 over the joint_pos term. Kept as a separate name
    so it shows up as Episode_Reward/r_kmp in tensorboard with the right
    semantic identity (this is the paper's r_kmp, not generic regularization).
    """
    raw = env.action_manager.get_term(action_term_name).raw_actions
    return torch.sum(torch.square(raw), dim=1)


# =============================================================================
# V5 additions — world-frame EE pose tracking (episode-static, spherical sample,
# curriculum-controlled reach distance), navigation-progress reward, pelvis-to-
# target observation, and a curriculum term that unlocks the distance cap.
# =============================================================================


class WorldFramePoseCommand(CommandTerm):
    """Per-env SE(3) EE pose command sampled in **env-local world** coords.

    Coupling with V4-style body-frame EE commands is the key V5 motivation:
    here the target is FIXED for the whole episode (no per-N-second resample),
    so the only way to drive the EE error to zero is to walk the pelvis to
    within reach. Gait and reach become mechanically coupled.

    Sampling:
      * Spherical around an XY "anchor" (per arm — left arm anchor is on the
        left of the env, right on the right) at a configurable height.
      * `r ∈ [r_min, r_max]` where r_max is mutable at runtime (curriculum).
      * Azimuth `θ ∈ ranges.theta`, elevation `φ ∈ ranges.phi`.
      * Orientation: uniform euler in `ranges.roll/pitch/yaw`.

    Resample timing: `cfg.resampling_time_range` should be set equal to the
    episode length so the command is only resampled on reset.

    Metrics:
      * `position_error` — ‖p_ee_world − target_world‖
      * `orientation_error` — angular distance between quats

    The command tensor returned by `command` is (N, 7) = (pos, quat-wxyz) in
    env-local world. Pair with `KMPResidualJointPositionActionV5`, which
    converts to body frame at action time.
    """

    cfg: "WorldFramePoseCommandCfg"

    def __init__(self, cfg: "WorldFramePoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # Buffer in env-local world frame: (x, y, z, qw, qx, qy, qz)
        self.pose_command_world = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_world[:, 3] = 1.0

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        # Mutable distance cap so curriculum can widen it later. Initialized to
        # cfg.ranges.r so first sample uses the configured initial bounds.
        self._r_min = float(cfg.ranges.r[0])
        self._r_max = float(cfg.ranges.r[1])

        # Cache env_origins (absolute world) so vis markers can be placed.
        self._env_origins = env.scene.env_origins

    @property
    def command(self) -> torch.Tensor:
        """The (env-local world) pose command. Shape (N, 7), quat wxyz."""
        return self.pose_command_world

    def set_distance_range(self, r_min: float, r_max: float):
        """Curriculum hook — bump the sampling radius bounds.

        New bounds take effect on the NEXT reset/resample, not retroactively.
        """
        self._r_min = float(r_min)
        self._r_max = float(r_max)

    @property
    def distance_range(self) -> tuple[float, float]:
        return (self._r_min, self._r_max)

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        device = self.device
        cfg = self.cfg

        r = torch.empty(n, device=device).uniform_(self._r_min, self._r_max)
        theta = torch.empty(n, device=device).uniform_(*cfg.ranges.theta)
        phi = torch.empty(n, device=device).uniform_(*cfg.ranges.phi)

        # Anchor (env-local). For two-arm symmetry, left anchor sits at +y and
        # right at -y (configurable via cfg.anchor_xy).
        ax, ay = cfg.anchor_xy
        az = cfg.anchor_z

        cos_phi = torch.cos(phi)
        dx = r * torch.cos(theta) * cos_phi
        dy = r * torch.sin(theta) * cos_phi
        dz = r * torch.sin(phi)

        self.pose_command_world[env_ids, 0] = ax + dx
        self.pose_command_world[env_ids, 1] = ay + dy
        self.pose_command_world[env_ids, 2] = az + dz

        euler = torch.empty(n, 3, device=device)
        euler[:, 0].uniform_(*cfg.ranges.roll)
        euler[:, 1].uniform_(*cfg.ranges.pitch)
        euler[:, 2].uniform_(*cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self.pose_command_world[env_ids, 3:7] = quat_unique(quat)

    def _update_command(self):
        # Episode-static — nothing to do per step.
        pass

    def _update_metrics(self):
        # EE body pose in absolute world.
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx]

        # Convert command env-local → absolute world by adding env_origins.
        target_pos_abs = self.pose_command_world[:, :3] + self._env_origins
        target_quat = self.pose_command_world[:, 3:7]

        self.metrics["position_error"] = torch.norm(ee_pos_w - target_pos_abs, dim=-1)

        # Orientation error: angle of relative quaternion.
        # rel = q_target^{-1} * q_ee
        q_target_conj = target_quat.clone()
        q_target_conj[:, 1:] = -q_target_conj[:, 1:]
        q_rel = quat_mul(q_target_conj, ee_quat_w)
        # Angular distance = 2 * acos(|w|)
        self.metrics["orientation_error"] = 2.0 * torch.acos(
            torch.clamp(torch.abs(q_rel[:, 0]), max=1.0)
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        target_pos_abs = self.pose_command_world[:, :3] + self._env_origins
        target_quat = self.pose_command_world[:, 3:7]
        self.goal_pose_visualizer.visualize(target_pos_abs, target_quat)


_WORLD_GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 1.0)),
        ),
    }
)


@configclass
class WorldFramePoseCommandCfg(CommandTermCfg):
    """Cfg for `WorldFramePoseCommand` (env-local world, episode-static)."""

    @configclass
    class Ranges:
        r: tuple[float, float] = MISSING
        """Initial radial distance bounds (m). Mutable at runtime via curriculum."""
        theta: tuple[float, float] = MISSING
        """Azimuth angle bounds (rad). 0 = +x ahead, π/2 = +y left."""
        phi: tuple[float, float] = MISSING
        """Elevation angle bounds (rad). 0 = horizontal, +π/2 = straight up."""
        roll: tuple[float, float] = (0.0, 0.0)
        pitch: tuple[float, float] = (0.0, 0.0)
        yaw: tuple[float, float] = (0.0, 0.0)

    class_type: type = WorldFramePoseCommand

    asset_name: str = "robot"
    body_name: str = MISSING
    """Robot body whose pose is tracked (e.g. left_wrist_yaw_link)."""

    anchor_xy: tuple[float, float] = (0.0, 0.0)
    """XY anchor (env-local world) around which the spherical sampling is
    centered. For per-arm asymmetry, set anchor_xy = (0, +0.2) for the left
    arm and (0, -0.2) for the right."""
    anchor_z: float = 0.94
    """Z anchor (env-local world). ~pelvis spawn height."""

    ranges: Ranges = MISSING

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = _WORLD_GOAL_MARKER_CFG.replace(
        prim_path="/Visuals/Command/world_ee_goal"
    )


# ---------------------------------------------------------------------------
# Bimanual coupled world-frame EE command
# ---------------------------------------------------------------------------
class BimanualWorldFramePoseCommand(WorldFramePoseCommand):
    """Two-arm coupled world-frame EE pose.

    Independent per-arm sampling (`WorldFramePoseCommand`) can place the LEFT
    and RIGHT targets in arrangements wider than the robot's arm span, making
    the pair physically impossible — the policy then settles on "favor one
    arm, sacrifice the other" and one error stays stuck.

    This class samples ONE world "task center" per episode (master only) and
    places each arm's target as a small Cartesian offset from that center.
    Max L-to-R distance is bounded by `2 * max_offset`, so every pair is
    reachable.

    Master vs slave:
      * cfg.linked_command_name = None    -> MASTER. Samples the center
        spherically from (anchor_xy, anchor_z) using `cfg.ranges.r/theta/phi`
        (same convention as the parent class), stores it on
        `self.episode_center_w`. Curriculum widens master.r_max.
      * cfg.linked_command_name = "<other-bimanual-term-name>" -> SLAVE.
        Reads the master's episode_center_w each resample; its own
        `cfg.ranges.r/theta/phi` are ignored.

    Both master and slave then sample their per-arm offset from
    `cfg.arm_offset_ranges` (Cartesian, env-local world axes) and write
    `pose_command_world = center + arm_offset`.

    Resampling order: the CommandManager processes terms in cfg-declaration
    order. Make sure the master is declared BEFORE the slave so the slave
    sees a fresh center on the same reset tick.
    """

    cfg: "BimanualWorldFramePoseCommandCfg"

    def __init__(self, cfg: "BimanualWorldFramePoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        # Shared center per env. Only the MASTER writes to this; the SLAVE
        # reads from the master's instance. Sized on every term so attribute
        # lookups don't crash if accessed off-master.
        self.episode_center_w = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def is_master(self) -> bool:
        return self.cfg.linked_command_name is None

    def _read_center(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Return the (N_env_ids, 3) shared center for these envs."""
        if self.is_master:
            return self.episode_center_w[env_ids]
        linked = self._env.command_manager.get_term(self.cfg.linked_command_name)
        if not hasattr(linked, "episode_center_w"):
            raise RuntimeError(
                f"linked_command_name='{self.cfg.linked_command_name}' is not a "
                f"BimanualWorldFramePoseCommand (no episode_center_w attribute)."
            )
        return linked.episode_center_w[env_ids]

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        device = self.device
        cfg = self.cfg

        # --- MASTER: sample shared center spherically around anchor ---------
        if self.is_master:
            r = torch.empty(n, device=device).uniform_(self._r_min, self._r_max)
            theta = torch.empty(n, device=device).uniform_(*cfg.ranges.theta)
            phi = torch.empty(n, device=device).uniform_(*cfg.ranges.phi)
            ax, ay = cfg.anchor_xy
            az = cfg.anchor_z
            cos_phi = torch.cos(phi)
            self.episode_center_w[env_ids, 0] = ax + r * torch.cos(theta) * cos_phi
            self.episode_center_w[env_ids, 1] = ay + r * torch.sin(theta) * cos_phi
            self.episode_center_w[env_ids, 2] = az + r * torch.sin(phi)

        # --- BOTH: per-arm offset (Cartesian box around shared center) -----
        center = self._read_center(env_ids)  # (n, 3)
        ox = torch.empty(n, device=device).uniform_(*cfg.arm_offset_ranges.x)
        oy = torch.empty(n, device=device).uniform_(*cfg.arm_offset_ranges.y)
        oz = torch.empty(n, device=device).uniform_(*cfg.arm_offset_ranges.z)
        self.pose_command_world[env_ids, 0] = center[:, 0] + ox
        self.pose_command_world[env_ids, 1] = center[:, 1] + oy
        self.pose_command_world[env_ids, 2] = center[:, 2] + oz

        # --- Orientation: uniform euler (independent per arm — small range) -
        euler = torch.empty(n, 3, device=device)
        euler[:, 0].uniform_(*cfg.ranges.roll)
        euler[:, 1].uniform_(*cfg.ranges.pitch)
        euler[:, 2].uniform_(*cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self.pose_command_world[env_ids, 3:7] = quat_unique(quat)

    def set_distance_range(self, r_min: float, r_max: float):
        """Curriculum hook. Only meaningful on the MASTER (slave reads center
        from master), so the slave's call is a no-op."""
        if self.is_master:
            super().set_distance_range(r_min, r_max)


@configclass
class BimanualWorldFramePoseCommandCfg(WorldFramePoseCommandCfg):
    """Cfg for `BimanualWorldFramePoseCommand`.

    See the class docstring for master/slave semantics. Both LEFT and RIGHT
    instances use this cfg; the only required difference is
    `linked_command_name` (None on master, the master's name on slave) and the
    per-axis `arm_offset_ranges.y` sign (LEFT positive, RIGHT negative).
    """

    @configclass
    class ArmOffsetRanges:
        """Per-axis offset from the shared episode center (env-local world m)."""
        x: tuple[float, float] = (-0.15, 0.15)
        y: tuple[float, float] = MISSING  # asymmetric per arm
        z: tuple[float, float] = (-0.15, 0.15)

    class_type: type = BimanualWorldFramePoseCommand

    linked_command_name: str | None = None
    """If None this term is the MASTER and samples a fresh shared center each
    episode. If set to another `BimanualWorldFramePoseCommand`'s name, this
    term is the SLAVE and reads the center from that term (its own r/theta/phi
    ranges are ignored)."""

    arm_offset_ranges: ArmOffsetRanges = MISSING
    """Per-axis offset from the shared center. Bounds (per-axis spread) define
    the per-arm reachable region around the task center. LEFT.y_range should
    be positive (e.g. (+0.05, +0.30)); RIGHT.y_range negative. Max L-R
    distance ≤ 2 * max(|offset|)."""


def world_ee_position_command_error(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 (absolute) error between EE world position and the world-frame command.

    Mirrors `reach_mdp.position_command_error` but for `WorldFramePoseCommand`
    (env-local world coords, converted to absolute world by adding env_origins).
    Use with a NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    body_idx = body_ids[0] if not isinstance(body_ids, int) else body_ids
    ee_pos_w = asset.data.body_pos_w[:, body_idx]
    cmd = env.command_manager.get_command(command_name)
    target_pos_abs = cmd[:, :3] + env.scene.env_origins
    return torch.norm(ee_pos_w - target_pos_abs, dim=-1)


def world_ee_position_command_error_tanh(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fine-grain reward `1 - tanh(err / std)` for tight world-EE tracking.

    Positive — pair with a positive weight. Saturates as err → 0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    body_idx = body_ids[0] if not isinstance(body_ids, int) else body_ids
    ee_pos_w = asset.data.body_pos_w[:, body_idx]
    cmd = env.command_manager.get_command(command_name)
    target_pos_abs = cmd[:, :3] + env.scene.env_origins
    err = torch.norm(ee_pos_w - target_pos_abs, dim=-1)
    return 1.0 - torch.tanh(err / std)


def world_ee_orientation_command_error(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quaternion angular distance between EE orient (world) and command orient.

    Returns 2·acos(|w_rel|). Use NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    body_idx = body_ids[0] if not isinstance(body_ids, int) else body_ids
    ee_quat = asset.data.body_quat_w[:, body_idx]
    cmd = env.command_manager.get_command(command_name)
    q_target = cmd[:, 3:7]
    q_conj = q_target.clone()
    q_conj[:, 1:] = -q_conj[:, 1:]
    q_rel = quat_mul(q_conj, ee_quat)
    return 2.0 * torch.acos(torch.clamp(torch.abs(q_rel[:, 0]), max=1.0))


def navigation_progress_reward(
    env: "ManagerBasedRLEnv",
    command_names: Sequence[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward proportional to how much closer the pelvis got to the EE target this step.

    For each named world EE command, computes the pelvis-XY → target-XY distance
    and rewards (prev_dist - curr_dist). Summed across both arms so the policy
    learns to navigate toward whichever target reduces total distance faster
    (or both, when they sit on opposite sides).

    State persists on the env via `env._nav_progress_prev` (dict keyed by
    command_name → (N,) tensor). Cleared lazily on env reset by checking the
    reset_buf — if a env reset just fired, its prev distance is forced to the
    current distance (so the first post-reset step gets 0 reward, not a giant
    discontinuity).

    Positive reward when closing, negative when moving away — pair with a
    small POSITIVE weight (~1-5).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pelvis_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    if not hasattr(env, "_nav_progress_prev"):
        env._nav_progress_prev = {}

    reset_mask = env.reset_buf.bool() if hasattr(env, "reset_buf") else None

    total = torch.zeros(env.num_envs, device=env.device)
    for name in command_names:
        cmd = env.command_manager.get_command(name)
        target_xy = cmd[:, :2]
        curr_dist = torch.norm(pelvis_xy - target_xy, dim=-1)

        prev_dist = env._nav_progress_prev.get(name)
        if prev_dist is None:
            prev_dist = curr_dist.clone()
        elif reset_mask is not None and reset_mask.any():
            prev_dist = torch.where(reset_mask, curr_dist, prev_dist)

        total = total + (prev_dist - curr_dist)
        env._nav_progress_prev[name] = curr_dist.detach()

    return total


def pelvis_to_target_xy_b(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Pelvis-to-target XY in the pelvis-yaw frame, plus the planar distance.

    Returns (N, 3): (Δx_b, Δy_b, ‖xy‖). Gives the actor an explicit navigation
    signal even before pelvis-yaw has aligned with the bearing. The KMP itself
    only sees the body-frame EE pose post-transform, which collapses when the
    target is far (saturates at full-extension arm); this obs term keeps a
    clean "go that way" signal regardless of arm saturation.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pelvis_pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins
    pelvis_quat_yaw = yaw_quat(asset.data.root_quat_w)

    cmd = env.command_manager.get_command(command_name)
    target_pos = cmd[:, :3]

    delta_w = torch.zeros_like(pelvis_pos)
    delta_w[:, :2] = target_pos[:, :2] - pelvis_pos[:, :2]
    # Express in pelvis-yaw frame so "ahead" is +x for both arms.
    delta_b = quat_apply_inverse(pelvis_quat_yaw, delta_w)
    dist = torch.norm(delta_b[:, :2], dim=-1, keepdim=True)
    return torch.cat([delta_b[:, :2], dist], dim=-1)


def _min_pelvis_to_target_xy(
    env: "ManagerBasedRLEnv",
    command_names: Sequence[str],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Min over `command_names` of ‖pelvis_xy − target_xy‖ (env-local world).

    Used as the "do I need to walk?" signal for V5's gait/stand-still gating
    once `base_velocity` is removed. Returns shape (N,)."""
    asset: Articulation = env.scene[asset_cfg.name]
    pelvis_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    dists = []
    for name in command_names:
        cmd = env.command_manager.get_command(name)
        d = torch.norm(pelvis_xy - cmd[:, :2], dim=-1)
        dists.append(d)
    return torch.stack(dists, dim=-1).min(dim=-1).values


def feet_air_time_world_ee_positive_biped(
    env: "ManagerBasedRLEnv",
    command_names: Sequence[str],
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    distance_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """`feet_air_time_positive_biped` but gated on EE distance, not velocity cmd.

    Fires only when the closer EE target is farther than `distance_threshold`
    from the pelvis (i.e. the robot needs to walk to reach it). When the
    target is within reach, no air-time bonus → robot is free to stand and
    let the arm do the work.

    Same biped single-stance kernel as `mdp.feet_air_time_positive_biped`.
    Use with POSITIVE weight.
    """
    # Lazy import to avoid a top-of-file circular dep with isaaclab.sensors
    from isaaclab.sensors import ContactSensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # Walking-required gate: closer target must be farther than `distance_threshold`.
    walk_required = _min_pelvis_to_target_xy(env, command_names, asset_cfg) > distance_threshold
    reward = reward * walk_required.float()
    return reward


def stand_still_world_ee_joint_deviation_l1(
    env: "ManagerBasedRLEnv",
    command_names: Sequence[str],
    distance_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """`stand_still_joint_deviation_l1` gated on EE distance, not velocity cmd.

    Penalizes joint deviation from default when the closer EE target is
    WITHIN `distance_threshold` of the pelvis (i.e. walking is not needed).
    Pushes the policy to a clean stand when the target is reachable from
    where it already is.

    Use NEGATIVE weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_dev = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=-1,
    )
    stand_mode = _min_pelvis_to_target_xy(env, command_names, asset_cfg) < distance_threshold
    return joint_dev * stand_mode.float()


def facing_target_world_ee(
    env: "ManagerBasedRLEnv",
    command_names: Sequence[str],
    distance_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward pelvis-yaw alignment with the bearing to the closer EE target.

    Returns the x-component of the bearing vector in pelvis-yaw frame —
    i.e. cos(angle between pelvis-forward and target-bearing). Range [-1, 1].
    Positive when target is ahead of the robot.

    Gated to fire only when walking is required (closer target farther than
    `distance_threshold`). Without this gate, the term would also pull the
    robot to rotate toward a target that's already reachable from a standing
    pose — wasted motion.

    Pair with a small POSITIVE weight (~0.5–1.0). This is the "turn before
    you walk" signal — gives the navigation gradient a heading-derivative
    even before the robot starts translating.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pelvis_pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins
    pelvis_yaw_q = yaw_quat(asset.data.root_quat_w)

    pelvis_xy = pelvis_pos[:, :2]
    # Pick the closer target's bearing.
    candidates = []
    for name in command_names:
        cmd = env.command_manager.get_command(name)
        delta_xy = cmd[:, :2] - pelvis_xy
        d = torch.norm(delta_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        candidates.append((delta_xy / d, d.squeeze(-1)))
    # Closer one per env.
    bearings = torch.stack([c[0] for c in candidates], dim=1)   # (N, K, 2)
    dists = torch.stack([c[1] for c in candidates], dim=1)      # (N, K)
    idx = dists.argmin(dim=1, keepdim=True)                     # (N, 1)
    bearing_xy = torch.gather(bearings, 1, idx.unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)  # (N, 2)

    # Pad to 3D for quat_apply_inverse, then take x-component in pelvis-yaw frame.
    bearing_w = torch.zeros(env.num_envs, 3, device=env.device)
    bearing_w[:, :2] = bearing_xy
    bearing_b = quat_apply_inverse(pelvis_yaw_q, bearing_w)
    cos_angle = bearing_b[:, 0]

    walk_required = dists.min(dim=-1).values > distance_threshold
    return cos_angle * walk_required.float()


def modify_world_ee_distance_cap(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_names: Sequence[str],
    r_min: float,
    r_max: float,
    num_steps: int,
):
    """Curriculum hook — bump each command's distance cap at `num_steps`.

    Mirrors `mdp.modify_reward_weight`'s "after N steps" semantics. Once the
    env's `common_step_counter` ≥ num_steps, the bound is widened (no-op
    afterward — single fire). Each fire also calls `set_distance_range` on the
    affected command terms, which take effect on the next reset.

    The curriculum framework normally returns a value; here we return 0 because
    the side effect is a method call, not a config field swap.
    """
    if env.common_step_counter < num_steps:
        return 0.0
    flag_key = f"_world_ee_curr_done_{r_max:.3f}"
    if getattr(env, flag_key, False):
        return 0.0
    for name in command_names:
        term = env.command_manager.get_term(name)
        if hasattr(term, "set_distance_range"):
            term.set_distance_range(r_min, r_max)
    setattr(env, flag_key, True)
    return 1.0


# =============================================================================
# V5-H additions — Stage 2 (hierarchical world-frame commander).
# Reward terms: workspace distance, base heading alignment, world EE tracking
# with active-arm mask. Curriculum: success-rate-gated distance expansion
# (paper Eq. 17). Mask command term: constant [1,1] for now (both arms always
# active); supports the paper's [1,0] / [0,1] / [1,1] variants if extended.
# =============================================================================


class ConstantVectorCommand(CommandTerm):
    """Per-env constant vector command (no sampling, no metric).

    Used in V5-H as the binary "active-arm mask" placeholder
    (`mask = [1.0, 1.0]`). All envs see the same constant value; resampling
    is a no-op. Cheap way to plug a static value into the obs+reward chain
    without writing a custom obs term.
    """

    cfg: "ConstantVectorCommandCfg"

    def __init__(self, cfg: "ConstantVectorCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        value = torch.tensor(cfg.value, dtype=torch.float32, device=self.device)
        self._command = value.unsqueeze(0).repeat(self.num_envs, 1)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@configclass
class ConstantVectorCommandCfg(CommandTermCfg):
    class_type: type = ConstantVectorCommand
    value: tuple[float, ...] = MISSING
    """The constant value broadcast to all envs."""


def workspace_distance_reward(
    env: "ManagerBasedRLEnv",
    command_names: Sequence[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize planar distance from pelvis XY to the closer EE target XY.

    Paper §V.C "Workspace Optimization": penalize the planar distance between
    the robot base and the active end-effector targets. We take the MIN over
    arms so the policy rewards moving close to whichever arm needs to reach.

    Use NEGATIVE weight.
    """
    return _min_pelvis_to_target_xy(env, command_names, asset_cfg)


def base_heading_alignment_reward(
    env: "ManagerBasedRLEnv",
    command_names: Sequence[str],
    distance_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward base-heading × bearing-to-target alignment (paper §V.C).

    Computes cos(angle) between pelvis-forward and the unit bearing to the
    nearest EE target. Gated to fire only when walking is required (target
    farther than `distance_threshold`) so it does not pull the robot to
    rotate when the target is already reachable.

    Use POSITIVE weight. Same kernel as `facing_target_world_ee` but named
    after the paper's term. Kept separate for clarity in the reward log.
    """
    return facing_target_world_ee(env, command_names, distance_threshold, asset_cfg)


def world_ee_position_error_masked(
    env: "ManagerBasedRLEnv",
    command_name: str,
    mask_command_name: str,
    mask_slot: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 EE-vs-target world error, multiplied by the mask slot for this arm.

    `mask_command_name` is a `ConstantVectorCommand` (or any command term)
    whose value at slot `mask_slot` is 0/1 = "this arm is inactive/active".
    For V5-H first run we wire mask = [1, 1] so this reduces to the plain L2,
    but the masking surface lets a future cfg train single-arm tasks too.

    Use NEGATIVE weight.
    """
    err = world_ee_position_command_error(env, command_name, asset_cfg)
    mask = env.command_manager.get_command(mask_command_name)[:, mask_slot]
    return err * mask


def world_ee_position_error_masked_tanh(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float,
    mask_command_name: str,
    mask_slot: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """`1 - tanh(err/std)` × mask. Pair with POSITIVE weight."""
    fine = world_ee_position_command_error_tanh(env, command_name, std, asset_cfg)
    mask = env.command_manager.get_command(mask_command_name)[:, mask_slot]
    return fine * mask


def v4_last_action_obs(
    env: "ManagerBasedRLEnv",
    action_term_name: str = "joint_pos",
) -> torch.Tensor:
    """Return the V4 (Stage 1) last residual as the "actions" obs for V4.

    In V5-H the env's action manager stores the Stage 2 19-D action, not the
    V4 28-D residual. But V4 was trained with `last_action` = V4 residual
    (28-D). Feeding it the Stage 2 action would shift its input distribution
    catastrophically (wrong dim → outright shape mismatch in the MLP).

    The Stage 2 action class (`Stage2WrappedAction`) keeps `_v4_residual`
    updated each step. This obs term reads it for V4's `actions` slot,
    preserving V4's exact input contract.
    """
    term = env.action_manager.get_term(action_term_name)
    return term.last_v4_residual


def world_base_pose_obs(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Pelvis pose in env-local world (7-D: x, y, z, qw, qx, qy, qz).

    Stage 2 observation per paper Eq. 13: `s_t^H = [s_t, w T_b, c_t]`.
    The `w T_b` here is the env-local-world pelvis pose so the Stage 2 policy
    knows where the robot is relative to the world-frame EE targets.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :3] - env.scene.env_origins
    quat = asset.data.root_quat_w
    return torch.cat([pos, quat], dim=-1)


def modify_world_ee_distance_cap_on_success(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_names: Sequence[str],
    error_metric_name: str,
    error_threshold: float,
    delta_r: float,
    r_max_ceiling: float,
    check_every_n_steps: int = 24,
    min_envs_below_threshold_frac: float = 0.5,
) -> float:
    """Success-rate-gated curriculum (HiWET paper Eq. 17).

    Each fire (`check_every_n_steps` steps), inspect the running EE position
    error reported by each command term's `metrics[error_metric_name]`.
    If the fraction of envs with `error < error_threshold` exceeds
    `min_envs_below_threshold_frac`, expand the radius by `delta_r` (capped
    at `r_max_ceiling`).

    Returns the current `r_max` (for tensorboard logging).

    Notes:
      * State persists on `env._curr_world_ee_r_max`. Both command terms
        share the same r_max — the policy is asked to handle symmetric
        ranges per arm.
      * `error_metric_name` is whatever the WorldFramePoseCommand publishes;
        in our impl that's "position_error".
    """
    # Guard: skip when no real sim steps have run yet (metric buffers are
    # still zero-initialized -> they would falsely pass `< error_threshold`
    # and trigger an immediate spurious widening on the very first call).
    if env.common_step_counter < check_every_n_steps:
        return float(getattr(env, "_curr_world_ee_r_max", 0.5))

    if env.common_step_counter % check_every_n_steps != 0:
        return float(getattr(env, "_curr_world_ee_r_max", 0.5))

    cm = env.command_manager
    err_total = None
    n_envs = env.num_envs
    for name in command_names:
        term = cm.get_term(name)
        err = term.metrics.get(error_metric_name)
        if err is None:
            return float(getattr(env, "_curr_world_ee_r_max", 0.5))
        err_total = err if err_total is None else err_total + err
    avg_err = err_total / len(command_names)
    # Mask out envs whose metric is exactly 0 — they almost certainly just
    # reset and haven't had a metric update yet. Counting them as success
    # would also trigger a spurious widening.
    valid = avg_err > 1e-6
    if valid.float().mean().item() < 0.5:
        # >half the envs are in "just reset" state — defer the check.
        return float(getattr(env, "_curr_world_ee_r_max", 0.5))
    frac_success = ((avg_err < error_threshold) & valid).float().sum().item() / valid.float().sum().item()

    r_max = getattr(env, "_curr_world_ee_r_max", None)
    if r_max is None:
        # Initialize from one of the command terms' current bounds.
        r_max = float(cm.get_term(command_names[0]).distance_range[1])
    if frac_success >= min_envs_below_threshold_frac and r_max < r_max_ceiling:
        r_max = min(r_max + delta_r, r_max_ceiling)
        for name in command_names:
            term = cm.get_term(name)
            if hasattr(term, "set_distance_range"):
                term.set_distance_range(term.distance_range[0], r_max)
    env._curr_world_ee_r_max = r_max
    return float(r_max)


def randomize_arm_joint_targets(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample joint position targets uniformly and write them as PD targets.

    Used at reset to pin the upper body (waist / arms / neck) to a default pose
    since the policy doesn't act on those joints.
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
            targets[:, i].uniform_(lo, hi)
        else:
            targets[:, i] = asset.data.joint_pos_target[env_ids, joint_ids_list[i]]

    asset.set_joint_position_target(targets, joint_ids=joint_ids_list, env_ids=env_ids)
