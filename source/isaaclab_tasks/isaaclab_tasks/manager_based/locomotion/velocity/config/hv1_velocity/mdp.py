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
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

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
