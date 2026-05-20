"""Custom MDP terms for the H1_2 standing task.

* `randomize_arm_joint_targets` — event term that samples a per-joint arm
  target uniformly from `position_range` and writes it into the articulation's
  PD target buffer. Used in mode="reset" and mode="interval".
* `arm_target_delta` — observation term returning the current arm PD target
  minus the default arm pose, so the policy can compensate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_arm_joint_targets(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample arm joint position targets uniformly and apply them as PD targets.

    `position_range` is keyed by joint name (URDF name, no env prefix) and maps
    to (lo, hi) bounds. Joints not in the dict keep their existing target.
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
    """Observation: current arm PD target minus default arm pose, shape (N, |joints|)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    return asset.data.joint_pos_target[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
