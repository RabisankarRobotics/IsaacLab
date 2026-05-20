"""Custom MDP terms for the HV1.2 standing task.

* `randomize_arm_joint_targets` — event term that samples a per-joint target
  uniformly from `position_range` and writes it into the articulation's PD
  target buffer. Used to pin waist/arms/head at fixed (or random) targets at
  reset and on intervals.
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
    """Sample joint position targets uniformly and write them as PD targets.

    `position_range` is keyed by joint name (URDF name, no env prefix) and maps
    to (lo, hi) bounds. Joints in `asset_cfg.joint_ids` but not in the dict
    keep their existing target.
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
