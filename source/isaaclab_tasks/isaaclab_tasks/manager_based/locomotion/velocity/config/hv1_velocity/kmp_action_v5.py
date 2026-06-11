"""KMP-residual joint position action for HV1 V5 world-frame loco-manipulation.

V5 difference vs V4 (`kmp_action.py`):
  EE pose commands are sampled in WORLD frame (env-local world, i.e. relative
  to the env spawn origin) and FIXED per episode. The KMP MLP however was
  trained on BODY-frame EE inputs and cannot be retrained cheaply. So we do
  the world → body conversion at runtime inside `process_actions` using the
  current pelvis pose, then feed the KMP the same body-frame 16-D vector it
  expects.

Coupling effect:
  * V4 (body-frame, resampled every ~4s): EE target moves with the pelvis,
    so the actor can converge to "stand still in KMP pose and the EE is
    already at target." Gait and reach are decoupled. This is the V4
    standing-pit failure mode.
  * V5 (world-frame, episode-static): pelvis must navigate to within reach
    AND orient the arm; the KMP outputs a body-frame arm pose, but the
    body-frame target itself shifts every step as the pelvis moves. The
    only way to drive the body-frame EE error to zero is to walk to where
    the world-frame target sits. Gait and reach are mechanically coupled.

Reused from V4 (no change):
  * KMP MLP checkpoint (`kmp_v1.pt`).
  * 28-D KMP output → action slot remap.
  * Per-joint residual scale handling.
  * `r_kmp` reward computation (reads `self._raw_actions`).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_KMP_SCRIPTS_DIR = "/home/rabisankar/IsaacLab/scripts/hv1/kmp"
if _KMP_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _KMP_SCRIPTS_DIR)
from kmp_model import KMP  # noqa: E402
from generate_kmp_dataset import ACTUATED_28 as KMP_OUTPUT_ORDER  # noqa: E402


# Command-name slots. V5 renames the EE commands to make the world-frame
# convention explicit; everything else carries over from V4.
_BODY_HEIGHT_CMD = "body_height"
_LEFT_EE_CMD = "world_left_ee_pose"
_RIGHT_EE_CMD = "world_right_ee_pose"
_ALPHA_CMD = "waist_regularization"


class KMPResidualJointPositionActionV5(JointPositionAction):
    """V5 KMP-residual action: transforms world EE → body EE, then runs the KMP.

    Per step:
      1. Read pelvis pose (env-local world) and the two world-frame EE targets.
      2. `subtract_frame_transforms` to get EE pose expressed in pelvis frame.
      3. Pack into the 16-D KMP input (body_height, left_ee_body, right_ee_body,
         alpha) and forward the frozen KMP.
      4. `q_target = q_prior + residual * scale`, identical to V4 from here on.

    The KMP MLP is unchanged from V4 — the body-frame contract it learned is
    still respected; only the source of the body-frame target has shifted from
    "directly sampled command" to "world target − current pelvis."
    """

    cfg: "KMPResidualJointPositionActionV5Cfg"

    def __init__(self, cfg: "KMPResidualJointPositionActionV5Cfg", env: "ManagerBasedEnv"):
        cfg.use_default_offset = False
        super().__init__(cfg, env)

        if cfg.residual_scale is not None:
            self._scale = float(cfg.residual_scale)

        if not os.path.isfile(cfg.kmp_checkpoint):
            raise FileNotFoundError(
                f"KMP checkpoint not found: {cfg.kmp_checkpoint}. "
                "Train one with scripts/hv1/kmp/train_kmp.py first."
            )
        self._kmp = KMP.load(cfg.kmp_checkpoint, map_location=str(self.device))
        self._kmp = self._kmp.to(self.device)
        self._kmp.eval()
        for p in self._kmp.parameters():
            p.requires_grad_(False)

        if self._num_joints != 28:
            raise RuntimeError(
                f"KMP outputs 28 joints; action term resolved {self._num_joints}. "
                "Check cfg.joint_names matches the 28-joint actuated list."
            )

        missing = [n for n in KMP_OUTPUT_ORDER if n not in self._joint_names]
        if missing:
            raise RuntimeError(
                f"KMP joints not in action term resolved names: {missing}."
            )
        self._kmp_to_action = torch.tensor(
            [self._joint_names.index(n) for n in KMP_OUTPUT_ORDER],
            dtype=torch.long, device=self.device,
        )

        self._cmd_buf = torch.zeros(self.num_envs, 16, device=self.device)
        self._last_q_prior = torch.zeros_like(self._raw_actions)

        # Cache env_origins handle so we can convert pelvis world → env-local
        # world without re-fetching per step. env_origins is a static tensor
        # set up during scene construction; safe to bind once.
        self._env_origins = env.scene.env_origins  # (num_envs, 3)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

        # --- Pelvis pose in env-local world frame -------------------------
        # `root_link_pose_w` is absolute simulator world. Subtract env_origins
        # so the position lives in the same env-local world the command terms
        # sample in. Orientation needs no shift (env origins are translations
        # only in IsaacLab's default scene layout).
        root_pose = self._asset.data.root_link_pose_w        # (B, 7) wxyz
        pelvis_pos = root_pose[:, :3] - self._env_origins    # (B, 3)
        pelvis_quat = root_pose[:, 3:7]                       # (B, 4) wxyz

        # --- Pull commands ------------------------------------------------
        cm = self._env.command_manager
        h = cm.get_command(_BODY_HEIGHT_CMD)                 # (B, 1)
        lpose_w = cm.get_command(_LEFT_EE_CMD)               # (B, 7) wxyz
        rpose_w = cm.get_command(_RIGHT_EE_CMD)              # (B, 7) wxyz
        alpha = cm.get_command(_ALPHA_CMD)                   # (B, 1)

        # --- World → body transform for both EE targets -------------------
        l_pos_b, l_quat_b = subtract_frame_transforms(
            pelvis_pos, pelvis_quat, lpose_w[:, 0:3], lpose_w[:, 3:7]
        )
        r_pos_b, r_quat_b = subtract_frame_transforms(
            pelvis_pos, pelvis_quat, rpose_w[:, 0:3], rpose_w[:, 3:7]
        )

        # --- Pack 16-D KMP input ------------------------------------------
        # KMP was trained with quaternions in xyzw order (scipy `as_quat()`),
        # but `subtract_frame_transforms` returns wxyz. Reorder here, matching
        # the V4 packer's swap.
        buf = self._cmd_buf
        buf[:, 0:1] = h
        buf[:, 1:4] = l_pos_b
        buf[:, 4:7] = l_quat_b[:, 1:4]    # qx, qy, qz
        buf[:, 7:8] = l_quat_b[:, 0:1]    # qw -> KMP xyzw tail
        buf[:, 8:11] = r_pos_b
        buf[:, 11:14] = r_quat_b[:, 1:4]
        buf[:, 14:15] = r_quat_b[:, 0:1]
        buf[:, 15:16] = alpha

        # --- KMP forward + slot remap + residual --------------------------
        with torch.no_grad():
            q_prior_kmp = self._kmp(buf)                     # (B, 28) KMP order
        q_prior_action = torch.empty_like(q_prior_kmp)
        q_prior_action[:, self._kmp_to_action] = q_prior_kmp
        self._last_q_prior = q_prior_action

        self._processed_actions = q_prior_action + self._raw_actions * self._scale

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    @property
    def last_q_prior(self) -> torch.Tensor:
        """Most recent KMP output (B, 28)."""
        return self._last_q_prior

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._last_q_prior.zero_()
        else:
            self._last_q_prior[env_ids] = 0.0


@configclass
class KMPResidualJointPositionActionV5Cfg(JointPositionActionCfg):
    """Config for V5 world-frame KMP-residual action.

    Same fields as the V4 cfg (`KMPResidualJointPositionActionCfg`). The only
    behavioral difference lives in the action class's `process_actions`, which
    expects the command terms `world_left_ee_pose` and `world_right_ee_pose`
    instead of V4's body-frame `left_ee_pose` / `right_ee_pose`.
    """

    class_type: type = KMPResidualJointPositionActionV5

    kmp_checkpoint: str = MISSING
    """Path to the frozen KMP MLP checkpoint (reuses V4's kmp_v1.pt)."""

    residual_scale: float | None = None
    """Scalar residual scale. If set, overrides `scale`. None = use per-joint dict."""

    use_default_offset: bool = False
    """Forced False — KMP supplies the per-step offset."""
