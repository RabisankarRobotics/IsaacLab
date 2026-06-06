"""KMP-residual joint position action for HV1 V4 loco-manipulation.

Implements HiWET paper Eq. 11:

    q_target = q_prior(commands) + residual * residual_scale

where `q_prior` is a frozen MLP (KMP) mapping the 16-D Stage-1 command vector
(body_height, left_wrist_pose, right_wrist_pose, alpha_t) to a 28-D feasible
joint posture trained offline on staged IK over the HV1 URDF.

Plugged into the env in place of the standard JointPositionActionCfg. The PD
controller downstream is unchanged — only the input it receives differs.

See scripts/hv1/kmp/ for KMP training/validation. The saved checkpoint
`deploy/model/kmp/kmp_v1.pt` is loaded once at env init and never updated.
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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Make KMP class importable. The model file lives in scripts/hv1/kmp/
_KMP_SCRIPTS_DIR = "/home/rabisankar/IsaacLab/scripts/hv1/kmp"
if _KMP_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _KMP_SCRIPTS_DIR)
from kmp_model import KMP  # noqa: E402
# KMP_OUTPUT_ORDER is the 28-name list the KMP MLP was trained to predict.
# Defined in generate_kmp_dataset.py as ACTUATED_28.
from generate_kmp_dataset import ACTUATED_28 as KMP_OUTPUT_ORDER  # noqa: E402


# Hardcoded command-name → command-slot mapping. These names match what
# loco_manip_v3_env_cfg.py declares; V4 inherits the same command terms.
_BODY_HEIGHT_CMD = "body_height"
_LEFT_EE_CMD = "left_ee_pose"
_RIGHT_EE_CMD = "right_ee_pose"
_ALPHA_CMD = "waist_regularization"


class KMPResidualJointPositionAction(JointPositionAction):
    """Joint position action where the offset is `KMP(current_commands)`.

    Each call to `process_actions`:
      1. Look up the four command terms from `env.command_manager`.
      2. Pack them into the 16-D layout the KMP expects.
      3. Forward through the frozen KMP -> `q_prior` shape (B, 28).
      4. `processed_actions = q_prior + raw_actions * residual_scale`.

    `apply_actions` then dispatches `set_joint_position_target` exactly like
    the parent class — the PD layer is identical to V3, only the target shifts.
    """

    cfg: "KMPResidualJointPositionActionCfg"

    def __init__(self, cfg: "KMPResidualJointPositionActionCfg", env: "ManagerBasedEnv"):
        # Force use_default_offset off — we overwrite the offset each step.
        cfg.use_default_offset = False
        super().__init__(cfg, env)

        # Scale resolution:
        #   - If `cfg.residual_scale` is set (scalar), use it uniformly and
        #     override the parent's _scale.
        #   - Else trust whatever the parent built from `cfg.scale` (works for
        #     both float and dict — dict becomes a per-joint (B, 28) tensor).
        if cfg.residual_scale is not None:
            self._scale = float(cfg.residual_scale)

        # Load the frozen KMP and move it to the env device.
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
                "Joint name list mismatch — check cfg.joint_names matches "
                "LEG_JOINTS + ARM_JOINT_NAMES + WAIST_ACTUATED_JOINTS."
            )

        # Joint-order remap: KMP_OUTPUT_ORDER is the order the KMP MLP was
        # trained to predict. The action term's `_joint_names` is whatever
        # order the asset resolver returned (regex matches in URDF order,
        # not the explicit list order from V3 cfg). We build a permutation
        # so `q_prior_remapped[:, action_slot] = q_prior_kmp[:, kmp_slot]`.
        # Use scatter to copy KMP-order values into action-term order.
        missing = [n for n in KMP_OUTPUT_ORDER if n not in self._joint_names]
        if missing:
            raise RuntimeError(
                f"KMP joints not in action term resolved names: {missing}. "
                "Check cfg.joint_names covers all 28 actuated joints."
            )
        # For each KMP output position i, this is the action-vector slot it
        # belongs in.
        self._kmp_to_action = torch.tensor(
            [self._joint_names.index(n) for n in KMP_OUTPUT_ORDER],
            dtype=torch.long, device=self.device,
        )

        # Scratch buffer for the 16-D command vector (avoids per-step alloc).
        self._cmd_buf = torch.zeros(self.num_envs, 16, device=self.device)

        # Cache the last q_prior so observers / loggers can read it.
        self._last_q_prior = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor) -> None:
        # Store raw actor output for r_kmp reward & action_rate computation.
        self._raw_actions[:] = actions

        # Pull current commands. Quat layout in IsaacLab pose commands is
        # [x, y, z, qw, qx, qy, qz] (w first). The KMP was trained with
        # quaternions in xyzw order (scipy as_quat()), so we reorder here.
        cm = self._env.command_manager
        h = cm.get_command(_BODY_HEIGHT_CMD)              # (B, 1)
        lpose = cm.get_command(_LEFT_EE_CMD)              # (B, 7)
        rpose = cm.get_command(_RIGHT_EE_CMD)             # (B, 7)
        alpha = cm.get_command(_ALPHA_CMD)                # (B, 1)

        buf = self._cmd_buf
        buf[:, 0:1] = h
        buf[:, 1:4] = lpose[:, 0:3]
        buf[:, 4:7] = lpose[:, 4:7]   # qx, qy, qz
        buf[:, 7:8] = lpose[:, 3:4]   # qw
        buf[:, 8:11] = rpose[:, 0:3]
        buf[:, 11:14] = rpose[:, 4:7]
        buf[:, 14:15] = rpose[:, 3:4]
        buf[:, 15:16] = alpha

        with torch.no_grad():
            q_prior_kmp = self._kmp(buf)                   # (B, 28) in KMP order
        # Reorder KMP output into the action term's slot order. Index_select
        # creates a new tensor; that's fine — fwd is small and infrequent.
        # q_prior_action[:, s] = q_prior_kmp[:, kmp_idx_for_s]
        # Equivalent: scatter q_prior_kmp into action slots using kmp_to_action.
        q_prior_action = torch.empty_like(q_prior_kmp)
        q_prior_action[:, self._kmp_to_action] = q_prior_kmp
        self._last_q_prior = q_prior_action

        # Residual on top of the kinematic prior.
        self._processed_actions = q_prior_action + self._raw_actions * self._scale

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    # apply_actions is inherited unchanged: set_joint_position_target.

    @property
    def last_q_prior(self) -> torch.Tensor:
        """Most recent KMP output (B, 28). Shape matches raw_actions."""
        return self._last_q_prior

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._last_q_prior.zero_()
        else:
            self._last_q_prior[env_ids] = 0.0


@configclass
class KMPResidualJointPositionActionCfg(JointPositionActionCfg):
    """Config for KMP-residual joint position action.

    Behaves like a plain JointPositionActionCfg with one twist: the per-step
    offset is computed live as `KMP(current_commands)` instead of being a
    constant default-pose vector. `scale` becomes the residual scale.

    Fields inherited from JointPositionActionCfg:
      - asset_name, joint_names, preserve_order, clip
      - scale: USED HERE AS THE RESIDUAL SCALE (paper uses 0.15)
      - offset: NOT USED — overwritten per-step by KMP output.
      - use_default_offset: NOT USED — forced False in __init__.

    New fields:
      - kmp_checkpoint: absolute path to the trained KMP .pt
      - residual_scale: alias for `scale` made explicit. If set, overrides
                        `scale`. Default 0.15 per HiWET paper.
    """

    class_type: type = KMPResidualJointPositionAction

    kmp_checkpoint: str = MISSING
    """Path to the frozen KMP MLP checkpoint (kmp_v1.pt)."""

    residual_scale: float | None = None
    """Scale applied to the actor's residual before adding to q_prior."""

    use_default_offset: bool = False
    """Forced False — KMP supplies the per-step offset."""
