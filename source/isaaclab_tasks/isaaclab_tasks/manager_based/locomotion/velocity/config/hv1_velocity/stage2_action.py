"""HV1 V5-H Stage 2 action — frozen V4 (Stage 1) wrapped inside an action class.

Implements the HiWET paper's two-stage hierarchy. The env's "action" is the
Stage 2 command `u_t = [v_b^des(3), h^des(1), bT_L^des(7), bT_R^des(7), α_t(1)]`
(19-D). Per env step, this class:

  1. Writes `u_t` into the V4 command buffers (base_velocity, body_height,
     left_ee_pose, right_ee_pose, waist_regularization).
  2. Reads the `v4_actor` observation group from the env (an exact replica
     of V4's training-time actor obs — same PolicyCfg, history_length=5).
     There is a 1-step lag: the obs was assembled at end of previous step,
     so the commands V4 sees are the PREVIOUS Stage 2 action. Acceptable
     at Stage 2's lower control frequency.
  3. Runs the frozen JIT-compiled V4 actor MLP on that obs -> 28-D residual.
  4. Runs the frozen KMP MLP on the FRESH `u_t` to get q_prior (28-D).
  5. q_target = q_prior + residual * per_joint_scale.

V4 weights come from the JIT-exported `deploy/model/.../policy.pt`. The
in-graph obs_normalizer is included, so we feed the raw obs straight in.

Why JIT instead of reinitializing an `rsl_rl.ActorCritic`:
  * Self-contained — no dependence on the exact rsl_rl version or actor-
    critic kwargs the V4 run used.
  * Fast — torch.jit.script-compiled forward.
  * Drift-proof — bumped rsl_rl version won't silently change MLP layout.

Quaternion handling on the Stage 2 output:
  Stage 2 emits raw 4-D quat slots; PPO doesn't constrain unit norm. We
  L2-normalize at the action class boundary before forwarding into the
  command buffers (which the KMP later consumes). Zero-norm corner case
  falls back to identity quat (1, 0, 0, 0) wxyz.
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
    from isaaclab.envs import ManagerBasedRLEnv


_KMP_SCRIPTS_DIR = "/home/rabisankar/IsaacLab/scripts/hv1/kmp"
if _KMP_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _KMP_SCRIPTS_DIR)
from kmp_model import KMP  # noqa: E402
from generate_kmp_dataset import ACTUATED_28 as KMP_OUTPUT_ORDER  # noqa: E402


# Stage 2 command term names V4 will end up consuming.
_BASE_VEL_CMD = "base_velocity"
_BODY_HEIGHT_CMD = "body_height"
_LEFT_EE_CMD = "left_ee_pose"
_RIGHT_EE_CMD = "right_ee_pose"
_ALPHA_CMD = "waist_regularization"

# Stage 2 19-D action layout:
#   0:3    v_b^des    (v_x, v_y, w_z)        -> base_velocity (3-D)
#   3:4    h^des                             -> body_height (1-D)
#   4:11   bT_L^des   (x, y, z, qw, qx, qy, qz)  -> left_ee_pose (7-D, wxyz)
#   11:18  bT_R^des                          -> right_ee_pose (7-D)
#   18:19  α_t                               -> waist_regularization (1-D)
STAGE2_ACTION_DIM = 19


class Stage2WrappedAction(JointPositionAction):
    """Stage 2 action: writes commands, runs frozen V4 + KMP inline.

    The action manager treats this as a 19-D action term over 28 joints
    (joint count comes from the parent JointPositionAction over the same
    joint list V4 uses). `process_actions` does the heavy lifting; the
    parent's `apply_actions` (set_joint_position_target) is reused.
    """

    cfg: "Stage2WrappedActionCfg"

    def __init__(self, cfg: "Stage2WrappedActionCfg", env: "ManagerBasedRLEnv"):
        # Buffer sizing trick: pass scale=1.0 to the parent so it does NOT try
        # to build a (N, action_dim) scale tensor from a per-joint dict
        # (action_dim is 19 here while the dict has 28 keys). We rebuild a
        # proper (N, 28) per-joint scale tensor after super().__init__.
        cfg.use_default_offset = False
        _saved_scale = cfg.scale
        cfg.scale = 1.0
        try:
            super().__init__(cfg, env)
        finally:
            cfg.scale = _saved_scale  # restore in case the cfg is reused

        # --- buffers --------------------------------------------------------
        # Parent sized _raw_actions and _processed_actions to (N, 19). We
        # rebuild _processed_actions to (N, 28) since apply_actions writes 28
        # joint targets. _raw_actions stays (N, 19) — that's the Stage 2
        # action PPO emits.
        self._processed_actions = torch.zeros(self.num_envs, 28, device=self.device)

        # Per-joint residual scale (N, 28) built from cfg.scale dict.
        if not isinstance(_saved_scale, dict):
            raise ValueError(
                "Stage2WrappedActionCfg.scale must be a per-joint dict (V4's "
                f"_KMP_RESIDUAL_SCALE); got {type(_saved_scale)}."
            )
        self._scale = torch.ones(self.num_envs, 28, device=self.device)
        for jname, sval in _saved_scale.items():
            if jname not in self._joint_names:
                continue
            jidx = self._joint_names.index(jname)
            self._scale[:, jidx] = float(sval)

        # Convenience aliases — the Stage 2 action seen by PPO + V4 residual.
        self._stage2_action = self._raw_actions  # alias, (N, 19)
        self._v4_residual = torch.zeros(self.num_envs, 28, device=self.device)

        # --- frozen V4 actor (JIT) ------------------------------------------
        if not os.path.isfile(cfg.v4_jit_policy):
            raise FileNotFoundError(
                f"V4 JIT policy not found: {cfg.v4_jit_policy}. "
                "Run play.py with --task Isaac-Tracking-LocoManipV4-HV1-Play-v0 "
                "first to export it."
            )
        self._v4_policy = torch.jit.load(cfg.v4_jit_policy, map_location=str(self.device))
        self._v4_policy.eval()
        for p in self._v4_policy.parameters():
            p.requires_grad_(False)

        # --- frozen KMP -----------------------------------------------------
        if not os.path.isfile(cfg.kmp_checkpoint):
            raise FileNotFoundError(f"KMP checkpoint not found: {cfg.kmp_checkpoint}.")
        self._kmp = KMP.load(cfg.kmp_checkpoint, map_location=str(self.device)).to(self.device)
        self._kmp.eval()
        for p in self._kmp.parameters():
            p.requires_grad_(False)

        # --- per-joint scale -------------------------------------------------
        # `cfg.scale` is the dict from V4 (legs 0.25 / arms 0.10 / waist 0.10).
        # parent's _scale is (1, N_joints) — we reuse it directly.

        # --- joint slot remap (KMP order -> action order) -------------------
        if self._num_joints != 28:
            raise RuntimeError(f"Expected 28 joints, got {self._num_joints}.")
        missing = [n for n in KMP_OUTPUT_ORDER if n not in self._joint_names]
        if missing:
            raise RuntimeError(f"KMP joints not in action term: {missing}.")
        self._kmp_to_action = torch.tensor(
            [self._joint_names.index(n) for n in KMP_OUTPUT_ORDER],
            dtype=torch.long, device=self.device,
        )

        # --- 16-D KMP input scratch buffer ----------------------------------
        self._cmd_buf = torch.zeros(self.num_envs, 16, device=self.device)

        # --- command-term handles for write-back ----------------------------
        # Resolved lazily on first process_actions because some terms may not
        # exist yet at action manager construction time.
        self._cmd_terms_bound = False
        self._t_base_velocity = None
        self._t_body_height = None
        self._t_left_ee = None
        self._t_right_ee = None
        self._t_alpha = None

        # Observation manager handle — same lazy resolve.
        self._v4_obs_group_name = cfg.v4_obs_group_name

    # --- action_dim override ------------------------------------------------
    @property
    def action_dim(self) -> int:
        # PPO sees a 19-D action space; the action manager uses this for
        # input validation and shape of self._raw_actions.
        return STAGE2_ACTION_DIM

    # NOTE: do NOT override `raw_actions` here — the parent's property reads
    # `self._raw_actions` which is sized (N, 19) thanks to the action_dim
    # override above and is built BEFORE self._stage2_action exists. Our
    # `self._stage2_action` is just a post-init alias to the same buffer.

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def last_v4_residual(self) -> torch.Tensor:
        return self._v4_residual

    # ----------------------------------------------------------------------
    def _bind_command_terms(self):
        cm = self._env.command_manager
        self._t_base_velocity = cm.get_term(_BASE_VEL_CMD)
        self._t_body_height = cm.get_term(_BODY_HEIGHT_CMD)
        self._t_left_ee = cm.get_term(_LEFT_EE_CMD)
        self._t_right_ee = cm.get_term(_RIGHT_EE_CMD)
        self._t_alpha = cm.get_term(_ALPHA_CMD)
        self._cmd_terms_bound = True

    def _write_commands(self, u: torch.Tensor):
        """Write the Stage 2 action `u` into V4's command buffers in-place.

        u layout: see STAGE2_ACTION_DIM block in module docstring.
        """
        # base_velocity (3-D) — UniformVelocityCommand stores under `.vel_command_b`
        # or similar. Patch its `.command` tensor wherever the term keeps it.
        # In IsaacLab the canonical buffer is `vel_command_b` (a 3-D tensor).
        if hasattr(self._t_base_velocity, "vel_command_b"):
            self._t_base_velocity.vel_command_b[:] = u[:, 0:3]
        else:
            # Fall back to overwriting via the public command property's
            # underlying tensor — not all CommandTerm impls expose this.
            raise RuntimeError("base_velocity term layout unknown.")

        # body_height (1-D) — UniformScalarCommand has `_command` (N, 1).
        self._t_body_height._command[:] = u[:, 3:4]

        # left_ee_pose (7-D) — UniformPoseCommand has `pose_command_b` (N, 7) wxyz.
        # Normalize the quat slot before writing.
        l_pos = u[:, 4:7]
        l_quat = u[:, 7:11]
        l_quat = _safe_normalize_quat(l_quat)
        self._t_left_ee.pose_command_b[:, 0:3] = l_pos
        self._t_left_ee.pose_command_b[:, 3:7] = l_quat

        # right_ee_pose (7-D)
        r_pos = u[:, 11:14]
        r_quat = u[:, 14:18]
        r_quat = _safe_normalize_quat(r_quat)
        self._t_right_ee.pose_command_b[:, 0:3] = r_pos
        self._t_right_ee.pose_command_b[:, 3:7] = r_quat

        # waist_regularization (1-D)
        self._t_alpha._command[:] = u[:, 18:19]

    def _read_v4_obs(self) -> torch.Tensor:
        """Pull the `v4_actor` observation group from the env's obs manager.

        This obs was computed at the END of the previous env step (before the
        Stage 2 action was applied). So the V4 commands it sees lag by one
        step. Acceptable; Stage 2 frequency is lower than the control freq.
        """
        om = self._env.observation_manager
        # `compute_group` returns the concat tensor for the group.
        return om.compute_group(self._v4_obs_group_name)

    def _kmp_forward(self, u: torch.Tensor) -> torch.Tensor:
        """Run KMP on the FRESH u_t (so q_prior reflects the action just taken).

        Reorders the EE quat slots from wxyz (V4 / Isaac) to xyzw (KMP).
        Returns the 28-D KMP output reindexed into action-term joint order.
        """
        # u quats are at slots 7:11 (left) and 14:18 (right), in wxyz layout
        # after _safe_normalize_quat.
        buf = self._cmd_buf
        buf[:, 0:1] = u[:, 3:4]                # h
        buf[:, 1:4] = u[:, 4:7]                # l_pos
        buf[:, 4:7] = u[:, 8:11]               # l qx, qy, qz
        buf[:, 7:8] = u[:, 7:8]                # l qw -> xyzw tail
        buf[:, 8:11] = u[:, 11:14]             # r_pos
        buf[:, 11:14] = u[:, 15:18]            # r qx, qy, qz
        buf[:, 14:15] = u[:, 14:15]            # r qw
        buf[:, 15:16] = u[:, 18:19]            # α

        with torch.no_grad():
            q_prior_kmp = self._kmp(buf)        # (B, 28) KMP order

        q_prior_action = torch.empty_like(q_prior_kmp)
        q_prior_action[:, self._kmp_to_action] = q_prior_kmp
        return q_prior_action

    # ----------------------------------------------------------------------
    def process_actions(self, actions: torch.Tensor) -> None:
        """Stage 2 step: write commands, run V4 residual, run KMP, set targets."""
        # Lazy bind command-term and obs-group handles on first call.
        if not self._cmd_terms_bound:
            self._bind_command_terms()

        # `actions` lands in self._raw_actions via the alias self._stage2_action.
        # Doing an explicit copy is redundant since the parent's process_actions
        # would write self._raw_actions[:] = actions; we replace that behavior
        # entirely so we copy here.
        self._stage2_action[:] = actions

        # 1) Write Stage 2 output into V4's command buffers.
        self._write_commands(self._stage2_action)

        # 2) Read V4 obs (1-step-lagged commands inside).
        v4_obs = self._read_v4_obs()

        # 3) Run frozen V4 actor (-> 28-D residual).
        with torch.no_grad():
            residual = self._v4_policy(v4_obs)
        self._v4_residual[:] = residual

        # 4) Run frozen KMP on FRESH commands.
        q_prior_action = self._kmp_forward(self._stage2_action)

        # 5) q_target = q_prior + residual * per_joint_scale
        self._processed_actions = q_prior_action + residual * self._scale

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    # apply_actions inherited from JointPositionAction — set_joint_position_target.

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._stage2_action.zero_()
            self._v4_residual.zero_()
        else:
            self._stage2_action[env_ids] = 0.0
            self._v4_residual[env_ids] = 0.0


def _safe_normalize_quat(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a (..., 4) quaternion buffer; fall back to identity wxyz if zero."""
    norm = q.norm(dim=-1, keepdim=True)
    safe = norm > eps
    q_normed = torch.where(safe, q / norm.clamp_min(eps), q)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.0
    return torch.where(safe, q_normed, identity)


@configclass
class Stage2WrappedActionCfg(JointPositionActionCfg):
    """Config for `Stage2WrappedAction`.

    Most fields mirror V4's `KMPResidualJointPositionActionCfg`. New fields:
      - `v4_jit_policy` — path to V4's JIT-exported actor (`policy.pt`)
      - `v4_obs_group_name` — name of the ObsGroup that exactly replicates V4's
        actor observation (history_length=5, V3 PolicyCfg layout).
      - `kmp_checkpoint` — same KMP V4 used.
      - `residual_scale` / `scale` — per-joint residual scale from V4.

    The action term's `joint_names` MUST resolve to the same 28 joints V4's
    KMP output order references (legs + arms + waist roll/pitch).
    """

    class_type: type = Stage2WrappedAction

    v4_jit_policy: str = MISSING
    v4_obs_group_name: str = "v4_actor"
    kmp_checkpoint: str = MISSING
    residual_scale: float | None = None
    use_default_offset: bool = False
