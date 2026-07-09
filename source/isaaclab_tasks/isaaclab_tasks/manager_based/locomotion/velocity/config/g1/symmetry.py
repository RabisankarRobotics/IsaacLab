# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry for the legs-only G1 (``Isaac-Velocity-Flat-Legs-G1-29Dof-Clean``).

Plugged into RSL-RL symmetry augmentation (``RslRlSymmetryCfg.data_augmentation_func``)
to force the actor to be left-right symmetric. This is the cure for the *random
"handedness" symmetry-break* we observed: the policy would learn to turn one way but
not the other (and the working side FLIPPED between training runs), and to strafe one
way cleanly while the mirror-image direction collided its feet. A symmetric penalty
cannot cause that — it is the policy arbitrarily picking a handedness. Augmenting every
minibatch with its left-right mirror removes that freedom, so the competent side's
skill is mirrored onto the other side (and it also evens out the "one leg more active"
step asymmetry).

Only a LEFT-RIGHT (sagittal-plane) mirror is applied — a biped is not front-back
symmetric like a quadruped — so the batch is augmented 2x (original + mirror).

Everything below is keyed to the EXACT obs/action layout of this task (dumped and
verified against the live env). If you change the obs term order, joint order, or the
gait offset, update the constants here.

Policy obs group (81 dims), in order::

    [ 0: 3)  base_ang_vel        (wx, wy, wz)      pseudovector -> (-wx,  wy, -wz)
    [ 3: 6)  projected_gravity   (gx, gy, gz)      polar vector -> ( gx, -gy,  gz)
    [ 6: 9)  velocity_commands   (vx, vy, yaw)     -> ( vx, -vy, -yaw)
    [ 9:11)  gait_phase          (sin, cos)        half-period shift -> (-sin, -cos)
    [11:23)  joint_pos  (12 legs, joint_pos_rel)   leg swap + sign flip
    [23:35)  joint_vel  (12 legs)                  leg swap + sign flip
    [35:47)  actions    (12 legs, last_action)     leg swap + sign flip
    [47:64)  upper_body_joint_pos (17)             upper swap + sign flip
    [64:81)  upper_body_joint_vel (17)             upper swap + sign flip

Critic obs group (84 dims) = the 81 policy dims (same order) + base_lin_vel(3) at
[81:84), which mirrors as a polar vector -> (vx, -vy, vz).

The mirror is an *involution* (applying it twice is the identity), and it maps the
(symmetric) default pose to itself — both are asserted in ``symmetry_check.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

# ---------------------------------------------------------------------------
# Obs block boundaries
# ---------------------------------------------------------------------------
_ANG = slice(0, 3)
_GRAV = slice(3, 6)
_CMD = slice(6, 9)
_GAIT = slice(9, 11)
_JPOS = slice(11, 23)
_JVEL = slice(23, 35)
_ACT = slice(35, 47)
_UPOS = slice(47, 64)
_UVEL = slice(64, 81)
_POLICY_DIM = 81
_LIN = slice(81, 84)  # base_lin_vel — critic group only
_CRITIC_DIM = 84

# ---------------------------------------------------------------------------
# Joint permutations + sign flips (output index i <- sign[i] * input[perm[i]])
# ---------------------------------------------------------------------------
# Leg joints, obs order (interleaved L,R pairs):
#   0 L_hip_pitch  1 R_hip_pitch  2 L_hip_roll  3 R_hip_roll  4 L_hip_yaw  5 R_hip_yaw
#   6 L_knee       7 R_knee       8 L_ank_pitch 9 R_ank_pitch 10 L_ank_roll 11 R_ank_roll
# Swap each L<->R pair; flip sign on the roll/yaw DOFs (hip_roll, hip_yaw, ankle_roll),
# which are antisymmetric about the sagittal plane. Pitch DOFs (hip_pitch, knee,
# ankle_pitch) are symmetric and keep their sign.
_LEG_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
_LEG_SIGN = [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1]

# Upper-body joints, obs order (3 midline waist DOFs, then interleaved arm L,R pairs):
#   0 waist_yaw   1 waist_roll  2 waist_pitch
#   3 L_sh_pitch  4 R_sh_pitch  5 L_sh_roll  6 R_sh_roll  7 L_sh_yaw  8 R_sh_yaw
#   9 L_elbow    10 R_elbow    11 L_wr_roll 12 R_wr_roll 13 L_wr_pitch 14 R_wr_pitch
#  15 L_wr_yaw   16 R_wr_yaw
# Waist stays in place; midline yaw/roll flip sign, waist_pitch keeps sign. Swap each
# arm L<->R pair; flip roll/yaw (shoulder_roll/yaw, wrist_roll/yaw), keep pitch/elbow.
_UP_PERM = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
_UP_SIGN = [-1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]

# Whole-vector sign flips for the leading vector blocks.
_ANG_FLIP = [-1.0, 1.0, -1.0]
_GRAV_FLIP = [1.0, -1.0, 1.0]
_CMD_FLIP = [1.0, -1.0, -1.0]
_GAIT_FLIP = [-1.0, -1.0]
_LIN_FLIP = [1.0, -1.0, 1.0]


def _flip(x: torch.Tensor, flip: list[float]) -> torch.Tensor:
    return x * torch.tensor(flip, device=x.device, dtype=x.dtype)


def _permute_flip(x: torch.Tensor, perm: list[int], sign: list[float]) -> torch.Tensor:
    """Return ``sign * x[..., perm]`` (swap joints, then flip signs)."""
    sign_t = torch.tensor(sign, device=x.device, dtype=x.dtype)
    return x[..., perm] * sign_t


def mirror_leg(x: torch.Tensor) -> torch.Tensor:
    """Left-right mirror a 12-dim leg vector (joint_pos_rel / joint_vel / action)."""
    return _permute_flip(x, _LEG_PERM, _LEG_SIGN)


def mirror_policy_obs(obs: torch.Tensor) -> torch.Tensor:
    """Left-right mirror the 81-dim policy observation."""
    m = obs.clone()
    m[..., _ANG] = _flip(obs[..., _ANG], _ANG_FLIP)
    m[..., _GRAV] = _flip(obs[..., _GRAV], _GRAV_FLIP)
    m[..., _CMD] = _flip(obs[..., _CMD], _CMD_FLIP)
    m[..., _GAIT] = _flip(obs[..., _GAIT], _GAIT_FLIP)  # anti-phase legs -> half-period shift
    m[..., _JPOS] = _permute_flip(obs[..., _JPOS], _LEG_PERM, _LEG_SIGN)
    m[..., _JVEL] = _permute_flip(obs[..., _JVEL], _LEG_PERM, _LEG_SIGN)
    m[..., _ACT] = _permute_flip(obs[..., _ACT], _LEG_PERM, _LEG_SIGN)
    m[..., _UPOS] = _permute_flip(obs[..., _UPOS], _UP_PERM, _UP_SIGN)
    m[..., _UVEL] = _permute_flip(obs[..., _UVEL], _UP_PERM, _UP_SIGN)
    return m


def _mirror_obs_group(x: torch.Tensor) -> torch.Tensor:
    """Mirror an observation group, dispatched by width (81 = policy, 84 = critic)."""
    w = x.shape[-1]
    if w == _POLICY_DIM:
        return mirror_policy_obs(x)
    if w == _CRITIC_DIM:
        m = x.clone()
        m[..., :_POLICY_DIM] = mirror_policy_obs(x[..., :_POLICY_DIM])
        m[..., _LIN] = _flip(x[..., _LIN], _LIN_FLIP)
        return m
    raise ValueError(f"g1 symmetry: unexpected obs width {w} (expected 81 policy or 84 critic)")


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv" = None,
    obs=None,
    actions: torch.Tensor | None = None,
):
    """Augment observations/actions with their left-right mirror (2x augmentation).

    Signature matches ``RslRlSymmetryCfg.data_augmentation_func``. ``obs`` is a
    ``TensorDict`` with a ``policy`` group (81) and a ``critic`` group (84); each is
    mirrored by width. Either input may be ``None`` (RSL-RL calls this three ways:
    obs+actions, obs-only, actions-only). The original samples occupy ``[:B]`` and the
    mirror occupies ``[B:2B]``.
    """
    num_aug = 2  # original + left-right mirror

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(num_aug)
        for key in obs.keys():
            obs_aug[key][:batch_size] = obs[key]
            obs_aug[key][batch_size : 2 * batch_size] = _mirror_obs_group(obs[key])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = actions.repeat(num_aug, 1)
        actions_aug[batch_size : 2 * batch_size] = mirror_leg(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
