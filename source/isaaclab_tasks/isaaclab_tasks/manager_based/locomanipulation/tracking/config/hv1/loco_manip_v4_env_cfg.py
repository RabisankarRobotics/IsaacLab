"""HV1 V4 loco-manipulation — KMP-residual action (HiWET Eq. 11).

V4 differences vs V3:
  * Action: `KMPResidualJointPositionActionCfg` replaces `JointPositionActionCfg`.
    The frozen KMP MLP (deploy/model/kmp/kmp_v1.pt) maps current commands ->
    a kinematically-feasible 28-D joint posture (`q_prior`). The actor's
    output is a small residual on top: q_target = q_prior + residual * scale,
    where `scale` is per-joint (legs 0.25, arms/waist 0.10) so leg swing
    has the range it needs while arm corrections stay small.
    PD layer downstream is unchanged.
  * New reward `r_kmp = -‖residual‖²` (weight -0.05) per HiWET Eq. 12.
  * Curriculum dropped — body-height tracking is enabled from iter 0. KMP
    makes height a kinematic constraint (already solved offline), so V3's
    two-stage walk-then-add-height curriculum is no longer needed.
  * Shaping rewards halved (action_rate, joint_deviation): KMP already
    smooths and anchors posture, the original shaping pressure is double-
    counting.
  * stand_still_legs softened (-2.0 -> -0.5): KMP at v=0 already produces a
    standing posture, less reward shaping needed to discourage marching.

Train from scratch — do NOT warm-start from V3. V3 weights are tuned for
"discover IK and dynamics simultaneously" and feeding them into V4 (where
IK is solved upstream) injects a wrong prior. Expected convergence: 5-8k
iters. Go/no-go: at iter 2000, EE position error should already beat V3
iter-32000 (~9 cm → target <3 cm).
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity import mdp as custom_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity.flat_env_cfg import (
    ARM_JOINT_NAMES,
    LEG_JOINTS,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity.kmp_action import (
    KMPResidualJointPositionActionCfg,
)

from .loco_manip_v2_env_cfg import (
    HV1LocoManipV2ActionsCfg,
    WAIST_ACTUATED_JOINTS,
)
from .loco_manip_v3_env_cfg import (
    HV1LocoManipV3EnvCfg,
    HV1LocoManipV3ObservationsCfg,
    HV1LocoManipV3RewardsCfg,
)


KMP_CKPT = "/home/rabisankar/IsaacLab/deploy/model/kmp/kmp_v1.pt"


# ---- actions: KMP-residual replaces plain JointPositionAction --------------
# Per-joint residual scale: legs get 0.25 rad/unit (need swing motion the KMP
# cannot anticipate — knee swings ~0.45 rad, hip ~0.7 rad during a step);
# arms and waist get 0.10 rad/unit (KMP's static reach pose is already close
# to right, only small corrections needed for momentum/balance).
#
# Total magnitude per-step is bounded by PPO's policy mean × scale + noise.
# With init_noise_std=0.6 and learned-mean-up-to-±3, the legs get an
# effective working range of ±~0.75 rad, arms ±~0.30 rad. Matches the
# kinematic demands of walking + dexterous arm motion respectively.
_KMP_RESIDUAL_SCALE = {
    # Legs — dynamic motion (swing, balance, foot placement)
    "left_hip_yaw_joint":     0.25,
    "left_hip_pitch_joint":   0.25,
    "left_hip_roll_joint":    0.25,
    "left_knee_joint":        0.25,
    "left_ankle_pitch_joint": 0.25,
    "left_ankle_roll_joint":  0.25,
    "right_hip_yaw_joint":     0.25,
    "right_hip_pitch_joint":   0.25,
    "right_hip_roll_joint":    0.25,
    "right_knee_joint":        0.25,
    "right_ankle_pitch_joint": 0.25,
    "right_ankle_roll_joint":  0.25,
    # Arms + waist — KMP-served statics; small residual is enough
    "left_shoulder_pitch_joint":  0.10,
    "left_shoulder_roll_joint":   0.10,
    "left_shoulder_yaw_joint":    0.10,
    "left_elbow_joint":           0.10,
    "left_wrist_roll_joint":      0.10,
    "left_wrist_pitch_joint":     0.10,
    "left_wrist_yaw_joint":       0.10,
    "right_shoulder_pitch_joint": 0.10,
    "right_shoulder_roll_joint":  0.10,
    "right_shoulder_yaw_joint":   0.10,
    "right_elbow_joint":          0.10,
    "right_wrist_roll_joint":     0.10,
    "right_wrist_pitch_joint":    0.10,
    "right_wrist_yaw_joint":      0.10,
    "waist_roll_joint":           0.10,
    "waist_pitch_joint":          0.10,
}


@configclass
class HV1LocoManipV4ActionsCfg(HV1LocoManipV2ActionsCfg):
    joint_pos = KMPResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS + ARM_JOINT_NAMES + WAIST_ACTUATED_JOINTS,
        # CRITICAL: KMP outputs joints in V3 action order (LEG + ARM + WAIST).
        # With preserve_order=False (the default) the action manager would
        # resolve joints alphabetically and q_prior slots would land on the
        # wrong joints. Force preserve_order=True so the action vector's
        # slot order = our joint_names order = KMP output order.
        preserve_order=True,
        kmp_checkpoint=KMP_CKPT,
        scale=_KMP_RESIDUAL_SCALE,
        residual_scale=None,   # None -> use the per-joint `scale` dict above
        use_default_offset=False,  # KMP supplies the offset per-step
    )


# ---- rewards: V3 + r_kmp, shaping weights softened -------------------------
@configclass
class HV1LocoManipV4RewardsCfg(HV1LocoManipV3RewardsCfg):
    """V3 rewards plus r_kmp; shaping weights softened in __post_init__."""

    r_kmp = RewTerm(
        func=custom_mdp.kmp_residual_l2,
        weight=-0.05,  # HiWET Eq. 12 default
        params={"action_term_name": "joint_pos"},
    )


# ---- env --------------------------------------------------------------------
@configclass
class HV1LocoManipV4EnvCfg(HV1LocoManipV3EnvCfg):
    """V4 = V3 + KMP residual action. Curriculum removed (height from iter 0)."""

    actions: HV1LocoManipV4ActionsCfg = HV1LocoManipV4ActionsCfg()
    observations: HV1LocoManipV3ObservationsCfg = HV1LocoManipV3ObservationsCfg()
    rewards: HV1LocoManipV4RewardsCfg = HV1LocoManipV4RewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- Drop the V3 height-tracking curriculum ------------------------
        # KMP already solves the height constraint kinematically, so PPO sees
        # a feasible standing-at-height pose from iter 0. No need to wait
        # 3000 iters for "walking + EE" to converge before adding height.
        # V3's base_height_tracking is already pinned to weight 5.0 in V3's
        # __post_init__; we leave that pinned value alone (V4 trains from
        # scratch but the pinned value matches what we want from iter 0).
        if hasattr(self.curriculum, "enable_height_tracking"):
            self.curriculum.enable_height_tracking = None

        # --- Soften shaping weights: KMP handles posture geometry ----------
        # action_rate penalizes high-freq joint chatter. KMP outputs a smooth
        # q_prior; only the residual contributes chatter, and the residual
        # is already small (scale 0.15). Halve the penalty so we don't
        # double-tax the residual.
        if hasattr(self.rewards, "action_rate_l2"):
            self.rewards.action_rate_l2.weight = (
                float(self.rewards.action_rate_l2.weight) * 0.5
            )
        if hasattr(self.rewards, "action_rate_arms_l2"):
            self.rewards.action_rate_arms_l2.weight = (
                float(self.rewards.action_rate_arms_l2.weight) * 0.5
            )

        # joint_deviation anchors joints to a default pose. The KMP IS the
        # anchor in V4 — its q_prior is the natural posture for any command.
        # Halve any joint_deviation terms inherited from V3.
        for term_name in [
            "joint_deviation_legs", "joint_deviation_arms",
            "joint_deviation_waist_rp",
        ]:
            if hasattr(self.rewards, term_name):
                term = getattr(self.rewards, term_name)
                if term is not None and term.weight is not None:
                    term.weight = float(term.weight) * 0.5

        # stand_still_legs: at v=0, KMP already produces a standing posture,
        # so we need less explicit pressure to suppress marching-in-place.
        if hasattr(self.rewards, "stand_still_legs"):
            self.rewards.stand_still_legs.weight = -0.5  # V3 had -2.0


@configclass
class HV1LocoManipV4EnvCfg_PLAY(HV1LocoManipV4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.commands.base_velocity.rel_standing_envs = 0.30
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.resampling_time_range = (8.0, 8.0)

        self.commands.left_ee_pose.resampling_time_range = (4.0, 4.0)
        self.commands.right_ee_pose.resampling_time_range = (4.0, 4.0)

        self.commands.body_height.range = (0.85, 0.95)
        self.commands.waist_regularization.range = (0.5, 2.0)
        self.commands.waist_regularization.log_uniform = False

        self.events.push_robot.interval_range_s = (3.0, 5.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}
        }
        self.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
        self.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)
