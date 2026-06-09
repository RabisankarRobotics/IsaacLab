"""HV1 V4 loco-manipulation — KMP-residual action (HiWET Eq. 11).

V4 differences vs V3:
  * Action: `KMPResidualJointPositionActionCfg` replaces `JointPositionActionCfg`.
    The frozen KMP MLP (deploy/model/kmp/kmp_v1.pt) maps current commands ->
    a kinematically-feasible 28-D joint posture (`q_prior`). The actor's
    output is a small residual on top: q_target = q_prior + residual * scale,
    where `scale` is per-joint (legs 0.25, arms/waist 0.10) so leg swing
    has the range it needs while arm corrections stay small.
    PD layer downstream is unchanged.
  * r_kmp DISABLED (weight 0). HiWET's -0.05 weight rewards staying near
    q_prior, which is the *standing* pose for any command (the KMP was
    trained only on staged-IK statics, no gait). With r_kmp on, the
    17400-iter run greedy-climbed into a "stand still in KMP pose" local
    optimum and never learned to walk. Re-enable later (small positive
    weight like 1e-3) for regularization once walking exists.
  * Curriculum dropped — body-height tracking is enabled from iter 0. KMP
    makes height a kinematic constraint (already solved offline).
  * Walking-escape shaping (post 17400-iter standing-still run):
      feet_air_time × 4  (was × 2 — too weak to dominate the standing pit)
      stand_still_legs   = -4.0   (was -2.0 — robot ignored it)
      rel_standing_envs  = 0.10   (was 0.20 — too fat a free-reward floor)
      track_lin_vel_xy_exp.weight = 4.0  (was 3.0 — lifts marginal value
                                          of "actually walk" above standing)
  * V4.1 standing-height fix (post 15k walking-good run):
      rel_standing_envs  = 0.20   (was 0.10 — more "stand AND track height"
                                   gradient; doesn't reopen the standing pit
                                   because the rest of the shaping holds)
      stand_still_legs joints = hip + ankle only (was hip + knee + ankle —
                                   knees free to bend so robot can squat
                                   to commanded height while standing)

Train from scratch — do NOT warm-start from V3. V3 weights are tuned for
"discover IK and dynamics simultaneously" and feeding them into V4 (where
IK is solved upstream) injects a wrong prior. Expected convergence: 5-8k
iters. Go/no-go: at iter 2000, EE position error should already beat V3
iter-32000 (~9 cm → target <3 cm).
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
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

    # r_kmp DISABLED for V4-walk pass.
    # The HiWET -0.05 weight rewards STAYING NEAR q_prior; q_prior is the
    # *standing* posture for any given command, since the KMP was trained
    # on staged-IK statics with no gait. With r_kmp on, the 17400-iter run
    # converged to "output zero residual, stand still, collect +9 reward,
    # ignore the -2 stand_still penalty." Setting weight=0 lets the actor
    # produce the swing-magnitude residuals walking actually needs. Re-
    # enable (small positive weight like 1e-3) only after walking emerges,
    # to gently regularize once a stable gait exists.
    r_kmp = RewTerm(
        func=custom_mdp.kmp_residual_l2,
        weight=0.0,
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

        # --- Restore V3 shaping weights -------------------------------------
        # Earlier "halve everything because KMP handles geometry" softening
        # was wrong: V3's tuned shaping was load-bearing for GAIT, not just
        # posture. At iter 3000 of the first V4 run we observed:
        #   * EE tracking already solved (~9 cm — V3 needed 32000 iters)
        #   * height tracking decent
        #   * feet_air_time ≈ 0.0005 (feet never airborne, robot slides)
        #   * base_contact termination = 57% (falls when commanded to walk)
        # The cheapest strategy with the over-softened weights was "stand
        # in q_prior pose and collect EE+height rewards." Restoring V3's
        # stand_still / action_rate / joint_deviation pressure rebuilds the
        # gradient toward "lift feet and walk" without touching the V4-
        # specific KMP terms (r_kmp + KMP residual action).
        #
        # No __post_init__ change vs V3 here: V3's __post_init__ already set
        # the tuned weights (Stage-4 values), and our super().__post_init__()
        # call applied them. Nothing further to do.

        # --- Walking-escape shaping (post 17400-iter standing-still run) ---
        # The previous V4 run converged to a standing-still local optimum:
        # feet_air_time=0.0009, error_vel_xy=0.48, base_contact term 16%,
        # ep_reward=103 (all from EE + height + standing envs). The fixes
        # below break the standing pit by (a) removing the r_kmp anchor
        # that pulled residuals to zero (see RewardsCfg above), (b) making
        # standing more expensive than transient walking instability, and
        # (c) shrinking the "free reward" floor from rel_standing_envs.

        # (1) Crank feet_air_time 5x — the only direct "lift your feet"
        #     signal in the reward stack. 2x in the previous run wasn't
        #     enough to dominate the standing pose reward sum.
        if hasattr(self.rewards, "feet_air_time"):
            self.rewards.feet_air_time.weight = float(self.rewards.feet_air_time.weight) * 4.0

        # (2) Crank stand_still_legs from -2.0 to -4.0, AND exclude knees
        #     from the penalty. The base term applies to all leg joints (hip
        #     + knee + ankle), which fights body-height tracking: matching a
        #     low commanded height requires knee bend, but knee bend itself
        #     was penalized -> robot stood at KMP-default height (~0.94) and
        #     ignored the height command in standing envs. Keeping hip +
        #     ankle ensures the robot still stands upright over its feet;
        #     freeing the knee lets it squat to the commanded height.
        if hasattr(self.rewards, "stand_still_legs"):
            self.rewards.stand_still_legs.weight = -4.0
            self.rewards.stand_still_legs.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                joint_names=[
                    "^(left|right)_hip_(yaw|pitch|roll)_joint$",
                    "^(left|right)_ankle_(pitch|roll)_joint$",
                ],
            )

        # (3) Standing-env share. 0.10 broke the standing-still pit (15k
        #     run walks well) but starved the "track height while standing"
        #     gradient — at deploy the robot stands at KMP-default height
        #     and ignores the body-height command. Bumping to 0.20 doubles
        #     the supervision for "stand AND match commanded height" without
        #     re-creating a free-reward floor that disincentivizes walking.
        self.commands.base_velocity.rel_standing_envs = 0.20

        # (4) Boost track_lin_vel_xy_exp weight to 5.0 (V3 default is 3.0).
        #     The previous run got 1.58/3.0 — only ~50% of available reward.
        #     A bigger ceiling on this term lifts the marginal value of
        #     "actually walk" above the marginal value of "stand still."
        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = 4.0

        # --- Dump effective reward weights (helps trace inheritance) -------
        # The reward weight you see in training logs is the result of every
        # parent __post_init__ + this one. Printing them here removes the
        # need to grep through the V1→V2→V3→V4 chain to find the live value.
        print("\n=== HV1 V4 effective reward weights (post-inheritance) ===")
        for _name in sorted(vars(self.rewards)):
            _term = getattr(self.rewards, _name)
            _w = getattr(_term, "weight", None)
            if _w is not None:
                print(f"  {_name:36s} = {_w:+.4f}")
        print()


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
