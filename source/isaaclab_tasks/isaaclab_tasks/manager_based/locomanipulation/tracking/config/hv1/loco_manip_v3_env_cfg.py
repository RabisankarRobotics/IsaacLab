"""HV1 V3 body-frame loco-manipulation task — HiWET Stage-1 robustification.

Trained from scratch with a **two-stage curriculum** (no V2 warm-start needed):

  * Stage 1 (iter 0 → 3000, common_step_counter ≤ 72000):
      `base_height_tracking.weight = 0.0` (disabled). Policy learns walking +
      EE tracking only. Walking reward is boosted (track_lin_vel_xy_exp=3.0)
      and EE L1 penalty is softened (-2.0 → -1.0) so the policy actually
      walks instead of standing still — V2's main failure mode.
  * Stage 2 (iter 3000+):
      `CurriculumTerm` flips `base_height_tracking.weight` to 2.5 with
      std=0.10. By this point the policy has muscle-memory walking + EE,
      so adding height tracking is a small adjustment rather than a
      competing objective. Range narrowed to [0.85, 0.95] (mild crouch only,
      no deep squat) to avoid the physical conflict between deep squat and
      simultaneous walking + EE reach.

Other deltas vs V2:
  * Observation history: `history_length=5` on the policy group (actor only).
    Stacked along the time axis and auto-flattened by ObservationManager.
    Gives the actor temporal context so it can implicitly estimate
    base_lin_vel from proprioception — paper's State Estimator without the
    auxiliary MLP head.
  * New scalar command `waist_regularization` (α_t) sampled log-uniform from
    [0.1, 3.0] per episode (narrowed from paper's [0.1, 10] — at α=10 the
    waist penalty was suppressing the flexion needed for crouch).
    Modulates the V2 waist-roll/pitch deviation penalty with a milder base
    weight (-0.05). Same semantics as the paper's α_t, but without KMP yet
    (added in V4).
  * Critic obs: gets the same new commands as the actor + privileged
    base_lin_vel (kept from V2). Critic does NOT use history (faster, and the
    value head has access to ground truth for everything anyway).
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity import mdp as custom_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity.flat_env_cfg import (
    ARM_JOINT_NAMES,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CurriculumCfg as BaseCurriculumCfg,
)

from .loco_manip_v2_env_cfg import (
    HV1LocoManipV2EnvCfg,
    HV1LocoManipV2ObservationsCfg,
    HV1LocoManipV2RewardsCfg,
    TORSO_BODY,
)


# ---- observations: V2 layout + new scalar command obs + history on policy ---
@configclass
class HV1LocoManipV3ObservationsCfg(HV1LocoManipV2ObservationsCfg):
    @configclass
    class PolicyCfg(HV1LocoManipV2ObservationsCfg.PolicyCfg):
        body_height_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "body_height"}
        )
        waist_alpha_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "waist_regularization"}
        )

        def __post_init__(self):
            super().__post_init__()
            # Stack 5 past steps (incl. current) → auto-flattened to (N, 5·D)
            # by ObservationManager. Same effect as a concat-style history
            # encoder — gives the actor enough temporal signal to implicitly
            # estimate base_lin_vel + disturbances.
            self.history_length = 5

    @configclass
    class CriticCfg(HV1LocoManipV2ObservationsCfg.CriticCfg):
        body_height_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "body_height"}
        )
        waist_alpha_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "waist_regularization"}
        )
        # No history on critic — privileged info already gives full state.

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---- rewards: V2 + body-height tracking + α-weighted waist dev --------------
@configclass
class HV1LocoManipV3RewardsCfg(HV1LocoManipV2RewardsCfg):

    # base_height_below = RewTerm(
    #     func=custom_mdp.base_height_below_target_l1,
    #     weight=0.0,  # disabled — V3 uses commanded height instead
    #     params={"target_height": 0.89},
    # )

    # Track the commanded body height. std=0.10 gives a forgiving ~10 cm sweet
    # spot so normal walking-bob (~3-5 cm vertical pelvis oscillation) doesn't
    # crush the reward and force the policy to stand still.
    #
    # Weight=2.5 is the post-curriculum (Stage-2) value. Curriculum
    # `CurriculumCfg.enable_height_tracking` would also push this to 2.5 at
    # iter ~3000 (common_step_counter > 72000) but only fires for fresh-from-
    # scratch training — `common_step_counter` is NOT persisted in RSL-RL
    # checkpoints, so on resume it resets to 0 and the curriculum waits
    # another 3000 iters before firing. We pin the post-curriculum weight
    # directly so resumed training sees the correct reward immediately.
    #
    # If retraining V3 from scratch (no resume), set this back to weight=0.0
    # so the staged curriculum applies — Stage-1 walking+EE then Stage-2 adds
    # height. The curriculum below will then flip to 2.5 at iter 3000.
    base_height_tracking = RewTerm(
        func=custom_mdp.base_height_tracking_exp,
        weight=2.5,  # was 0.0 (Stage-1) — pinned for resume robustness
        params={"command_name": "body_height", "std": 0.10},
    )

    # Extra action-rate penalty on the arm joints only — sits on top of the
    # inherited global `action_rate_l2 = -0.003`. Arms have no joint-deviation
    # anchor (only EE pose tracking) and during walking they tend to flap as
    # emergent counter-balance to leg swing. The global rate covers them at
    # -0.003 already; this term adds -0.007 → effective -0.010 for arm dims,
    # which suppresses high-frequency jitter without affecting steady reach
    # (slow arm motion costs almost nothing in (a_t-a_{t-1})²). Legs and waist
    # are untouched so the in-progress gait is not disturbed mid-resume.
    action_rate_arms_l2 = RewTerm(
        func=custom_mdp.action_rate_l2_joint_subset,
        weight=-0.007,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
        },
    )

    # Replace V2's fixed-weight waist deviation with an α_t-modulated version.
    # The RewTerm weight here is the *base* magnitude; α_t multiplies it
    # per-env per-step. With α∈[0.1, 3.0] and base=-0.05, effective weight
    # ranges from -0.005 to -0.15 — wide enough for the policy to learn the
    # α_t → waist-use mapping without crushing deep-crouch behaviour.
    joint_deviation_waist_rp = RewTerm(
        func=custom_mdp.joint_deviation_l1_alpha_weighted,
        weight=-0.05,
        params={
            "command_name": "waist_regularization",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["waist_roll_joint", "waist_pitch_joint"]
            ),
        },
    )


# ---- curriculum: enable body-height tracking after walking + EE converge ---
@configclass
class HV1LocoManipV3CurriculumCfg(BaseCurriculumCfg):
    """Stage-1 → Stage-2 trigger.

    `common_step_counter` advances by `num_steps_per_env` (=24) per PPO iter,
    so `num_steps=72000` ≈ iter 3000. By this point walking + EE should be
    stable; the curriculum flips `base_height_tracking.weight` from 0.0 to
    2.5 and the policy starts learning the height command on top.

    Inherits `terrain_levels` from the base (disabled by flat env's
    __post_init__ via `self.curriculum.terrain_levels = None`).
    """

    enable_height_tracking = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "base_height_tracking",
            "weight": 2.5,
            "num_steps": 72000,  # iter 3000 × num_steps_per_env 24
        },
    )


# ---- env config -------------------------------------------------------------
@configclass
class HV1LocoManipV3EnvCfg(HV1LocoManipV2EnvCfg):
    observations: HV1LocoManipV3ObservationsCfg = HV1LocoManipV3ObservationsCfg()
    rewards: HV1LocoManipV3RewardsCfg = HV1LocoManipV3RewardsCfg()
    curriculum: HV1LocoManipV3CurriculumCfg = HV1LocoManipV3CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # --- Stage-1 reward rebalancing (walking + EE focus) ----------------
        # V2 inherited from V1 set track_lin_vel_xy = 2.0 and EE L1 = -2.0.
        # That was the source of V2's "small steps / doesn't walk" failure:
        # the L1 EE penalty (linear in error, up to ~-1.0/step per hand) was
        # crushing the walking signal whenever EE drift opened up during a
        # step. Bumping walking and softening the L1 penalty gives the policy
        # room to commit to a real gait while still tracking EE coarsely.
        # The tanh fine reward (weight=1.5, std=0.10) is unchanged — it
        # provides the fine-grained EE tracking signal once the policy is
        # walking and roughly aligned.
        self.rewards.track_lin_vel_xy_exp.weight = 3.0   # was 2.0
        self.rewards.left_ee_pos_tracking.weight = -1.0  # was -2.0 (L1)
        self.rewards.right_ee_pos_tracking.weight = -1.0  # was -2.0 (L1)

        # --- Stage-3 rebalance (applied for resume at iter ~9500) -----------
        # At iter ~9500 walking/ang_vel/height were all >85% saturated while
        # EE fine sat at ~15% (0.24/1.5) — policy abandoned EE because the
        # softened L1 (-1.0) and weak orient (-0.2) couldn't compete with
        # +6.6/step from walking+height. Paper trains EE+walking simultaneously
        # (HiWET Eq. 3), so we keep that structure and pull EE back into the
        # gradient by restoring strong weights. std=0.15 (was 0.10) widens the
        # tanh sweet spot so walking-bob (~3-5 cm wrist motion) doesn't crush
        # the gradient mid-step. Orient weight doubled to anchor end-effector
        # roll/pitch/yaw which was at 0.4 rad error.
        self.rewards.left_ee_pos_tracking.weight = -2.5       # was -1.0
        self.rewards.right_ee_pos_tracking.weight = -2.5      # was -1.0
        self.rewards.left_ee_pos_tracking_fine.weight = 2.5   # was 1.5
        self.rewards.right_ee_pos_tracking_fine.weight = 2.5  # was 1.5
        self.rewards.left_ee_pos_tracking_fine.params["std"] = 0.15   # was 0.10
        self.rewards.right_ee_pos_tracking_fine.params["std"] = 0.15
        self.rewards.left_ee_orient_tracking.weight = -0.4    # was -0.2
        self.rewards.right_ee_orient_tracking.weight = -0.4

        # --- Body-height command --------------------------------------------
        # Range narrowed to mild crouch only: 0.85 m = ~10 cm below natural
        # standing. No deep squat — that creates physical conflict with
        # simultaneous walking + EE reach (heavily flexed legs can't generate
        # a useful gait). Stage-2 Commander will issue heights inside this
        # range; deeper crouch can be unlocked later if a task demands it.
        # Resample every 4–6 s so each 14 s episode sees ~3 height changes.
        self.commands.body_height = custom_mdp.UniformScalarCommandCfg(
            resampling_time_range=(4.0, 6.0),
            range=(0.85, 0.95),
            log_uniform=False,
            metric_source="root_pos_z",  # → Metrics/body_height/error in tensorboard
            debug_vis=True,  # red plate = h^des, green plate = pelvis z
        )

        # Safety floor: penalty if pelvis drops below 0.80 m. With commanded
        # height ≥ 0.85, a floor of 0.80 gives 5 cm of margin for transient
        # bob during walking before flagging "policy collapsed / fell over".
        self.rewards.base_height_below.params["target_height"] = 0.80

        # α_t: log-uniform [0.1, 3.0] per episode (resample once at reset).
        # Wide resample range so it's effectively per-episode, not per-step.
        # Upper bound narrowed from paper's 10 because at α=10 with base
        # weight -0.05 the effective penalty was -0.5/step — strong enough to
        # suppress the waist flexion the policy occasionally needs.
        self.commands.waist_regularization = custom_mdp.UniformScalarCommandCfg(
            resampling_time_range=(20.0, 20.0),
            range=(0.1, 3.0),
            log_uniform=True,
        )


@configclass
class HV1LocoManipV3EnvCfg_PLAY(HV1LocoManipV3EnvCfg):
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

        # PLAY uses the same range as training [0.85, 0.95] (mild crouch
        # only — no deep squat). If you want to stress-test the policy
        # beyond its training distribution, widen this range manually.
        self.commands.body_height.range = (0.85, 0.95)
        # Keep α_t mid-range during play so waist behavior is "average".
        self.commands.waist_regularization.range = (0.5, 2.0)
        self.commands.waist_regularization.log_uniform = False

        self.events.push_robot.interval_range_s = (3.0, 5.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}
        }
        self.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
        self.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)
