"""HV1 V3 body-frame loco-manipulation task — HiWET Stage-1 robustification.

Delta vs V2:
  * Observation history: `history_length=5` on the policy group (actor only).
    Stacked along the time axis and auto-flattened by ObservationManager. Gives
    the actor temporal context so it can implicitly estimate base_lin_vel from
    proprioception — paper's State Estimator without the auxiliary MLP head.
  * New scalar command `body_height` ∈ [0.55, 0.78] m. Resampled every 6–10 s
    per env. Tracked via `base_height_tracking_exp`. Required for Stage-2
    later (Commander must be able to issue a height target, e.g. crouch to
    reach a low EE goal).
  * New scalar command `waist_regularization` (α_t) sampled log-uniform from
    [0.1, 10] per episode. Modulates the V2 waist-roll/pitch deviation penalty
    so the policy learns to map α_t → how much it leans on the waist. Same
    semantics as the paper's α_t, but without KMP yet (added in V4).
  * Critic obs: gets the same new commands as the actor + privileged
    base_lin_vel (kept from V2). Critic does NOT use history (faster, and the
    value head has access to ground truth for everything anyway).

Action space and reward weights inherited from V2 unchanged so V3 can warm-
start from a V2 checkpoint via --resume.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity import mdp as custom_mdp

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

    base_height_below = RewTerm(
        func=custom_mdp.base_height_below_target_l1,
        weight=0.0,  # disabled — V3 uses commanded height instead
        params={"target_height": 0.89},
    )

    # Track the commanded body height. std=0.05 → ~5 cm sweet spot.
    base_height_tracking = RewTerm(
        func=custom_mdp.base_height_tracking_exp,
        weight=1.0,
        params={"command_name": "body_height", "std": 0.05},
    )

    # Replace V2's fixed-weight waist deviation with an α_t-modulated version.
    # The RewTerm weight here is the *base* magnitude; α_t multiplies it
    # per-env per-step. With α∈[0.1, 10] and base=-0.2, effective weight
    # ranges from -0.02 to -2.0 — two decades, as in the paper.
    joint_deviation_waist_rp = RewTerm(
        func=custom_mdp.joint_deviation_l1_alpha_weighted,
        weight=-0.2,
        params={
            "command_name": "waist_regularization",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["waist_roll_joint", "waist_pitch_joint"]
            ),
        },
    )


# ---- env config -------------------------------------------------------------
@configclass
class HV1LocoManipV3EnvCfg(HV1LocoManipV2EnvCfg):
    observations: HV1LocoManipV3ObservationsCfg = HV1LocoManipV3ObservationsCfg()
    rewards: HV1LocoManipV3RewardsCfg = HV1LocoManipV3RewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Body-height command: tall stand 0.78 m (default), down to 0.55 m
        # crouch. Resample every 6–10 s so the policy actually sees a change
        # within a 14 s episode.
        self.commands.body_height = custom_mdp.UniformScalarCommandCfg(
            resampling_time_range=(6.0, 10.0),
            range=(0.55, 0.78),
            log_uniform=False,
        )

        # α_t: log-uniform [0.1, 10] per episode (resample once at reset).
        # Wide resample range so it's effectively per-episode, not per-step.
        self.commands.waist_regularization = custom_mdp.UniformScalarCommandCfg(
            resampling_time_range=(20.0, 20.0),
            range=(0.1, 10.0),
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

        # Hold body height steady during play so the user can visually inspect
        # whether the new commands are working (we mostly want to see walking +
        # EE reach here, height variation can be tested separately).
        self.commands.body_height.range = (0.75, 0.78)
        # Keep α_t mid-range during play so waist behavior is "average".
        self.commands.waist_regularization.range = (0.5, 2.0)
        self.commands.waist_regularization.log_uniform = False

        self.events.push_robot.interval_range_s = (3.0, 5.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}
        }
        self.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
        self.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)
