"""HV1 V2 body-frame loco-manipulation task (Stage 5).

Changes vs HV1LocoManipEnvCfg (V1):
  * Action: 26 → 28  (legs + arms + waist_roll + waist_pitch; waist_yaw stays
    PD-pinned because the first V2-actuated run showed it drifting ~5° and
    causing visible pelvis jerk as the robot tried to keep yaw at zero).
  * Actor obs: drop `base_lin_vel` (not measurable from a single IMU on the
    real robot), add `projected_gravity_torso` + `torso_ang_vel` (torso IMU
    so the policy can see what waist actuation does to the upper body).
  * Critic obs: keep everything the actor has + privileged `base_lin_vel`
    (asymmetric actor-critic). Mapped via `obs_groups` in the runner cfg.
  * Reward: `joint_deviation_waist_rp` (mild) + torso-frame
    `flat_orientation_l2` + `ang_vel_xy_l2` so the upper body stays upright
    once the waist is actuated (built-in versions read the pelvis only).
  * Event: narrow `pin_waist_target_reset` to waist_yaw only.

DR, command ranges, and all other rewards are kept identical to V1 so this
run isolates the effect of the obs+action change.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity import mdp as custom_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.hv1_velocity.flat_env_cfg import (
    ARM_JOINT_NAMES,
    LEG_JOINTS,
)

from .loco_manip_env_cfg import (
    HV1LocoManipActionsCfg,
    HV1LocoManipEnvCfg,
    HV1LocoManipObservationsCfg,
    HV1LocoManipRewardsCfg,
)


# Torso link in the HV1 URDF — verified in hv1.xml.
TORSO_BODY = "torso_link"


# ---- actions: legs + arms + waist_roll + waist_pitch (28 joints) ---------
# waist_yaw is held by PD at the default (0 rad) via `pin_waist_yaw_target_reset`.
WAIST_ACTUATED_JOINTS = ["waist_roll_joint", "waist_pitch_joint"]


@configclass
class HV1LocoManipV2ActionsCfg(HV1LocoManipActionsCfg):
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS + ARM_JOINT_NAMES + WAIST_ACTUATED_JOINTS,
        scale=0.25,
        use_default_offset=True,
    )


# ---- observations: asymmetric actor / critic -----------------------------
@configclass
class HV1LocoManipV2ObservationsCfg:
    """Two obs groups for RSL-RL asymmetric actor-critic.

    The runner cfg maps  actor ← policy ,  critic ← critic .
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # Actor obs — only what the real robot can measure from a single
        # torso IMU + joint encoders + command bus.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        projected_gravity_torso = ObsTerm(
            func=custom_mdp.projected_gravity_body,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TORSO_BODY)},
        )
        torso_ang_vel = ObsTerm(
            func=custom_mdp.body_ang_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TORSO_BODY)},
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        left_ee_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_ee_pose"}
        )
        right_ee_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_ee_pose"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # Critic obs — same as actor + privileged base_lin_vel.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        projected_gravity_torso = ObsTerm(
            func=custom_mdp.projected_gravity_body,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TORSO_BODY)},
        )
        torso_ang_vel = ObsTerm(
            func=custom_mdp.body_ang_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=TORSO_BODY)},
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        left_ee_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "left_ee_pose"}
        )
        right_ee_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "right_ee_pose"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---- rewards: V1 + waist + torso shaping ---------------------------------
@configclass
class HV1LocoManipV2RewardsCfg(HV1LocoManipRewardsCfg):
    # Split the waist penalty: yaw is the cheat lever (whole upper body spins
    # for free yaw tracking), roll/pitch are mildly useful for reach.
    # Earlier V2 run had a single -0.1 term — too weak vs +2.0 track_ang_vel_z.
    # waist_yaw is PD-pinned (not in action space) — no deviation reward needed.
    # The roll/pitch term keeps the actuated waist from twisting torso aggressively.
    joint_deviation_waist_rp = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["waist_roll_joint", "waist_pitch_joint"]
            )
        },
    )
    # Built-in flat_orientation_l2 / ang_vel_xy_l2 read pelvis only. With the
    # waist now actuated the torso is decoupled and can pitch back to "swing"
    # the arms toward high EE targets — this term constrains that.
    torso_flat_orientation = RewTerm(
        func=custom_mdp.flat_orientation_l2_body,
        weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=TORSO_BODY)},
    )
    torso_ang_vel_xy = RewTerm(
        func=custom_mdp.body_ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=TORSO_BODY)},
    )


# ---- env config ----------------------------------------------------------
@configclass
class HV1LocoManipV2EnvCfg(HV1LocoManipEnvCfg):
    actions: HV1LocoManipV2ActionsCfg = HV1LocoManipV2ActionsCfg()
    observations: HV1LocoManipV2ObservationsCfg = HV1LocoManipV2ObservationsCfg()
    rewards: HV1LocoManipV2RewardsCfg = HV1LocoManipV2RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # waist_yaw stays PD-pinned at 0; waist_roll/pitch are in action space.
        # The inherited event was pinning all 3 waist joints — narrow it to yaw.
        self.events.pin_waist_target_reset.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist_yaw_joint"], preserve_order=True
        )
        self.events.pin_waist_target_reset.params["position_range"] = {
            "waist_yaw_joint": (0.0, 0.0)
        }


@configclass
class HV1LocoManipV2EnvCfg_PLAY(HV1LocoManipV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        # Mix walking + reaching for visible playback.
        self.commands.base_velocity.rel_standing_envs = 0.30
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.resampling_time_range = (8.0, 8.0)

        self.commands.left_ee_pose.resampling_time_range = (4.0, 4.0)
        self.commands.right_ee_pose.resampling_time_range = (4.0, 4.0)

        # Visible disturbances during play.
        self.events.push_robot.interval_range_s = (3.0, 5.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}
        }
        self.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
        self.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)
