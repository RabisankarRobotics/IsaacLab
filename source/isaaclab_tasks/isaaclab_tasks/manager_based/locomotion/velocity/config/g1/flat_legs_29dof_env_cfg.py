# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-ground omnidirectional velocity locomotion for the Unitree G1 29-DOF model.

Differences from the stock ``Isaac-Velocity-Flat-G1-v0`` task:

* Uses ``G1_29DOF_CFG`` (legs + waist + arms + wrists, no 3-finger ``_zero.._six``
  hand joints used by the minimal model) instead of ``G1_MINIMAL_CFG``.
* The action space drives **legs only** (hip yaw/roll/pitch, knee, ankle
  pitch/roll = 12 joints). The waist/arm/wrist/finger joints are held at their
  default pose by their stiff implicit actuators and are not part of the policy
  output.
* Velocity command ranges are widened to full omnidirectional walking
  (forward/back, left/right strafe, and turn in place).
* **Deploy-ready asymmetric actor-critic.** The actor (``policy`` group) sees
  only quantities a real G1 can measure on hardware — IMU gyro
  (``base_ang_vel``), IMU orientation (``projected_gravity``), the velocity
  command, the **12 leg** joint encoders (position + velocity), and the last
  action. The unmeasurable ``base_lin_vel`` lives only in a privileged
  ``critic`` group used during training. There is **no observation history**
  and **no observation of the fixed waist/arm/wrist/finger joints** (they are
  held at their default pose and carry no information). The trained actor is
  therefore directly deployable from IMU + the 12 leg encoders.
"""

import torch

import isaaclab.utils.math as math_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ...velocity_env_cfg import ObservationsCfg as VelocityObservationsCfg
from .rough_env_cfg import G1Rewards, G1RoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_29DOF_CFG  # isort: skip


# Joints driven by the policy (legs only).
LEG_JOINT_NAMES = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]

# Arm joints — owned by a separate manipulation policy at deploy time. During
# locomotion training they are the disturbance source we randomize.
ARM_JOINT_NAMES = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]

# Waist joints — also upper-body, randomized over a smaller range (waist motion
# shifts the whole torso, so a little goes a long way).
WAIST_JOINT_NAMES = ["waist_.*_joint"]

# Upper-body joints the *walker* observes (arm-aware locomotion). Fingers are
# excluded — negligible mass, no meaningful CoM effect.
UPPER_BODY_JOINT_NAMES = WAIST_JOINT_NAMES + ARM_JOINT_NAMES


def randomize_and_hold_joint_pose(
    env,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    set_state: bool = True,
):
    """Drive a set of (non-actioned) joints to a random pose and hold them there.

    Used to make the legs-only walker robust to arbitrary upper-body
    configurations. These joints are not part of any action term, so the only
    way to move/hold them is to write their PD *target*:

    * ``set_state=True``  (reset): teleport the joints to the sampled pose *and*
      set the target — the episode starts with the arms already parked.
    * ``set_state=False`` (interval): only set a new target, so the stiff
      actuators drive the joints there smoothly mid-episode. This produces the
      reaction torques / momentum the legs must reject — i.e. it emulates a
      manipulation policy actively moving the arms while walking.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # sample a target pose around the default
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

    # set the PD target so the stiff actuators hold/drive the joints to this pose
    asset.set_joint_position_target(joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)

    if set_state:
        zero_vel = torch.zeros_like(joint_pos)
        asset.write_joint_state_to_sim(joint_pos, zero_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


@configclass
class G1Legs29DofActionsCfg:
    """Action specifications for the legs-only G1 29-DOF MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINT_NAMES, scale=0.5, use_default_offset=True
    )


@configclass
class G1Legs29DofObservationsCfg(VelocityObservationsCfg):
    """Asymmetric, deploy-ready observation groups.

    * ``policy`` (actor): only hardware-measurable terms — IMU gyro, IMU tilt,
      the command, the **12 leg** encoders, and the last action. No
      ``base_lin_vel`` (not measurable on hardware) and no history.
    * ``critic``: same terms **plus** the privileged simulator ``base_lin_vel``.

    Both groups observe only the leg joints; the fixed waist/arm/wrist/finger
    joints are held at default and contribute no observation.
    """

    @configclass
    class PolicyCfg(VelocityObservationsCfg.PolicyCfg):
        # Leg encoders (the actuated joints).
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        # Arm-aware: the walker observes the *live* upper-body joint state so it
        # can anticipate the CoM / momentum from arm + waist motion instead of
        # blindly rejecting it. At deploy, feed real arm/waist encoders here
        # (≈0 when walking standalone with arms parked at zero).
        upper_body_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        upper_body_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINT_NAMES)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        def __post_init__(self):
            super().__post_init__()
            # base_lin_vel is not measurable on the real robot — drop it from
            # the actor. The critic re-adds it as privileged information.
            self.base_lin_vel = None

    @configclass
    class CriticCfg(PolicyCfg):
        # Privileged group: same terms as the actor plus the simulator's true
        # base linear velocity.
        def __post_init__(self):
            super().__post_init__()
            self.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1Legs29DofRewards(G1Rewards):
    """Reward terms adapted to the 29-DOF joint naming.

    The stock :class:`G1Rewards` deviation terms reference minimal-model joint
    names (``*_elbow_pitch_joint``, ``*_elbow_roll_joint``, ``torso_joint``,
    ``*_zero.._six_joint``) that do not exist on the 29-DOF model, so they are
    re-expressed here against names defined by ``G1_29DOF_CFG``.
    """

    # Penalize deviation of the non-actuated upper-body joints from default so a
    # stray PD transient does not leave the arms/waist in an odd posture.
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )


@configclass
class G1FlatLegs29DofEnvCfg(G1RoughEnvCfg):
    rewards: G1Legs29DofRewards = G1Legs29DofRewards()
    actions: G1Legs29DofActionsCfg = G1Legs29DofActionsCfg()
    observations: G1Legs29DofObservationsCfg = G1Legs29DofObservationsCfg()

    def __post_init__(self):
        # post init of parent (G1RoughEnvCfg) — sets up the rough velocity env
        super().__post_init__()

        # Use the 29-DOF model instead of the minimal model.
        self.scene.robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # G1_29DOF_CFG ships with contact sensors disabled (it targets
        # manipulation tasks). The velocity env attaches a foot contact sensor,
        # so enable the contact reporter API on the robot bodies.
        self.scene.robot.spawn.activate_contact_sensors = True

        # The minimal-model finger deviation term has no 29-DOF counterpart.
        self.rewards.joint_deviation_fingers = None
        # The arms/waist are now a deliberately randomized disturbance, not
        # joints the policy controls — penalizing their deviation from default
        # would fight the randomization with reward noise the legs can't act on.
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_torso = None

        # ----- Flatten the terrain (mirror of G1FlatEnvCfg) -----
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan (flat ground) — drop from both observation groups
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # ----- Rewards (mirror of G1FlatEnvCfg tuning) -----
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )

        # ----- Commands: full omnidirectional walking -----
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ===== Upper-body domain randomization (loco-manip robustness) =====
        # At reset, park the arms in a random pose (forward / to the side / up,
        # etc.) so the legs learn to balance under any static upper-body CoM.
        self.events.randomize_arm_pose = EventTerm(
            func=randomize_and_hold_joint_pose,
            mode="reset",
            params={
                "position_range": (-1.0, 1.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
                "set_state": True,
            },
        )
        # Waist gets a smaller range — it swings the whole torso.
        self.events.randomize_waist_pose = EventTerm(
            func=randomize_and_hold_joint_pose,
            mode="reset",
            params={
                "position_range": (-0.25, 0.25),
                "asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINT_NAMES),
                "set_state": True,
            },
        )
        # Mid-episode: re-target the arms every few seconds so they *move* while
        # walking, generating the reaction torques / momentum a manipulation
        # policy would. set_state=False → the PD drives them there smoothly.
        self.events.move_arms = EventTerm(
            func=randomize_and_hold_joint_pose,
            mode="interval",
            interval_range_s=(2.0, 4.0),
            params={
                "position_range": (-1.0, 1.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
                "set_state": False,
            },
        )

        # ===== Base disturbance DR (off in the stock G1 rough cfg) =====
        # Payload mass + CoM shift on the torso (the upper body carries a moving
        # arm/tool), plus random pushes for general disturbance rejection.
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "mass_distribution_params": (-3.0, 3.0),
                "operation": "add",
            },
        )
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )


@configclass
class G1FlatLegs29DofEnvCfg_PLAY(G1FlatLegs29DofEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # standalone-walking demo: keep the arms parked at default (zero), the
        # same condition as deploying the walker without a manipulation policy.
        self.events.randomize_arm_pose = None
        self.events.randomize_waist_pose = None
        self.events.move_arms = None
