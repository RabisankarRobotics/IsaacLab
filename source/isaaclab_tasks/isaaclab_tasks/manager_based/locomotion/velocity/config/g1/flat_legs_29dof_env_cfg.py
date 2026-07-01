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
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ...velocity_env_cfg import ObservationsCfg as VelocityObservationsCfg
from . import mdp as custom_mdp
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

    # ===== Gait-shaping terms (ported from the tuned HV1.2 velocity task) =====

    # Symmetric stepping: penalize variance in L/R air- and contact-time so one
    # foot doesn't stay planted while the other does all the work.
    feet_airtime_variance = RewTerm(
        func=custom_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )

    # Crisp, visible swing-foot lift (clearer steps). Positive weight; only pays
    # out while the foot is moving, so it doesn't reward marching in place.
    foot_clearance = RewTerm(
        func=custom_mdp.foot_clearance_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "target_height": 0.07,
            "std": 0.05,
            "tanh_mult": 2.0,
        },
    )

    # Anti-lock only: G1's knee default is already 0.30 rad, so the threshold is
    # set *below* default (0.15) — it fires only when a knee approaches the
    # near-straight forward-step pose (knee min limit is 0.061 rad), and is zero
    # at the natural stance. This avoids forcing a permanent crouch. Light weight.
    knee_too_straight = RewTerm(
        func=custom_mdp.knee_too_straight_penalty,
        weight=-0.5,
        params={
            "threshold": 0.15,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
        },
    )

    # Maintain a minimum lateral gap between the feet (no crossing / scissoring),
    # measured in the yaw frame so forward stride length isn't penalized.
    # Measured natural lateral foot separation for G1 is ~0.236 m; min_distance
    # is set to ~half (0.12) so it only catches scissoring, not normal swing.
    feet_lateral_clearance = RewTerm(
        func=custom_mdp.feet_lateral_distance_clearance,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "min_distance": 0.12,
        },
    )

    # Pull the ankle-roll joints toward flat (0 rad) — fixes the feet rolling
    # inward / walking on the inner edges.
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])},
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
        # Yaw tracking was the weak dimension AND the cause of "veers when given
        # only vx": with weak yaw tracking the policy lets a small yaw rate creep
        # in. Bump well above stock to firmly hold heading on straight commands.
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        # Longer, more deliberate steps (fixes the "small stepping"): reward more
        # air time and require a slightly longer swing before it pays out.
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.45
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )

        # ===== Walk-tall / natural-gait fixes =====
        # (1) Anchor the pelvis AT its natural standing height (~0.784 m at the
        #     default pose). A target BELOW natural (the old 0.76) literally
        #     rewards crouching — it pulls the pelvis down, which forces a bent
        #     knee. Set it at natural height so the height reward stops paying for
        #     the crouch. The stance-knee term (3) then does the active
        #     straightening. Raise toward 0.80 for an even taller gait, lower if
        #     it looks stiff/bouncy or walks on tiptoe.
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-25.0,
            params={"target_height": 0.78},
        )
        # (2) Drop the knee-bend-forcing penalty. It was added to stop a
        #     stiff-legged lock, but it pushes the knee toward MORE bend, which
        #     is exactly the over-crouch you see. The base-height anchor above
        #     now prevents collapse, so this is no longer needed.
        self.rewards.knee_too_straight = None
        # (2b) LESS KNEE BEND: extend the STANCE knee. The crouch walk keeps the
        #      knee flexed the whole cycle and steps from the hip; this penalizes
        #      a loaded (foot-on-ground) knee bent past ~0.25 rad, pulling the
        #      stance leg tall and natural. It is contact-gated, so the SWING
        #      knee is untouched and still flexes to clear the ground. Feet and
        #      knees are passed as explicit L,R lists with preserve_order=True so
        #      each foot gates its own knee. Raise threshold toward 0.30 (the
        #      default knee angle) or lower the weight if the knee locks /
        #      hyperextends; strengthen the weight if it still crouches.
        self.rewards.knee_stance_extension = RewTerm(
            func=custom_mdp.knee_bent_stance_penalty,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                    preserve_order=True,
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["left_knee_joint", "right_knee_joint"],
                    preserve_order=True,
                ),
                "threshold": 0.25,
            },
        )
        # (3) Stronger L/R cadence symmetry (helps straight-line walking and a
        #     symmetric step pattern).
        self.rewards.feet_airtime_variance.weight = -1.5
        # (4) STAND STILL at zero command. The velocity-tracking reward gives no
        #     incentive to actually stop, so the gait limit-cycle keeps running
        #     and the robot shuffles / drifts in place. Two complementary terms,
        #     both full-command gated (turning in place is never penalized):
        #
        #   (4a) POSITION: pull the legs back to the default stance. This is what
        #        stops the feet slowly creeping apart while idle ("legs
        #        separating"). Strengthened -1.0 -> -2.0 because the drift means
        #        the previous weight lost to the residual gait limit-cycle.
        self.rewards.stand_still = RewTerm(
            func=custom_mdp.stand_still_joint_deviation_l1,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
                "command_threshold": 0.1,
            },
        )
        #   (4b) VELOCITY: penalize leg-joint motion while idle. The position
        #        term alone lets the legs oscillate around the default (mean
        #        position ~correct but still moving) — that is the standing
        #        "vibration". Penalizing joint velocity forces the legs to go
        #        quiet. Start moderate; raise toward -1.0 if it still trembles.
        self.rewards.stand_still_vel = RewTerm(
            func=custom_mdp.stand_still_joint_vel_l1,
            weight=-0.5,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
                "command_threshold": 0.1,
            },
        )
        # Actually TRAIN standing. 5% of envs was too little signal for a distinct
        # skill, so at exactly-zero command the policy was near out-of-distribution
        # (hence the vibration/drift). 15% makes standing a first-class behavior.
        self.commands.base_velocity.rel_standing_envs = 0.15

        # ----- Terminations: stop the "sit and survive" local optimum -----
        # The stock config only terminates on torso_link ground contact. The
        # legs-only walker can collapse into an L-shape (legs flat on the floor,
        # pelvis upright) where the torso never touches the ground, so the
        # episode never ends and the policy farms reward while lying down.
        # Terminate when the pelvis drops well below its 0.75 m nominal height
        # (collapsed) or when the base tilts too far over.
        # root_height_below_minimum reads the root (pelvis) world height.
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.5},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 1.0},  # ~57 deg of base tilt
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
                # Was ±1.0 rad (≈±57°). That full-range static offset, together
                # with the mid-episode flailing below, made a defensive crouch
                # the easy way to stay balanced. ±0.8 still covers a wide held
                # arm posture without forcing the legs into a permanent crouch.
                "position_range": (-0.8, 0.8),
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
            # Slower retarget (was 2-4 s) and smaller range (was ±1.0) so the
            # arms emulate a manipulation policy's smoother motion rather than
            # violent flailing — enough reaction torque to harden the legs, not
            # so much that crouching becomes the only way to survive.
            interval_range_s=(3.0, 5.0),
            params={
                "position_range": (-0.6, 0.6),
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
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
        # # standalone-walking demo: keep the arms parked at default (zero), the
        # # same condition as deploying the walker without a manipulation policy.
        # self.events.randomize_arm_pose = None
        # self.events.randomize_waist_pose = None
        # self.events.move_arms = None
