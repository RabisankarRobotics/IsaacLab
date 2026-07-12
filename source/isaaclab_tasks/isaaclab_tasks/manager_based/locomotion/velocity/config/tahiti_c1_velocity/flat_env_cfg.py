"""Tahiti C1 velocity-tracking (walking) task on flat ground.

12 DoF bipedal robot (6 per leg). Policy actuates all 12 leg joints. No upper
body — the URDF has no arms / waist / head.

First-training defaults:
    * Mild domain randomization (±5 % motor gains / joint params, no persistent
      pelvis wrench, small reset velocity, moderate push_robot).
    * Three-phase stand→slow-walk→full-walk curriculum for a clean start under
      DelayedPD actuator lag.
    * Rewards shaped for realistic knee-bent gait: feet_air_time (threshold
      0.4 s), foot_clearance (target 0.10 m), knee_too_straight, base height
      floor, and stand-still deviation + base_ang_vel penalties so the robot
      actually freezes when commanded to stand.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventCfg,
    LocomotionVelocityRoughEnvCfg,
)

from isaaclab_assets import TAHITI_C1_CFG  # isort: skip

from . import mdp as custom_mdp


# ---- joint-name regexes -------------------------------------------------
LEG_JOINTS = [
    "^(left|right)_hip_(yaw|pitch|roll)_joint$",
    "^(left|right)_knee_joint$",
    "^(left|right)_ankle_(pitch|roll)_joint$",
]


@configclass
class TahitiC1VelocityActionsCfg:
    """Policy actions on all 12 leg joints.

    Scale 0.25 for a conservative first training. Smaller than the 0.5 that
    HV1.2 uses because Tahiti C1 is ~40 % lighter and has no arm inertia — a
    given raw action produces a bigger joint displacement, so we shrink the
    mapping to keep the initial swing smooth. Once a clean gait is converged,
    can bump to 0.5 in a second-stage refinement run.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class TahitiC1VelocityObservationsCfg:
    """Observations restricted to what a real Tahiti C1 can sense:
    IMU (base_ang_vel + projected_gravity) and joint encoders. No base_lin_vel
    — that requires a state estimator not present on hardware.
    Per-term Unoise mimics sensor noise so the policy is robust at deploy.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.3, n_max=0.3))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-2.0, n_max=2.0))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TahitiC1VelocityEventCfg(EventCfg):
    """Aggressive sim-to-real DR for the robustness refinement run.

    ±15 % on Kp/Kd and armature/friction (Berkeley uses ±20 %). ±0.05 rad
    per-joint encoder zero-offset via randomize_joint_default_pos — the
    highest-value DR term for the arc/drift symptoms seen on hardware.
    """

    actuator_gains_randomize = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_params_randomize = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_distribution_params": (0.85, 1.15),
            "armature_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # Per-joint encoder zero-offset randomization. joint_pos_rel obs and the
    # JointPositionActionCfg (use_default_offset=True) both anchor on
    # default_joint_pos, so this shifts both the sensed zero and the commanded
    # zero for each env — matches real hardware where each motor's absolute
    # encoder is mounted with a small angular error from the URDF nominal.
    joint_default_pos_randomize = EventTerm(
        func=custom_mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )


@configclass
class TahitiC1VelocityRewardsCfg:
    """Velocity-tracking rewards shaped for realistic bent-knee walking."""

    # ---- tracking ------------------------------------------------------
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.7},
    )

    # ---- gait shaping (realistic knee swing) --------------------------
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            # 0.4 s → longer swing than HV1.2's 0.3 s. Tahiti C1 is lighter and
            # this gives visibly bigger knee swing / cleaner stride at the cost
            # of a slower cadence — matches the "realistic knee swing" ask.
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_airtime_variance = RewTerm(
        func=custom_mdp.air_time_variance_penalty,
        weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    foot_clearance = RewTerm(
        func=custom_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            # 0.15 m target → ankle_roll link sits ~1.3 cm below its origin,
            # so this yields ~13-14 cm of visible foot lift above the ground
            # during swing. Weight bumped 0.3 → 0.5 so the reward can meaningfully
            # pull the policy toward a taller swing arc without being swamped
            # by tracking / smoothness terms.
            "target_height": 0.15,
            "std": 0.05,
            "tanh_mult": 2.0,
        },
    )
    knee_too_straight = RewTerm(
        func=custom_mdp.knee_too_straight_penalty,
        weight=-0.5,
        params={
            # 0.35 rad is just under the 0.36 default — swing-phase knees
            # (>= 0.7 rad) pay 0, stance-phase knees at rest pay ~0, only
            # actively locked-straight knees (stilt walk) pay meaningful cost.
            "threshold": 0.35,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_knee_joint$"]),
        },
    )

    # ---- stability -----------------------------------------------------
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # Below-target base height only — free to stand tall. 0.85 m is 5 cm below
    # the settled ~0.90 m stance height, so normal walking pays 0, only real
    # crouching or falling registers.
    base_height_below = RewTerm(
        func=custom_mdp.base_height_below_target_l1,
        weight=-10.0,
        params={"target_height": 0.85},
    )

    # ---- effort / smoothness ------------------------------------------
    # Bumped 4-5× vs first-training defaults to kill the deploy-time jitter that
    # emerged after obs noise + DR were widened. Under wider DR the policy
    # tends to hedge with reactive action deltas; raising the price of every
    # delta forces it toward a smoother mapping.
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-6)

    # ---- safety --------------------------------------------------------
    is_alive = RewTerm(func=mdp.is_alive, weight=0.05)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_ankle_(pitch|roll)_joint$", ".*_hip_yaw_joint$"],
            ),
        },
    )

    # ---- posture shaping ----------------------------------------------
    # Keeps hip_yaw close to zero on average — an anti-drift wall. Kept small
    # (-0.1) so it can't fight the symmetry / turning terms.
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_hip_yaw_joint$"])},
    )
    # Symmetry-only hip_yaw penalty — zero when left/right mirror, non-zero
    # when both drift the same way (the "walks in a circle" symptom). Softened
    # during turn commands so it doesn't fight yaw tracking.
    # hip_yaw_lr_symmetry = RewTerm(
    #     func=custom_mdp.hip_yaw_symmetry_l1,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"],
    #             preserve_order=True,
    #         ),
    #     },
    # )
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_hip_roll_joint$"])},
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_ankle_roll_joint$"])},
    )

    # ---- standstill freeze --------------------------------------------
    # "Do not deviate any joint when the operator is not commanding motion."
    # Fires only when ||cmd_vel|| < 0.1; zero during walking. Pins all 12 leg
    # joints at their default pose so the robot visibly freezes rather than
    # micro-cycling the feet.
    stand_still_no_cmd = RewTerm(
        func=custom_mdp.stand_still_joint_deviation_l1,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["^(left|right)_(hip_yaw|hip_pitch|hip_roll|knee|ankle_pitch|ankle_roll)_joint$"],
            ),
        },
    )
    # Kill the standing sway directly — L2 on base_ang_vel gated to standstill.
    stand_still_base_ang_vel = RewTerm(
        func=custom_mdp.stand_still_base_ang_vel_l2,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )
    # Standstill-only action-rate penalty — the dedicated jitter killer.
    # At weight -0.5 this fires 10× harder than the always-on action_rate_l2
    # whenever the operator is not commanding motion, so the policy learns to
    # freeze the raw action vector at rest. Zero during any commanded walk.
    stand_still_action_rate = RewTerm(
        func=custom_mdp.stand_still_action_rate_l2,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )

    feet_contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-0.001,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 600.0,
        },
    )

    feet_lateral_clearance = RewTerm(
        func=custom_mdp.feet_lateral_distance_clearance,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "min_distance": 0.12,
        },
    )


@configclass
class TahitiC1VelocityFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: TahitiC1VelocityActionsCfg = TahitiC1VelocityActionsCfg()
    observations: TahitiC1VelocityObservationsCfg = TahitiC1VelocityObservationsCfg()
    rewards: TahitiC1VelocityRewardsCfg = TahitiC1VelocityRewardsCfg()
    events: TahitiC1VelocityEventCfg = TahitiC1VelocityEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # ---------------- scene: flat plane, no height scanner ---------------
        self.scene.robot = TAHITI_C1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # ---------------- 3-phase stand→walk command curriculum --------------
        self.curriculum.command_phase = CurrTerm(
            func=custom_mdp.stand_to_walk_command_curriculum,
            params={
                "stand_until_iters": 2000,
                "slow_until_iters": 5000,
                "slow_scale": 0.3,
                "lin_vel_x_full": (-1.0, 1.0),
                "lin_vel_y_full": (-0.5, 0.5),
                "ang_vel_z_full": (-1.0, 1.0),
                "rel_standing_envs_phase1": 1.0,
                "rel_standing_envs_phase2": 0.3,
                "rel_standing_envs_phase3": 0.1,
            },
        )

        # ---------------- commands: final (phase-3) ranges -------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.rel_standing_envs = 0.1

        # ---------------- domain randomization on the base link --------------
        # Root body is base_link (not "base" — the parent env cfg's default).
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        # Asymmetric mass DR: (-1, +3) kg around the ~13 kg base_link. Positive
        # bias reflects real hardware typically over CAD mass. Tighter than
        # HV1.2's (-2, +5) because Tahiti C1's base is 6× lighter — same
        # percentage envelope, absolute values scaled down.
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 5.0)

        # ±4 cm horizontal, ±2 cm vertical CoM offset on base_link.
        self.events.base_com.params["asset_cfg"].body_names = "base_link"
        self.events.base_com.params["com_range"] = {
            "x": (-0.04, 0.04),
            "y": (-0.04, 0.04),
            "z": (-0.02, 0.02),
        }

        # Persistent per-episode wrench on the base: ±2 N linear, ±2 N·m torque
        # (matches Berkeley). Simulates CoM misalignment + a small aero/cable
        # bias the robot must counter for the whole episode.
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.base_external_force_torque.params["force_range"] = (-2.0, 2.0)
        self.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)

        # Ground friction: static 0.5-1.0, dynamic 0.4-0.9. Narrower than
        # HV1.2's 0.4-1.2 / 0.3-1.0 for a milder first run.
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 0.9)

        # Reset joint pose scale (0.5, 1.5): each env spawns with all joints at
        # 50–150 % of default_joint_pos — forces the policy to recover from
        # off-nominal starting postures instead of overfitting to a clean pose.
        self.events.reset_robot_joints.params["position_range"] = (0.5, 1.5)
        # ±0.5 pos and ±0.5 vel on every axis (matches Berkeley). Trains real
        # push-recovery / random-init-state robustness.
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        }
        # Push every 12-15 s with ±1.0 m/s velocity impulse — Berkeley-strength
        # perturbation without stacking hits.
        self.events.push_robot.interval_range_s = (12.0, 15.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}}

        # ---------------- terminations: base_link contact only ---------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

        # ---------------- runtime --------------------------------------------
        self.episode_length_s = 20.0
        self.decimation = 4  # policy at 50 Hz with sim.dt = 0.005


@configclass
class TahitiC1VelocityFlatEnvCfg_PLAY(TahitiC1VelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Obs corruption OFF during play — flip to True to inspect noisy-obs
        # deployment conditions.
        self.observations.policy.enable_corruption = False
        # Push robot during play for visual push-recovery inspection.
        self.events.push_robot.interval_range_s = (6.0, 8.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}}
        # Disable curriculum in play (common_step_counter starts at 0, would
        # force Phase 1 and overwrite the play ranges).
        self.curriculum.command_phase = None
        # Spread envs across the full command space.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.rel_standing_envs = 0.2
