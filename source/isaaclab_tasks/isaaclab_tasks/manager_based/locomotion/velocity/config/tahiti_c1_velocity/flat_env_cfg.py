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

    Per-joint scale (2026-07-17) to match robo_control deploy base yaml:
      ankle_roll (L, R): 0.10  — reduced authority damps ankle-roll vibration
      all other joints:  0.25
    Any change here must be mirrored in deploy/config/policy.yaml.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        # Patterns must be mutually exclusive — Isaac Lab's resolver rejects
        # a joint name that matches two patterns. Enumerate the non-ankle-roll
        # joints instead of using a ".*" catchall.
        scale={
            ".*_hip_yaw_joint": 0.25,
            ".*_hip_pitch_joint": 0.25,
            ".*_hip_roll_joint": 0.25,
            ".*_knee_joint": 0.25,
            ".*_ankle_pitch_joint": 0.25,
            ".*_ankle_roll_joint": 0.10,
        },
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
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # 5-step obs history (2026-07-17). Matches robo_control deploy base
            # yaml. Gives the policy temporal context so noisy single-step obs
            # averages out — main smoothness lever for sim-to-real. Obs dim per
            # step = 45; total policy input = 45 × 5 = 225.
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()


@configclass
class TahitiC1VelocityEventCfg(EventCfg):
    """Mild sim-to-real motor DR for the first training run.

    ±5 % on Kp/Kd and armature/friction. Halved again vs the HV1.2 first-run
    settings (±10 %) because Tahiti C1 has fewer joints and no upper-body
    compensation DoFs — the policy has less bandwidth to hedge against DR.
    Ramp to ±10 % / ±20 % in a second-stage refinement once a converged
    baseline exists.
    """

    actuator_gains_randomize = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_params_randomize = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_distribution_params": (0.95, 1.05),
            "armature_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
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
        params={"command_name": "base_velocity", "std": 0.5},
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
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    foot_clearance = RewTerm(
        func=custom_mdp.foot_clearance_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            # 0.13 m → larger clearance than HV1.2's 0.07 m. Tahiti C1's ankle
            # roll link sits ~1.3 cm below its origin, so this yields ~8-9 cm of
            # visible foot lift above the ground during swing — the "realistic
            # knee swing" gait.
            "target_height": 0.13,
            "std": 0.05,
            "tanh_mult": 2.0,
        },
    )
    knee_too_straight = RewTerm(
        func=custom_mdp.knee_too_straight_penalty,
        weight=-0.5,
        params={
            # 0.29 rad is just under the 0.30 default (updated 2026-07-17) —
            # swing-phase knees (>= 0.7 rad) pay 0, stance knees at rest pay
            # ~0, only actively locked-straight knees pay meaningful cost.
            "threshold": 0.29,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_knee_joint$"]),
        },
    )
    # Anti-toe-in / anti-foot-crossover. Fires only when the two feet get
    # laterally closer than min_distance in the yaw frame (measures actual
    # geometry, not command). Small weight because the natural stance already
    # sits well above 0.12 m — this term is insurance against turn-in-place
    # cheat gaits, not an active shaper.
    feet_lateral_clearance = RewTerm(
        func=custom_mdp.feet_lateral_distance_clearance,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            # Bumped 2026-07-17: 0.22 → 0.26 m demanded lateral separation, and
            # weight -5 → -8 for sharper penalty. Feet-crossing was the visible
            # corner-walk symptom in MuJoCo Test B — widen the natural stance
            # so the two feet cannot pass under the base.
            "min_distance": 0.26,
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
    # Bumped 2026-07-16 to shape a smoother gait — real deploy amplifies any
    # jitter seen in sim, and the policy also slams feet on touchdown. Both
    # symptoms have the same fix (smoother action → smoother joint acc →
    # softer footfall). dof_acc 2.5e-7 → 4e-7 (60%), action_rate 0.01 → 0.015
    # (50%). Combined delta ≈ -0.25 mean reward. Small enough to stay safely
    # inside sum-of-deltas tolerance on hot resume.
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-4.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.015)

    # ---- safety --------------------------------------------------------
    is_alive = RewTerm(func=mdp.is_alive, weight=0.05)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
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
        weight=-0.7,
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
        weight=-0.7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_hip_roll_joint$"])},
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_ankle_roll_joint$"])},
    )

    # ---- foot-flat stance (kills toe-up bias / heel-only pathology) ----
    # Gated to feet-in-contact. When a foot is in contact, its ankle_pitch
    # should sit at the default (+0.20 in the new pose = flat foot). Any
    # deviation (toe pumped higher or lower) pays L1 cost.
    feet_stance_flat_ankle = RewTerm(
        func=custom_mdp.feet_stance_flat_ankle,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_ankle_pitch_joint", "right_ankle_pitch_joint"],
                preserve_order=True,
            ),
            "force_threshold": 5.0,
        },
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
                "lin_vel_x_full": (-0.8, 0.8),
                "lin_vel_y_full": (-0.5, 0.5),
                "ang_vel_z_full": (-0.5, 0.5),
                "rel_standing_envs_phase1": 1.0,
                "rel_standing_envs_phase2": 0.3,
                "rel_standing_envs_phase3": 0.1,
            },
        )

        # ---------------- commands: final (phase-3) ranges -------------------
        # lin_vel_x narrowed 1.0 → 0.8 (2026-07-16) — user asked to cap forward
        # / backward at 0.8 m/s to help the policy find a smoother stride at
        # deploy-realistic speeds. Y and yaw ranges unchanged.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        self.commands.base_velocity.rel_standing_envs = 0.1

        # ---------------- domain randomization on the base link --------------
        # Root body is base_link (not "base" — the parent env cfg's default).
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        # Asymmetric mass DR: (-1, +3) kg around the ~13 kg base_link. Positive
        # bias reflects real hardware typically over CAD mass. Tighter than
        # HV1.2's (-2, +5) because Tahiti C1's base is 6× lighter — same
        # percentage envelope, absolute values scaled down.
        # Restored (-2, +5) 2026-07-16: real Tahiti C1 measured +5 kg heavier
        # than URDF. The Jul-11 policy that walked cleanly on real hardware
        # was trained with this wider range; narrowing it to (-1, +3) is one
        # of the causes of the current real-robot vibration.
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 6.0)

        # ±4 cm horizontal, ±1 cm vertical CoM offset on base_link. Bumped
        # from ±2 cm / ±0.5 cm (2026-07-15) after observing that backward
        # walk struggles because the policy has no CoM-shift robustness —
        # real robot's CoM sits off-nominal from cables/electronics and any
        # offset compounds the already-marginal heel-loaded backward stance.
        self.events.base_com.params["asset_cfg"].body_names = "base_link"
        self.events.base_com.params["com_range"] = {
            "x": (-0.04, 0.04),
            "y": (-0.04, 0.04),
            "z": (-0.01, 0.01),
        }

        # First-training: NO persistent world-frame wrench on the base. This
        # is the single most "directional" DR effect (each env must produce a
        # constant counter-torque for its randomly-sampled wrench for the whole
        # episode), and it's the reason to introduce it later once a clean
        # baseline exists. Setting force/torque range to (0,0) effectively
        # disables the term while keeping the event registered — easy to
        # re-enable in a second-stage refinement.
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        # Restored 2026-07-16: real robot has persistent wrench from cable
        # drag, off-CoM electronics, and the +5 kg mass surplus. The Jul-11
        # policy trained WITH ±2 N force / ±2 N-m torque wrench held steady
        # on real hardware; disabling this term (0, 0) removed that
        # robustness and is the second cause of real-robot vibration.
        self.events.base_external_force_torque.params["force_range"] = (-2.0, 2.0)
        self.events.base_external_force_torque.params["torque_range"] = (-2.0, 2.0)

        # Ground friction: static 0.5-1.0, dynamic 0.4-0.9. Narrower than
        # HV1.2's 0.4-1.2 / 0.3-1.0 for a milder first run.
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 0.9)

        # Spawn at exactly the default joint pose (no random scale) so all envs
        # start from the same clean stance during Phase 1.
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # Reset velocity noise bumped ±0.1 → ±0.3 (2026-07-15) so every episode
        # starts mid-perturbation, not from near-still. Forces the policy to
        # build a recovery reflex, not just a start-from-still reflex —
        # complements the push_robot bump and CoM DR for backward-walk balance.
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.3, 0.3),
                "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.3, 0.3),
            },
        }
        # Round C (2026-07-15): stronger push training to fix -0.5 m/s backward
        # walk fall in MuJoCo. A -0.8 m/s x-push is mechanically identical to
        # walking backward at 0.8 m/s — every pushed episode drills the exact
        # heel-loaded backward-recovery skill that was capping around -0.3 m/s.
        # Interval tightened 12-15 s → 7-10 s so pushes are more frequent per
        # episode, more gradient signal per iteration.
        self.events.push_robot.interval_range_s = (7.0, 10.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}}

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
        self.events.push_robot.params = {"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}
        # Disable curriculum in play (common_step_counter starts at 0, would
        # force Phase 1 and overwrite the play ranges).
        self.curriculum.command_phase = None
        # Spread envs across the full command space.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.rel_standing_envs = 0.2
