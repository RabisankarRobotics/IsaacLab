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

    2026-07-28 Round Vibration (V1): switched to NoisyJointPositionActionCfg
    to inject ±0.02 rad (~1.15°) uniform noise on the joint pos target every
    step. Attacks the sim-to-real vibration gap: real robot jitters during
    walking from PID inner-loop chatter + gearbox backlash + comm timing —
    none of which Isaac/MuJoCo simulate. Training with target-level noise
    teaches the policy to be robust to those disturbances rather than
    adding another reward smoothness term (which would trade vibration for
    weaker gait authority). Deploy-side sees no change; the actuator noise
    is training-only.
    """

    joint_pos = custom_mdp.NoisyJointPositionActionCfg(
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
        noise_range=(-0.02, 0.02),
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
        # Obs noise bumped 2026-07-17/18 to close the sim-to-real gap. Real
        # robot showed 2-3x more jitter than mujoco even on the smoother V3
        # policy — noisier training obs should teach the policy to filter noise.
        # base_ang_vel  ±0.2 → ±0.3  rad/s   (real IMU under vibration)
        # joint_pos     ±0.01 → ±0.05  rad   (2026-07-18 doubled: encoder +
        #                                     backlash under load; ~2.9° 1-σ
        #                                     matches worst-case measured
        #                                     Tahiti hardware backlash)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.3, n_max=0.3))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.03, n_max=0.03))
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

    # 2026-07-18: ±5 % → ±10 % on Kp/Kd/friction/armature. Each env samples
    # INDEPENDENT scales per joint (verified _randomize_prop_by_op uses tensor
    # shape [num_envs, num_joints]), so widening the range widens the L/R
    # spread every env sees — the intended attack on structural gait asymmetry
    # observed in mujoco V3@30k (L longer air-time forward, R deeper knee
    # backward). ±10 % is still comfortably below the ±20 % ceiling used on
    # HV1.2 after its baseline converged.
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
    # 2026-07-22 Round Big-Stride Part 4 (P5): xy tracking weight 2.0 → 2.5.
    # Prior run at 2.0 left error_vel_xy = 0.41 (actual undershoots cmd by
    # ~50 % of max cmd 0.8 m/s). Bigger step needs bigger forward payoff —
    # air_time and foot_clearance don't reward forward displacement, only
    # velocity tracking does. Yaw kept at 2.0 (already at error_vel_yaw 0.35,
    # working fine).
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": 0.3},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.3},
    )

    # ---- gait shaping (realistic knee swing) --------------------------
    # 2026-07-21 Round Big-Stride Part 3: 1.5 → 2.0. Prior 1.5 got MuJoCo
    # air_times only to 0.15-0.18s (log reward 0.069, threshold cap 0.5s so
    # lots of headroom). The reason we halved from 3.0 was over-vertical
    # stomping, but now foot_contact_force (700 N cap, weight -1e-3) puts
    # a real ceiling on vertical push. With that ceiling in place, air_time
    # weight can climb again without producing 5× BW landings — the two
    # rewards constrain different axes (temporal vs force).
    # 2026-07-28 Round Long-Stride (Option 1) REVERTED. Bumped weight 2.5→3.5
    # and threshold 0.4→0.55 to reward longer single-stance duration.
    # Regression on every axis vs prior run: L peak force 3.87→4.33×BW, air-
    # time asym 13%→18%, knee_amp reached 1.41 rad (81°) — classic vertical
    # parade. Root cause: feet_air_time is a SCALAR duration term with no
    # direction bias. Making the max-per-step payoff larger and the
    # saturation later gave the policy more room to farm it VERTICALLY
    # (knee-driven lift keeps foot off ground longer with less base motion
    # than a real forward push, which is metabolically cheap for the policy
    # to learn first). foot_clearance target + foot_contact_velocity did NOT
    # stop it — those cap the peak height and the touchdown speed, not the
    # knee flexion during swing. Reverted to 2.5 / 0.4. Horizontal-stride
    # push must come from a DIRECTIONAL term (Option 2: hip_pitch_swing
    # amplitude, or a base-forward-progress-per-step reward) — attempting
    # that only after this revert lands.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
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
    # 2026-07-28 Round Long-Stride (Option 2): directional stride-amplitude
    # reward. Pays |L_hip_pitch - R_hip_pitch| capped at 0.6 rad — a real
    # walking gait keeps hips opposed (one forward, one back), so this
    # differential tracks stride length directly. The vertical-parade
    # failure mode co-flexes both hips to lift the knees, driving the
    # differential toward zero, so this term explicitly punishes parade
    # shape. Gated by cmd>0.1 m/s AND at least one foot currently airborne
    # (>50 ms) — cannot be farmed by standing hip-flex or static split
    # stance. Weight 0.5 is conservative: max per-step raw = 0.6, so
    # per-episode ceiling with 70 % moving, 50 % swinging, 1000 steps ~ a
    # few tenths, comparable to feet_air_time. Bump to 0.75-1.0 if the
    # stride doesn't lengthen after 10k iters and other terms are healthy.
    hip_pitch_swing = RewTerm(
        func=custom_mdp.hip_pitch_swing_amplitude,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"],
                preserve_order=True,
            ),
        },
    )
    # 2026-07-18 Round HW: -3.0 → -5.0. V4@20k showed limping walk (one leg
    # dominant swing) which is exactly what variance measures. Earlier -4.0
    # bump caused thrashing because dof_acc/action_rate were also at max —
    # now that those are relaxed, the variance penalty can drive symmetry
    # without forcing a policy collapse. Attacks the L/R rolling-window gap
    # at cycle scale, not instantaneous.
    # 2026-07-28: reverted from -15 back to -5. The -15 experiment (Option A)
    # made the variance term dominant enough that the CHEAPEST way for the
    # policy to zero it out was to take tiny symmetric steps (air time ~0,
    # variance ~0). Killed stride length: feet_air_time reward dropped from
    # ~0.5 to 0.14 (14 % of max). Lesson: variance-only pressure has a
    # degenerate optimum at "no gait" — must be paired with a MIN air-time
    # counter-force before bumping again. Next attempt (if the limp comes
    # back at -5) will bump `feet_air_time` weight in parallel, or add a
    # stride-length reward that variance can't game.
    feet_airtime_variance = RewTerm(
        func=custom_mdp.air_time_variance_penalty,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    foot_clearance = RewTerm(
        func=custom_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            # 2026-07-22 Round Big-Stride Part 4 (P4): 0.09 → 0.06 m. Prior
            # 0.09 still let the policy earn reward on vertical lift; MuJoCo
            # showed high knees but short forward stride. 0.06 m is at the
            # low end of the human range (5-8 cm), so vertical lift saturates
            # early and the remaining swing energy has to go forward via hip
            # pitch — direct pressure toward longer strides.
            "target_height": 0.06,
            "std": 0.05,
            "tanh_mult": 2.0,
        },
    )
    # knee_too_straight = RewTerm(
    #     func=custom_mdp.knee_too_straight_penalty,
    #     weight=-0.5,
    #     params={
    #         # 2026-07-18 Round HW: 0.29 → 0.10 rad. THE key lever for human-
    #         # like walk. At 0.29 rad (17°) the penalty fired every time either
    #         # knee tried to extend during late swing, forcing the parade-
    #         # marching gait the user saw in V4. Real human late-swing extends
    #         # the knee to ~0.05-0.15 rad (heel-strike-ready). New threshold
    #         # 0.10 rad (~6°) means only truly hyperextended knees pay,
    #         # allowing the late-swing reach that produces a real stride.
    #         "threshold": 0.10,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_knee_joint$"]),
    #     },
    # )
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
    # 2026-07-18: -3.5 → -3.0 (from Bundle B's -4). V3@30k backward base_z
    # bottomed at 0.911 (target 0.910), so -3.5 was doing its job but the same
    # over-penalization signature (action_std ↑, tracking ↓) affects this. Trim
    # 15 % back — pelvis lean stays under control without dominating.
    # 2026-07-28: reverted from -2 back to -3. The -2 experiment (W3) let the
    # policy discover a "shopping-cart" forward-lean gait — lean forward, drag
    # feet in tiny steps, cash tracking reward without needing real stride
    # (this compounded with the Option A variance bump). A symmetric relaxation
    # affects both directions and the forward direction wins the local search.
    # For backward-walk stability, a DIRECTIONAL orientation term (soft on
    # rearward tilt only) is the correct approach, not a symmetric weight cut.
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    # Below-target base height only — free to stand tall. 0.85 m is 5 cm below
    # the settled ~0.90 m stance height, so normal walking pays 0, only real
    # crouching or falling registers.
    base_height_below = RewTerm(
        func=custom_mdp.base_height_below_target_l1,
        weight=-10.0,
        params={"target_height": 0.85},
    )

    # ---- effort / smoothness ------------------------------------------
    # 2026-07-22 Round Big-Stride Part 4 (P1, P2): dof_acc -1.25e-7 → -2.0e-7,
    # action_rate -0.005 → -0.008. Prior halving (to unlock push recovery)
    # went too far — real robot vibrated visibly under the July-21 policy
    # while MuJoCo replay looked clean. Bringing weights 60 % back toward
    # the pre-halving values (-2.5e-7 / -0.010) recovers joint smoothness
    # for real hardware without fully re-triggering the recovery-authority
    # problem. Complements the new foot_contact_velocity term (P6) which
    # attacks foot-specific chatter.
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
    # 2026-07-22 Round Big-Stride Part 4 (P3): weight -1e-3 → -2e-3. Prior
    # run reached foot_contact_force reward = -0.089 (up from -0.029) — real
    # gradient now firing, but still small vs stability terms and MuJoCo
    # showed peaks still at 3.5-4× BW. Doubling the weight doubles gradient
    # without changing the physics-anchored threshold. Threshold 700 N stays:
    # (a) 1.33× BW = upper end of healthy human walk (Winter/Perry gait);
    # (b) 70 % of X6-60 ankle motor limit (20 Nm ÷ 0.02 m arm ≈ 1000 N).
    # Both anchors agree. Static two-foot 262 N and single-foot 525 N both
    # below threshold — standing pays zero.
    # Now paired with P6 (foot_contact_velocity_penalty) which reduces the
    # cause (Δv_z at touchdown) upstream of this reactive force cap.
    foot_contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-2.0e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 850.0,
        },
    )
    # 2026-07-22 Round Big-Stride Part 4 (P6): soft-landing shaper.
    # Penalizes |foot z-velocity| while foot in contact. Upstream of the
    # reactive contact_forces cap above — teaches the policy to slow the
    # foot before touchdown rather than paying the impulse afterward.
    # Physics: F_peak ∝ m Δv / Δt, so cutting Δv_z at contact directly cuts
    # peak GRF. Small weight (-0.5) so it doesn't dominate stance-phase
    # small oscillations. force_threshold 5 N matches the standard used
    # elsewhere (feet_slide, feet_stance_flat_ankle).
    foot_contact_velocity = RewTerm(
        func=custom_mdp.foot_contact_velocity_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "force_threshold": 5.0,
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
    # Bumped -0.7 → -1.5 (2026-07-17): direct arc-walk attack. mujoco V3@10k
    # showed hip_roll raw 0.074 (~2° per side) — real amplifies to ~5° drift
    # which steers the walk. Stronger penalty forces near-zero hip_roll in
    # both stance and swing.
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_hip_roll_joint$"])},
    )
    # 2026-07-18 Round HW: -0.5 → -0.8. User observed ankle_roll still tilting
    # (foot not flat with ground) in V4 mujoco. feet_stance_flat_ankle only
    # covers ankle_pitch — this is the roll-axis equivalent, applied globally.
    # Modest bump so the penalty is visible without stopping the roll motion
    # needed for push recovery.
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.8,
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
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.03, 0.03),
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
        # 2026-07-18: (-2, 2) → (-3, 3) N / N-m. Real hardware wrench from
        # cables + off-CoM electronics is closer to ±3 than ±2; larger range
        # also gives the policy more balance-recovery practice per episode.
        self.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
        self.events.base_external_force_torque.params["torque_range"] = (-3.0, 3.0)

        # 2026-07-18: widened toward HV1.2's range because Tahiti C1 baseline
        # has converged (V3@30k walks). More friction variance = more real-
        # world floor coverage without materially slowing convergence.
        # Static 0.4-1.0 → 0.3-1.2, dynamic 0.4-0.9 → 0.3-1.0.
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)

        # Spawn at exactly the default joint pose (no random scale) so all envs
        # start from the same clean stance during Phase 1.
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # Reset velocity noise bumped ±0.1 → ±0.3 (2026-07-15) so every episode
        # starts mid-perturbation, not from near-still. Forces the policy to
        # build a recovery reflex, not just a start-from-still reflex —
        # complements the push_robot bump and CoM DR for backward-walk balance.
        # 2026-07-28 Round Vibration (W1): x-vel range (-0.3, 0.3) → (-0.5, 0.2).
        # Backward-biased. MuJoCo showed robot cannot recover from small
        # backward velocities — under-samples backward-recovery training
        # because forward starts survive longer per episode. Now ~70 % of
        # resets begin with x-velocity ≤ 0 → policy sees "started moving
        # backward, control it" as a primary training scenario, not an edge
        # case. Y and rotational ranges kept symmetric.
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.2), "y": (-0.3, 0.3), "z": (-0.3, 0.3),
                "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.3, 0.3),
            },
        }
        # Round C (2026-07-15): stronger push training to fix -0.5 m/s backward
        # walk fall in MuJoCo. A -0.8 m/s x-push is mechanically identical to
        # walking backward at 0.8 m/s — every pushed episode drills the exact
        # heel-loaded backward-recovery skill that was capping around -0.3 m/s.
        # 2026-07-18 Round HW: interval 5-8 s kept, but amplitude x ±1.5 → ±1.0
        # and y ±1.2 → ±0.8. V4@20k proved ±1.5 was gymnastic-level — the
        # policy either had to give up smoothness (thrashing seen at iter 18k)
        # or fall (base_contact 35.9 %). Real robot never sees ±1.5 m/s kicks;
        # ±1.0 covers the realistic disturbance envelope while leaving budget
        # for the smoothness penalties to shape the gait.
        self.events.push_robot.interval_range_s = (5.0, 8.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-1.0, 1.0), "y": (-0.8, 0.8)}}

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
        # 2026-07-18: interval 6-8 → 3-5 s and amplitude 1.0 → 1.8 m/s so the
        # kicks are visible in the viewer (user reported "cannot see any push
        # applied" — 1 m/s absorbed too quickly by the policy).
        self.events.push_robot.interval_range_s = (3.0, 5.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}}
        # Disable curriculum in play (common_step_counter starts at 0, would
        # force Phase 1 and overwrite the play ranges).
        self.curriculum.command_phase = None
        # Spread envs across the full command space.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.rel_standing_envs = 0.2
