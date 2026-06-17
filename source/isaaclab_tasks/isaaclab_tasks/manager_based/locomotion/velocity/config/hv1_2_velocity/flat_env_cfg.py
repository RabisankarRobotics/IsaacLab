"""HV1.2 walking task on flat ground (Isaac Lab manager-based).

Phase 2: policy controls ONLY the 12 leg joints. Upper body (3-DoF waist +
14-DoF arms + 3-DoF head) is held at the default pose by the implicit-actuator
PD, with targets re-asserted at every reset via custom event terms.

Differences vs. hv1_2_stand:
  * Non-zero velocity command ranges (forward 0..1 m/s, yaw ±1 rad/s).
  * Feet-air-time and feet-slide rewards (foot contact sensor on ankle_roll_link).
  * Stronger joint_deviation penalties on upper body to actively resist torque
    feedback from leg swings.
  * Larger action scale (0.5) so the policy has range to lift the feet.
"""

from __future__ import annotations

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

from isaaclab_assets import HV1_2_CFG  # isort: skip

from . import mdp as custom_mdp


# ---- joint-name regexes -------------------------------------------------
LEG_JOINTS = [
    "^(left|right)_hip_(yaw|pitch|roll)_joint$",
    "^(left|right)_knee_joint$",
    "^(left|right)_ankle_(pitch|roll)_joint$",
]
WAIST_JOINT_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
HEAD_JOINT_NAMES = ["head_pitch_joint", "head_yaw_joint"]  # 2-DoF head (no head_roll in without_ee URDF)
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Default upper-body pose (matches HV1_2_CFG.init_state).
WAIST_TARGETS = {n: (0.0, 0.0) for n in WAIST_JOINT_NAMES}
HEAD_TARGETS = {n: (0.0, 0.0) for n in HEAD_JOINT_NAMES}
ARM_TARGETS_PIN = {
    "left_shoulder_pitch_joint": (0.4, 0.4),
    "left_shoulder_roll_joint":  (0.0, 0.0),
    "left_shoulder_yaw_joint":   (0.0, 0.0),
    "left_elbow_joint":          (0.3, 0.3),
    "left_wrist_roll_joint":     (0.0, 0.0),
    "left_wrist_pitch_joint":    (0.0, 0.0),
    "left_wrist_yaw_joint":      (0.0, 0.0),
    "right_shoulder_pitch_joint": (0.4, 0.4),
    "right_shoulder_roll_joint":  (0.0, 0.0),
    "right_shoulder_yaw_joint":   (0.0, 0.0),
    "right_elbow_joint":          (0.3, 0.3),
    "right_wrist_roll_joint":     (0.0, 0.0),
    "right_wrist_pitch_joint":    (0.0, 0.0),
    "right_wrist_yaw_joint":      (0.0, 0.0),
}


@configclass
class HV1_2VelocityActionsCfg:
    """Policy actions on legs only (12 of 32 joints).

    Scale 0.5 — the standard locomotion scale from velocity_env_cfg.
    We tried 0.3 (with base_lin_vel removed) for smoother gait, but the
    policy diverged: smaller scale forces the actor to output 5/3×
    larger raw values to reach the same joint target, and action_rate_l2
    (which is computed on RAW actions, not scaled) exploded. Combined
    with the obs change it caused full divergence (mean_reward 2.5,
    action_std 2.7). Keep scale 0.5 here.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class HV1_2VelocityObservationsCfg:
    """Policy observations match what a real HV1.2 robot can actually sense:
    IMU (base_ang_vel + projected_gravity) and joint encoders. No base_lin_vel
    — that would require a state estimator we don't have on hardware.
    Per-term Unoise mimics real sensor noise so the trained policy is robust.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # IMU
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        # Commands (from operator, no sensor noise)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # Encoders
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # Internal
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HV1_2VelocityEventCfg(EventCfg):
    """Inherits velocity-task events and adds reset-time pinning for
    waist / arms / head joints (since the policy no longer drives them)."""

    pin_waist_target_reset = EventTerm(
        func=custom_mdp.randomize_arm_joint_targets,
        mode="reset",
        params={
            "position_range": WAIST_TARGETS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINT_NAMES, preserve_order=True),
        },
    )
    pin_arms_target_reset = EventTerm(
        func=custom_mdp.randomize_arm_joint_targets,
        mode="reset",
        params={
            "position_range": ARM_TARGETS_PIN,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_TARGETS_PIN.keys()), preserve_order=True),
        },
    )
    pin_head_target_reset = EventTerm(
        func=custom_mdp.randomize_arm_joint_targets,
        mode="reset",
        params={
            "position_range": HEAD_TARGETS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=HEAD_JOINT_NAMES, preserve_order=True),
        },
    )


@configclass
class HV1_2VelocityRewardsCfg:
    """Velocity-tracking rewards (walking)."""

    # ---- tracking (restored to 1.0 — strong velocity command pulls the
    #      robot forward, which is impossible while balancing on one leg) ----
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        # std 0.5 → 0.25 was too aggressive on hot resume — action_std jumped
        # 0.4 → 0.77 from the reward-landscape shock, exploration destabilized
        # the gait. Backed off to 0.35: at v_cmd=0.5, half-speed (actual=0.25)
        # earns 0.60 (vs 0.78 at std=0.5, 0.37 at std=0.25). Still a clear
        # pressure to close the tracking gap, but the step from the
        # checkpoint's value-function expectation is small enough that PPO
        # won't panic-explore.
        params={"command_name": "base_velocity", "std": 0.35},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        # std widened 0.5 → 1.0. Previous run had error_vel_yaw=2.42 rad/s on a
        # ±1 command — exp(-(2.42/0.5)²) ≈ 0, so the gradient on yaw vanished
        # and the policy never learned to turn. With std=1.0 the reward stays
        # measurable while error is in the 1–2 rad/s range, giving PPO signal
        # to actually shrink yaw error.
        params={"command_name": "base_velocity", "std": 1.0},
    )
    # ---- gait: keep the bipedal single-stance reward (it got the robot
    # stepping) but dial weights down to G1-style values and ADD gait-shape
    # terms — variance penalty (anti-asymmetry) and foot clearance reward
    # (smooth swing). Reference: Unitree G1 rough_env_cfg uses weight=0.25.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            # Threshold 0.4 → 0.3 s. At 0.4 s the reward saturates only at long
            # swings, so the policy adopted a slow-cadence gait that also gave
            # hip_yaw room to arc outward during the leisurely swing. 0.3 s
            # gives full reward at a brisker step rhythm — directly tightens
            # cadence AND removes the time-window for the yaw arc.
            "threshold": 0.3,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,  # G1 uses -0.1 (too weak for HV1.2's heavier feet),
                      # HV1's working config uses -1.0. -0.5 is a measured
                      # midpoint that suppresses shuffling without forcing
                      # the policy to lock the support foot.
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )
    # NEW: direct asymmetric-gait penalty (the yoga-walk fix).
    # Penalizes variance of last_air_time and last_contact_time across the
    # two feet. Zero when both feet step in the same rhythm.
    feet_airtime_variance = RewTerm(
        func=custom_mdp.air_time_variance_penalty,
        weight=-5.0,  # was -2.0 — at iter 6673 the episode-reward for this term
                       # was still -0.15, i.e. asymmetric stepping persisted
                       # because other reward terms dominated. -5.0 makes it
                       # large enough to actually shape symmetric cadence.
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link")},
    )
    # NEW: rewards swing-foot clearance — encourages a clean foot-lift arc
    # rather than dragging. Target = 0.10 m above ground (HV1.2 foot
    # thickness is ~0.04 m, so this is ~6 cm of clearance). Only active
    # while the foot is moving (tanh gate on xy-velocity).
    foot_clearance = RewTerm(
        func=custom_mdp.foot_clearance_reward,
        # 0.5 → 0.3. foot_clearance is not command-masked, so at v_cmd=0 the
        # policy was harvesting ~0.49 per step by marching in place (parade
        # gait). Cutting weight + adding stand_still_no_cmd below removes
        # the incentive to cycle feet at standstill.
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            # target_height 0.10 → 0.07 m. Policy was reliably hitting 10 cm
            # which makes the gait look high-lift / slow-motion and gives more
            # time for hip_yaw to drift outward. 7 cm clearance is still safe
            # over flat ground and shortens the swing arc — both visual
            # speed-up and a smaller hip_yaw deviation window.
            "target_height": 0.07,
            "std": 0.05,
            "tanh_mult": 2.0,
        },
    )
    # ---- stability ----
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # One-sided L1 "no-squat" penalty: zero when pelvis is at or above 0.92 m,
    # linear in shortfall below.
    #   normal walk at 0.93     → 0 penalty
    #   stance push-off at 0.96 → 0 penalty (allows natural pelvis lift)
    #   crouch at 0.84          → shortfall 0.08, penalty -0.8
    #   fall at 0.10            → shortfall 0.82, penalty -8.2 (large but bounded)
    # Restored after symmetric L2 over-constrained the gait into slow-motion
    # deep-knee walking — the policy was avoiding the natural up-bob during
    # stance push-off because L2 punished any height > 0.92 as well.
    base_height_below = RewTerm(
        func=custom_mdp.base_height_below_target_l1,
        weight=-10.0,
        # target 0.92 → 0.88 m. At 0.92 the floor was hugging the resting
        # pelvis height (~0.94 m at default knee bend), so the policy kept
        # support knees rigid to avoid even small pelvis dips → stilt-walk
        # / straight-knee gait. 0.88 m gives ~6 cm of pelvis travel for
        # natural knee-bend stance compression while still catching any
        # actual crouch-walk attempt.
        params={"target_height": 0.88},
    )
    # ---- effort / smoothness ----
    # Backed off from -0.005 — paired with the height-penalty rework, that
    # was too many tight new constraints at once and PPO went chaotic.
    # -0.003 is between original (-0.002) and the over-aggressive (-0.005).
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.003)
    # ---- safety ----
    # is_alive / termination_penalty softened (0.15 / -200 → 0.05 / -100) to
    # match HV1's working config. The large -200 termination spike was inflating
    # value-target variance whenever an env fell, which previously contributed
    # to the policy-collapse failure mode after the obs+scale change.
    is_alive = RewTerm(func=mdp.is_alive, weight=0.05)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        # Added hip_yaw to the limits-tracked set. Without it the joint had no
        # mechanical-limit penalty AT ALL — combined with the earlier hard
        # mask, hip_yaw drifted to ~90° outward on one leg. The limit penalty
        # is one-sided (fires only when q crosses the soft limit) so it has
        # zero cost during normal walking but acts as a wall against extreme
        # drift if any other reward term becomes too permissive.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_(pitch|roll)_joint$", ".*_hip_yaw_joint$"])},
    )
    # Penalize the OUTCOME (feet too close in yaw-frame Y) rather than the
    # MEANS (hip deviation). Lets the policy use hip_roll freely for balance
    # while strictly preventing leg crossing. One-sided: zero when clear.
    feet_lateral_clearance = RewTerm(
        func=custom_mdp.feet_lateral_distance_clearance,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "min_distance": 0.18,  # ~half of standing 0.34 m separation
        },
    )
    # Split hip-deviation into two terms so we can target each axis at the
    # right strength:
    #   * hip_YAW = the "duck-foot / outward leg rotation" axis. Strong weight
    #     because the policy was visibly twisting one leg out.
    #   * hip_ROLL = sideways leg splay. Kept moderate; feet_lateral_clearance
    #     already prevents the bad outcome (legs crossing or splaying too far).
    #
    # Path A v2 (post iter-15840 collapse): switched from a hard mask to a
    # SOFT scaling on |ang_vel_z_cmd|. The hard mask released the penalty
    # entirely above threshold — with ang_vel_z sampled uniformly from
    # (-0.5, 0.5), that left hip_yaw unpenalized 72% of the time, and the
    # joint drifted to its mechanical limit (one-leg 90° outward limp).
    # Soft scaling: penalty is ALWAYS firing (so hip_yaw always feels a
    # restoring force toward 0), but weight is multiplied by
    # exp(-(ang_vel_z_cmd^2) / 0.25) which gives:
    #     ang_vel_z = 0.0  → 100% penalty (full -0.7 / -0.2)
    #     ang_vel_z = 0.25 → 78%  penalty
    #     ang_vel_z = 0.5  → 37%  penalty (relaxed for turning)
    # The constant baseline kills the 90° collapse; the smooth fall-off still
    # gives the policy room to use hip_yaw for in-place pivot.
    joint_deviation_hip_yaw = RewTerm(
        func=custom_mdp.joint_deviation_turn_softened_l1,
        weight=-0.7,
        params={
            "command_name": "base_velocity",
            "turn_softness_std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_hip_yaw_joint$"]),
        },
    )
    joint_vel_hip_yaw = RewTerm(
        func=custom_mdp.joint_vel_turn_softened_l2,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "turn_softness_std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_hip_yaw_joint$"]),
        },
    )
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_hip_roll_joint$"])},
    )
    # One-sided "no stilt-walk" penalty: penalize knees straighter than 0.4 rad.
    # Default rest bend is 0.36 rad, so even neutral pose triggers a tiny push to
    # bend more. Swing-leg knees naturally bend to 0.7-1.0 rad → 0 penalty during
    # swing. Only locked-straight stance knees (which give the stilt-walk look)
    # actually pay.
    # Pairs with base_height_below_target=0.88: that change removed the wall
    # forcing rigid stance, this term adds positive pressure to dip.
    knee_too_straight = RewTerm(
        func=custom_mdp.knee_too_straight_penalty,
        weight=-1.0,
        params={
            "threshold": 0.4,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["^(left|right)_knee_joint$"]),
        },
    )
    # "Be still when commanded to stand." Active only when ||cmd_vel|| < 0.1
    # (i.e. the standing-env subset). Penalizes L1 deviation of the swing-relevant
    # leg joints from default. Kills the parade-march at v_cmd=0 directly —
    # complements the foot_clearance weight drop (which removes the *reward*
    # for marching; this adds the *anti-reward*).
    # Targets hip_pitch / knee / ankle_pitch only — hip_roll and ankle_roll
    # remain free for static balance compensation.
    stand_still_no_cmd = RewTerm(
        func=custom_mdp.stand_still_joint_deviation_l1,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["^(left|right)_(hip_pitch|knee|ankle_pitch)_joint$"],
            ),
        },
    )


@configclass
class HV1_2VelocityFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: HV1_2VelocityActionsCfg = HV1_2VelocityActionsCfg()
    observations: HV1_2VelocityObservationsCfg = HV1_2VelocityObservationsCfg()
    rewards: HV1_2VelocityRewardsCfg = HV1_2VelocityRewardsCfg()
    events: HV1_2VelocityEventCfg = HV1_2VelocityEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # ---------------- scene: flat plane, no height scanner ----------------
        self.scene.robot = HV1_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # ---------------- commands: walking ranges ---------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # Narrowed ±1.0 → ±0.5 rad/s. With std=1.0 on track_ang_vel_z_exp,
        # a ±0.5 command is well within the reward's "learnable" range
        # (exp(-(0.5/1.0)²) ≈ 0.78). After the policy converges on this range
        # we can widen back to ±1.0 in a second-stage curriculum run.
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)
        # Bump standing envs 0.02 → 0.1. 2% (≈80 of 4096 envs) gave the policy
        # only a thin slice of stand-still training. 10% (≈410 envs) makes
        # zero-command behavior a first-class case the policy has to handle
        # cleanly — important for safe deploy where the operator can leave the
        # joystick at neutral.
        self.commands.base_velocity.rel_standing_envs = 0.1

        # ---------------- domain randomization on the base link --------------
        # Parent points these at a body named "base" which doesn't exist on
        # HV1.2 (root is "pelvis"). Re-target them and dial the ranges down
        # to "small but useful" so the policy is robust across mass / COM
        # variation without being asked to track an impossible target.

        # ±3 kg around the 83 kg total — ≈ 3.6% body mass.
        self.events.add_base_mass.params["asset_cfg"].body_names = "pelvis"
        self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 3.0)

        # ±3 cm horizontal, ±1 cm vertical pelvis COM offset.
        self.events.base_com.params["asset_cfg"].body_names = "pelvis"
        self.events.base_com.params["com_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.01, 0.01),
        }

        # Re-target the reset-time external force/torque term (default ranges
        # are 0/0, so this just keeps it from erroring out — push_robot does
        # the actual mid-episode perturbation).
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "pelvis"

        # Widen ground friction randomization — single-bucket friction is too
        # narrow for "stable across environments". Static 0.4..1.2 covers
        # smooth indoor floor to grippy rubber; dynamic 0.3..1.0 similarly.
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)

        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        # push robot every 8-12 s for robustness
        self.events.push_robot.interval_range_s = (8.0, 12.0)
        self.events.push_robot.params = {"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}

        # ---------------- terminations: only "pelvis" contact -----------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = "pelvis"

        # ---------------- runtime ---------------------------------------------
        self.episode_length_s = 20.0
        self.decimation = 4  # policy at 50 Hz when sim.dt = 0.005


@configclass
class HV1_2VelocityFlatEnvCfg_PLAY(HV1_2VelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # Spread the 50 envs across the FULL command space so you can visually
        # see forward / backward / sideways / turn gaits side-by-side. Each env
        # gets its own random command at reset; resample_time_range controls how
        # often each env picks a new command mid-episode.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)   # forward & backward
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)   # side-step both ways
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)   # turn both ways
        # New command every 5 s so the playback shows multiple gait types per env.
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        # 20% standing during play — 10 of the 50 envs will hold a zero-command
        # stand. Lets you visually verify the policy can stop cleanly without
        # foot shuffling or drift.
        self.commands.base_velocity.rel_standing_envs = 0.2
