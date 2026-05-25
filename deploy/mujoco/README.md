# HV1 MuJoCo Sim-to-Sim Deployment

Run an Isaac-Lab-trained HV1 loco-manip policy in MuJoCo for sim-to-sim validation.

## One-time setup

```bash
cd deploy/mujoco

# 1. Install deps (separate from Isaac Lab env)
pip install mujoco torch pyyaml numpy

# 2. Convert URDF → MJCF (run once; re-run after URDF changes)
python convert_urdf_to_mjcf.py \
    /home/rabisankar/IsaacLab/source/isaaclab_assets/data/custom_robot/urdf_mesh/hv1/hv1.urdf \
    hv1.xml
```

This produces `hv1.xml` next to `scene.xml`. `scene.xml` already `<include>`s `hv1.xml`, adds a ground plane, lights, and skybox.

## Per-policy setup

After training a new policy, run play + dump on the Isaac Lab side:

```bash
# 1. Export policy.pt
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Tracking-LocoManip-HV1-Play-v0 --num_envs 1

# 2. Dump MuJoCo config
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/dump_mujoco_config.py \
    --task Isaac-Tracking-LocoManip-HV1-Play-v0 --num_envs 1
```

This writes `logs/.../<run>/exported/mujoco_config.yaml` alongside `policy.pt`.

## Run

```bash
cd deploy/mujoco

# Test 1: Stand (zero command, defaults)
python deploy_mujoco_hv1.py \
    --config <run_dir>/exported/mujoco_config.yaml \
    --urdf scene.xml

# Test 2: Walk forward
python deploy_mujoco_hv1.py \
    --config <run_dir>/exported/mujoco_config.yaml \
    --urdf scene.xml \
    --cmd_lin_x 1.0

# Test 3: Reach
python deploy_mujoco_hv1.py \
    --config <run_dir>/exported/mujoco_config.yaml \
    --urdf scene.xml \
    --left_ee  0.45  0.15 0.45  1.0 0.0 0.0 0.0 \
    --right_ee 0.45 -0.15 0.45  1.0 0.0 0.0 0.0
```

> `--urdf scene.xml` overrides the YAML's URDF path so the scene wrapper
> (with ground + lights) is loaded instead of the bare robot.

## File layout

```
deploy/mujoco/
├── convert_urdf_to_mjcf.py   # URDF → MJCF (one-time)
├── deploy_mujoco_hv1.py      # main runner
├── scene.xml                 # MuJoCo scene (ground + lights + include)
├── hv1.xml                   # generated MJCF (from convert script)
└── README.md
```

## Why scene.xml is necessary

MuJoCo can load URDF directly, but URDF doesn't define:
- Ground plane → robot falls forever
- Lighting → dark viewport
- Camera angles
- Skybox

Unitree's `deploy_mujoco/` uses the same pattern: a per-robot `<robot>.xml`
generated from URDF, plus a `scene.xml` that includes it and adds the world.
