# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sanity spawn of the HV1 humanoid on a flat ground plane.

Two HV1 robots are spawned side-by-side. No control is applied — the robot
holds its default joint targets via the PD drives configured in HV1_CFG, so it
should stand (or sag a bit). Use this to verify the URDF imports cleanly, the
joint names resolve, the actuator groups bind, and the default pose looks sane.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/01_assets/run_hv1.py
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sanity spawn the HV1 humanoid.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the imports."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets import HV1_CFG  # isort:skip


def design_scene() -> tuple[dict, list[list[float]]]:
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origins = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
    sim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    sim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    hv1_cfg = HV1_CFG.copy()
    hv1_cfg.prim_path = "/World/Origin.*/Robot"
    hv1 = Articulation(cfg=hv1_cfg)

    return {"hv1": hv1}, origins


def run_simulator(sim: SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    robot = entities["hv1"]
    sim_dt = sim.get_physics_dt()
    count = 0
    while simulation_app.is_running():
        # Reset every ~5 seconds (5.0 / sim_dt steps).
        if count % int(5.0 / sim_dt) == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            print(
                f"[INFO] Reset. {robot.num_joints} joints, "
                f"{robot.num_bodies} bodies. Default pose held by PD."
            )
        # Hold the default joint targets — no random action.
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.005)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 3.5, 2.0], [0.5, 0.0, 0.8])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO] Setup complete. HV1 spawned.")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
