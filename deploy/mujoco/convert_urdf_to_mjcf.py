"""One-time URDF → MJCF conversion for HV1.

MuJoCo can load URDF directly, but the in-memory model has no ground, no
lighting, no actuators tuned for MuJoCo. The standard pattern (used by all
Unitree examples) is:

    URDF  --convert-->  hv1.xml (MJCF)  --include-->  scene.xml

This script does the convert step. Run once after any URDF change.

Usage:
    python convert_urdf_to_mjcf.py <input.urdf> <output.xml>

Example:
    python convert_urdf_to_mjcf.py \\
        /home/rabisankar/IsaacLab/source/isaaclab_assets/data/custom_robot/urdf_mesh/hv1/hv1.urdf \\
        hv1.xml
"""

import argparse
import os
import sys

import mujoco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf", type=str, help="Path to input URDF.")
    parser.add_argument("output", type=str, help="Path to write MJCF (.xml).")
    args = parser.parse_args()

    if not os.path.isfile(args.urdf):
        sys.exit(f"URDF not found: {args.urdf}")

    print(f"[convert] loading URDF: {args.urdf}")
    # Loading URDF compiles it into MuJoCo's internal representation.
    # mj_saveLastXML writes that representation back out as canonical MJCF.
    m = mujoco.MjModel.from_xml_path(args.urdf)
    print(f"[convert]   nbody={m.nbody}  njnt={m.njnt}  ngeom={m.ngeom}")

    # Save the compiled model as MJCF
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    mujoco.mj_saveLastXML(args.output, m)
    print(f"[convert] wrote MJCF: {args.output}")
    print()
    print("Next: write/check scene.xml that <include>s this file, then run")
    print("    python deploy_mujoco_hv1.py --config <yaml> --urdf scene.xml")


if __name__ == "__main__":
    main()
