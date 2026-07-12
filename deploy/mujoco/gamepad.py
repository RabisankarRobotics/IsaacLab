"""Xbox gamepad reader for MuJoCo deploy commands.

Reads the left thumbstick for (vx, wz) and the right thumbstick for (vy),
scales the raw [-1, +1] axis values to the policy's training command range,
and returns a 3-vector [vx, vy, wz] that can be dropped into the runner's
velocity_commands slot each policy tick.

Mapping (Xbox 360 / Xbox One / Series controllers via SDL2 axis order on Linux):
    Left  vertical    axis 1  (up = -1)   -> vx   (stick UP    = +vx forward)
    Left  horizontal  axis 0  (left = -1) -> wz   (stick LEFT  = +wz turn left)
    Right horizontal  axis 3  (left = -1) -> vy   (stick LEFT  = +vy strafe left)

Install:
    pip install pygame

Self-test (prints live stick values so you can confirm the mapping):
    python deploy/mujoco/gamepad.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except ImportError as exc:
    raise ImportError(
        "gamepad.py needs pygame. Install with: pip install pygame"
    ) from exc


@dataclass
class GamepadAxes:
    """SDL2 axis indices for an Xbox controller.

    Modern Linux + xpad driver reports the two sticks on axes 0/1 (left) and
    3/4 (right); axis 2 and 5 are the analog triggers. If your controller
    reports right-stick on axis 2, construct GamepadAxes(right_x=2, right_y=3).
    """
    left_x:  int = 0
    left_y:  int = 1
    right_x: int = 3
    right_y: int = 4


class GamepadReader:
    """Poll an Xbox controller and produce a [vx, vy, wz] command vector
    scaled to the policy's training command range.

    Args:
        vx_max: forward-velocity magnitude at full stick deflection (m/s).
                Should match the training env's lin_vel_x range.
        vy_max: lateral-velocity magnitude at full deflection (m/s).
        wz_max: yaw-rate magnitude at full deflection (rad/s).
        deadzone: raw-stick magnitude below which the axis is zeroed. 0.1
                  is enough for a resting stick on a clean controller;
                  worn analog sticks may need 0.15-0.2.
        axes: SDL axis-index override (see GamepadAxes docstring).
        device_index: pygame joystick index if multiple controllers are
                      attached. Default 0.
    """

    def __init__(
        self,
        vx_max: float = 1.0,
        vy_max: float = 0.5,
        wz_max: float = 0.5,
        deadzone: float = 0.1,
        axes: "GamepadAxes | None" = None,
        device_index: int = 0,
    ):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No gamepad detected. Check `ls /dev/input/js*` and re-plug the "
                "controller. If using a wireless Xbox pad, `xboxdrv` or the kernel "
                "`xpad` module must be loaded."
            )
        self._js = pygame.joystick.Joystick(device_index)
        self._js.init()
        print(
            f"[gamepad] connected: {self._js.get_name()!r} "
            f"({self._js.get_numaxes()} axes, {self._js.get_numbuttons()} buttons)"
        )
        print(
            f"[gamepad] command scale: vx=+/-{vx_max:.2f} m/s, "
            f"vy=+/-{vy_max:.2f} m/s, wz=+/-{wz_max:.2f} rad/s, deadzone={deadzone}"
        )

        self.vx_max = float(vx_max)
        self.vy_max = float(vy_max)
        self.wz_max = float(wz_max)
        self.deadzone = float(deadzone)
        self.axes = axes if axes is not None else GamepadAxes()

    # ---- internals -------------------------------------------------------

    def _apply_deadzone(self, v: float) -> float:
        """Zero out small stick noise, then rescale so (deadzone, 1) maps to
        (0, 1) linearly — a partial push does not appear amplified."""
        if abs(v) < self.deadzone:
            return 0.0
        sign = 1.0 if v > 0.0 else -1.0
        return sign * (abs(v) - self.deadzone) / (1.0 - self.deadzone)

    # ---- public API ------------------------------------------------------

    def read(self) -> np.ndarray:
        """Return the current command as float32 [vx, vy, wz]. Non-blocking."""
        pygame.event.pump()  # drain the event queue so axis state is fresh
        raw_lx = self._js.get_axis(self.axes.left_x)
        raw_ly = self._js.get_axis(self.axes.left_y)
        raw_rx = self._js.get_axis(self.axes.right_x)

        lx = self._apply_deadzone(raw_lx)
        ly = self._apply_deadzone(raw_ly)
        rx = self._apply_deadzone(raw_rx)

        # SDL Y-axis convention: up = -1. Flip so stick UP -> +vx forward.
        vx = -ly * self.vx_max
        # SDL X-axis: left = -1. Isaac +wz = CCW = turn left. Flip.
        wz = -lx * self.wz_max
        # SDL X-axis: left = -1. Isaac +vy = left in body frame. Flip.
        vy = -rx * self.vy_max

        return np.array([vx, vy, wz], dtype=np.float32)

    def close(self) -> None:
        self._js.quit()
        pygame.joystick.quit()
        pygame.quit()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Print live stick values so the user can verify the axis mapping."""
    import time
    reader = GamepadReader(vx_max=1.0, vy_max=0.5, wz_max=0.5, deadzone=0.05)
    print("[self-test] hold each stick in turn — Ctrl-C to exit")
    try:
        while True:
            cmd = reader.read()
            print(
                f"\rvx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  wz={cmd[2]:+.2f}   ",
                end="", flush=True,
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        reader.close()


if __name__ == "__main__":
    _self_test()
