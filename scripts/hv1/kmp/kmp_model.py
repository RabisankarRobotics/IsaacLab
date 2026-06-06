"""KMP MLP: maps a 16-D HiWET Stage-1 command vector to a 28-D joint posture.

Input layout (matches generate_kmp_dataset.py command_layout):
    [h, lx, ly, lz, lqx, lqy, lqz, lqw, rx, ry, rz, rqx, rqy, rqz, rqw, alpha_t]

Output:
    28-D vector of actuated joint angles (rad) in V3 action order
    (LEG_JOINTS + ARM_JOINTS + WAIST_ACTUATED).

The model wraps input/output normalization stats so the same .pt file can be
loaded both for training continuation and for deploy (no separate stats file).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class KMPNormStats:
    """Per-dimension mean/std for input and output. Computed on the training
    split and frozen for the rest of training + deploy."""

    cmd_mean: torch.Tensor   # (16,)
    cmd_std: torch.Tensor    # (16,)
    q_mean: torch.Tensor     # (28,)
    q_std: torch.Tensor      # (28,)


class KMP(nn.Module):
    """3-hidden-layer MLP for the kinematic manifold prior.

    Network operates in normalized space; the public forward() takes raw
    commands and returns raw joint angles by applying the stats internally.
    """

    def __init__(self, in_dim: int = 16, out_dim: int = 28, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d_prev = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d_prev, hidden), nn.SiLU()]
            d_prev = hidden
        layers.append(nn.Linear(d_prev, out_dim))
        self.net = nn.Sequential(*layers)

        # Buffers (saved/loaded with the model — kept in sync via set_norm)
        self.register_buffer("cmd_mean", torch.zeros(in_dim))
        self.register_buffer("cmd_std", torch.ones(in_dim))
        self.register_buffer("q_mean", torch.zeros(out_dim))
        self.register_buffer("q_std", torch.ones(out_dim))

    def set_norm(self, stats: KMPNormStats) -> None:
        self.cmd_mean.copy_(stats.cmd_mean)
        self.cmd_std.copy_(stats.cmd_std)
        self.q_mean.copy_(stats.q_mean)
        self.q_std.copy_(stats.q_std)

    def forward(self, cmd: torch.Tensor) -> torch.Tensor:
        """Raw commands in, raw joint angles out (rad).

        cmd: (B, 16) — the 16-D command vector defined above.
        returns: (B, 28) — joint angles in V3 action order.
        """
        z = (cmd - self.cmd_mean) / self.cmd_std
        y = self.net(z)
        return y * self.q_std + self.q_mean

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "in_dim": self.net[0].in_features,
            "out_dim": self.net[-1].out_features,
            "hidden": self.net[0].out_features,
            "depth": sum(1 for m in self.net if isinstance(m, nn.SiLU)),
        }, path)

    @staticmethod
    def load(path: str, map_location: str | None = None) -> "KMP":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = KMP(in_dim=ckpt["in_dim"], out_dim=ckpt["out_dim"],
                    hidden=ckpt["hidden"], depth=ckpt["depth"])
        model.load_state_dict(ckpt["state_dict"])
        return model
