"""Supervised training of the KMP MLP on the IK dataset.

Loss: weighted MSE on joint angles. The weights mirror the IK weighting:
legs and waist get full weight (1.0); arm joints get the same since wrist
position tracking is a primary objective. The point of the weighting hook
is to make it easy to up-weight problematic joints later.

Usage:
  conda activate env_isaaclab
  python scripts/hv1/kmp/train_kmp.py \
      --dataset deploy/model/kmp/kmp_dataset_v1.npz \
      --out deploy/model/kmp/kmp_v1.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kmp_model import KMP, KMPNormStats  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------ data ----------------------------------------------------------
    print(f"[LOAD] {args.dataset}")
    d = np.load(args.dataset)
    commands = d["commands"].astype(np.float32)
    qpos = d["joint_pos"].astype(np.float32)
    N = len(commands)
    print(f"[INFO] N={N}, cmd_dim={commands.shape[1]}, q_dim={qpos.shape[1]}")

    # Split
    idx = np.random.permutation(N)
    n_val = int(N * args.val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    cmd_train, q_train = commands[train_idx], qpos[train_idx]
    cmd_val, q_val = commands[val_idx], qpos[val_idx]
    print(f"[SPLIT] train={len(train_idx)} val={len(val_idx)}")

    # Normalization stats from training split
    cmd_mean = torch.tensor(cmd_train.mean(0), dtype=torch.float32)
    cmd_std = torch.tensor(cmd_train.std(0) + 1e-6, dtype=torch.float32)
    q_mean = torch.tensor(q_train.mean(0), dtype=torch.float32)
    q_std = torch.tensor(q_train.std(0) + 1e-6, dtype=torch.float32)
    stats = KMPNormStats(cmd_mean=cmd_mean, cmd_std=cmd_std, q_mean=q_mean, q_std=q_std)

    # Tensors on device
    cmd_train_t = torch.tensor(cmd_train, device=args.device)
    q_train_t = torch.tensor(q_train, device=args.device)
    cmd_val_t = torch.tensor(cmd_val, device=args.device)
    q_val_t = torch.tensor(q_val, device=args.device)

    train_ds = TensorDataset(cmd_train_t, q_train_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # ------ model ---------------------------------------------------------
    model = KMP(in_dim=16, out_dim=28, hidden=args.hidden, depth=args.depth).to(args.device)
    model.set_norm(stats)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] params={n_params}  hidden={args.hidden}  depth={args.depth}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    # ------ train ---------------------------------------------------------
    best_val = float("inf")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for cmd_b, q_b in train_loader:
            opt.zero_grad()
            pred = model(cmd_b)
            loss = F.mse_loss(pred, q_b)
            loss.backward()
            opt.step()
            sched.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        # Val
        model.eval()
        with torch.no_grad():
            pred_v = model(cmd_val_t)
            val_loss = F.mse_loss(pred_v, q_val_t).item()
            # Per-joint mean abs error in degrees
            mae_per_joint = (pred_v - q_val_t).abs().mean(0)  # (28,)
            mae_deg = mae_per_joint.cpu().numpy() * 180 / np.pi
            mae_max_deg = float(mae_deg.max())
            mae_mean_deg = float(mae_deg.mean())

        if val_loss < best_val:
            best_val = val_loss
            model.save(args.out)
            tag = " ✓ saved"
        else:
            tag = ""

        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d}  train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"mae_mean={mae_mean_deg:.2f}°  mae_max={mae_max_deg:.2f}°  "
                  f"lr={sched.get_last_lr()[0]:.2e}{tag}")

    dt = time.time() - t0
    print(f"\n[DONE] {dt:.1f}s. Best val_loss={best_val:.6f}")
    print(f"[SAVED] {args.out}")

    # Final eval on val: per-joint stats
    model = KMP.load(args.out, map_location=args.device).to(args.device)
    model.eval()
    with torch.no_grad():
        pred_v = model(cmd_val_t).cpu().numpy()
    err = pred_v - q_val.astype(np.float32)
    err_deg = np.abs(err) * 180 / np.pi
    print(f"\n[VAL PER-JOINT MAE (deg)]:")
    joint_names = list(d["actuated_joint_names"])
    for name, m, x in zip(joint_names, err_deg.mean(0), err_deg.max(0)):
        print(f"  {name:35s}  mean={m:5.2f}°  max={x:5.2f}°")


if __name__ == "__main__":
    main()
