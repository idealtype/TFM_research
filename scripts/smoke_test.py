#!/usr/bin/env python3
"""CPU-only smoke tests for FuncDec experiment modules."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "src" / "experiments"
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from loader_utils import dataloader_kwargs, resolve_data_path  # noqa: E402


EXPERIMENTS = {
    "hard_mask": "build_fine_mask_basis",
    "soft_mask": "build_soft_mask_basis",
    "nogate_softmask": "build_soft_mask_basis",
}


def purge_experiment_modules() -> None:
    for name in list(sys.modules):
        if name == "common" or name == "model" or name.startswith("model."):
            del sys.modules[name]


def smoke_experiment(name: str, basis_fn_name: str, batch_size: int, horizon: int) -> None:
    exp_dir = EXPERIMENTS_ROOT / name
    sys.path.insert(0, str(exp_dir))
    try:
        purge_experiment_modules()
        common = importlib.import_module("common")
        model_mod = importlib.import_module("model.decomp_funcdec")

        cfg = dict(common.DEFAULT_CONFIG)
        cfg["horizon"] = horizon
        model = model_mod.FuncDecModel(cfg, load_backbone=False).cpu()

        emb = torch.randn(batch_size, int(cfg["embed_dim"]))
        if basis_fn_name == "build_fine_mask_basis":
            bases = getattr(common, basis_fn_name)("hourly", int(cfg["context_len"]), horizon)
        else:
            bases = getattr(common, basis_fn_name)("hourly", horizon)
        daily, weekly, monthly, yearly = common.expand_bases(bases, batch_size, torch.device("cpu"))

        pred, decomp = model(emb, daily, weekly, monthly, yearly)
        target = torch.randn_like(pred)
        loss = F.l1_loss(pred, target)
        loss.backward()

        expected = (batch_size, horizon)
        assert pred.shape == expected, f"{name}: pred shape {tuple(pred.shape)} != {expected}"
        for key in ("trend", "seasonal", "residual"):
            assert decomp[key].shape == expected, f"{name}: {key} shape mismatch"
        print(f"[ok] {name}: pred={tuple(pred.shape)} loss={loss.item():.6f}")
    finally:
        while str(exp_dir) in sys.path:
            sys.path.remove(str(exp_dir))
        purge_experiment_modules()


def smoke_dataloader(args: argparse.Namespace) -> None:
    dataset = TensorDataset(torch.arange(8).float().view(4, 2))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, **dataloader_kwargs(args, "cpu"))
    first = next(iter(loader))[0]
    assert first.shape == (2, 2)
    resolved = resolve_data_path("/home/sia2/project/data/real_eval_lot_ett", args.data_root)
    assert resolved == args.data_root / "real_eval_lot_ett"
    print(
        f"[ok] dataloader: num_workers={args.num_workers} "
        f"pin_memory={args.pin_memory} data_root={args.data_root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--data_root", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke_dataloader(args)
    for name, basis_fn_name in EXPERIMENTS.items():
        smoke_experiment(name, basis_fn_name, args.batch_size, args.horizon)


if __name__ == "__main__":
    main()
