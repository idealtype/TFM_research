"""Runtime helpers for VESSL/local data paths and PyTorch DataLoaders."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch


DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", os.environ.get("VESSL_DATA_ROOT", "/workspace/data")))


def resolve_data_path(path: str | Path, data_root: str | Path | None = None) -> Path:
    """Resolve old project data paths under DATA_ROOT while preserving explicit paths."""
    p = Path(path)
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    old_prefix = Path("/home/sia2/project/data")
    try:
        rel = p.relative_to(old_prefix)
    except ValueError:
        return p
    return root / rel


def resolve_project_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve old /home/sia2/project paths under PROJECT_ROOT when requested."""
    p = Path(path)
    root = Path(project_root) if project_root is not None else Path(
        os.environ.get("PROJECT_ROOT", "/workspace")
    )
    old_prefix = Path("/home/sia2/project")
    try:
        rel = p.relative_to(old_prefix)
    except ValueError:
        return p
    return root / rel


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root for mounted datasets/caches. Defaults to DATA_ROOT/VESSL_DATA_ROOT or /workspace/data.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(os.environ.get("NUM_WORKERS", "2")),
        help="DataLoader worker count. Use 0 for minimal local debugging.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        default=None,
        help="Enable pinned host memory for CUDA transfers.",
    )
    parser.add_argument(
        "--no_pin_memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned host memory.",
    )


def pin_memory_enabled(args: argparse.Namespace, device: torch.device | str | None = None) -> bool:
    if args.pin_memory is not None:
        return bool(args.pin_memory)
    if device is None:
        return torch.cuda.is_available()
    dev = torch.device(device)
    return dev.type == "cuda"


def dataloader_kwargs(
    args: argparse.Namespace,
    device: torch.device | str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    workers = max(0, int(getattr(args, "num_workers", 0)))
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": pin_memory_enabled(args, device),
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(os.environ.get("PREFETCH_FACTOR", "2"))
    kwargs.update(extra)
    return kwargs
