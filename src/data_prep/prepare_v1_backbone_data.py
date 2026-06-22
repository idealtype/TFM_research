#!/usr/bin/env python3
"""Prepare TimesFM v1 backbone caches for the soft-mask warm-start experiment.

This script keeps the same compact row schedule as ``prepare_data.py`` and
rebuilds only the ``backbone_emb*.pt`` files with TimesFM v1 last-patch hidden
states. Targets, component labels, masks, manifests, and sidecar files are
copied from the existing v2.5 cache tree.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PREP_DIR = REPO_ROOT / "src" / "data_prep"
LOTSA_DIR = REPO_ROOT / "data_source_light" / "data_lotsa"
for path in [DATA_PREP_DIR, LOTSA_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import prepare_data as compact  # noqa: E402
from build_cache import slice_context  # noqa: E402
from shared_utils import target_values  # noqa: E402


MODEL_REPO = "google/timesfm-1.0-200m-pytorch"
EMBED_DIM = 1280
DEFAULT_SRC_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data"))
DEFAULT_DST_DATA_ROOT = Path("/tmp/data_v1_backbone")
DEFAULT_REAL_EVAL_ROOT = Path("/workspace/data/real_eval_lot_ett")
DEFAULT_REAL_EVAL_DST_ROOT = Path("/tmp/real_eval_lot_ett_v1_backbone")
DEFAULT_CHECKPOINT_PATH = Path("/tmp/timesfm-1.0-200m-pytorch/torch_model.ckpt")
DEFAULT_EXP_DIR = (
    REPO_ROOT
    / "src"
    / "experiments"
    / "backbone_adjustment"
    / "soft_warm_s10_oldloss_best_v1_backbone"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_data_root", type=Path, default=DEFAULT_SRC_DATA_ROOT)
    parser.add_argument("--dst_data_root", type=Path, default=DEFAULT_DST_DATA_ROOT)
    parser.add_argument("--lotsa_cache_root", type=Path, default=None)
    parser.add_argument("--domain_config", type=Path, default=compact.DEFAULT_DOMAIN_CONFIG)
    parser.add_argument("--horizons", nargs="+", type=int, default=compact.DEFAULT_HORIZONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--fourier_batch_size", type=int, default=None)
    parser.add_argument("--fourier_warmup_steps", type=int, default=125)
    parser.add_argument("--mixed_steps", type=int, default=2500)
    parser.add_argument("--synth_interval", type=int, default=10)
    parser.add_argument("--residual_steps", type=int, default=500)
    parser.add_argument("--real_group_chunk_steps", type=int, default=63)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--v1_src", type=Path, default=None,
                        help="Path to timesfm_origin/v1/src. If set, inserted before importing timesfm.")
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--encode_batch_size", type=int, default=256)
    parser.add_argument("--skip_lotsa_subsets", nargs="*", default=None,
                        help="Subset names to exclude from real data planning (e.g. HZMETRO SHMETRO)")
    parser.add_argument("--skip_train_cache", action="store_true")
    parser.add_argument("--real_eval_root", type=Path, default=DEFAULT_REAL_EVAL_ROOT)
    parser.add_argument("--real_eval_dst_root", type=Path, default=DEFAULT_REAL_EVAL_DST_ROOT)
    parser.add_argument("--skip_real_eval_cache", action="store_true")
    parser.add_argument("--exp_dir", type=Path, default=DEFAULT_EXP_DIR,
                        help="v1 backbone experiment directory whose train.py defines the schedule.")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {name}, but CUDA is not available")
    return torch.device(name)


def freq_bucket(freq: str) -> int:
    freq = str(freq).lower()
    if freq in {"weekly", "w", "w-sun", "w-mon", "monthly", "m", "ms", "me"}:
        return 1
    if freq in {"quarterly", "yearly", "q", "qs", "qe", "y", "ys", "a", "as"}:
        return 2
    return 0


class TimesFmV1Encoder:
    def __init__(
        self,
        v1_src: Path | None,
        checkpoint_path: Path,
        hf_cache_dir: str | None,
        device: torch.device,
        batch_size: int,
    ) -> None:
        if v1_src is not None:
            sys.path.insert(0, str(v1_src.resolve()))
        from timesfm.timesfm_base import TimesFmCheckpoint, TimesFmHparams  # type: ignore
        from timesfm.timesfm_torch import TimesFmTorch  # type: ignore

        backend = "gpu" if device.type == "cuda" else "cpu"
        checkpoint = TimesFmCheckpoint(
            path=str(checkpoint_path) if checkpoint_path.exists() else None,
            huggingface_repo_id=None if checkpoint_path.exists() else MODEL_REPO,
            local_dir=hf_cache_dir,
        )
        self.tfm = TimesFmTorch(
            hparams=TimesFmHparams(
                context_len=512,
                horizon_len=128,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                num_heads=16,
                model_dims=1280,
                per_core_batch_size=int(batch_size),
                backend=backend,
            ),
            checkpoint=checkpoint,
        )
        self.device = self.tfm._device
        if self.device != device:
            print(f"[v1] effective device={self.device}", flush=True)
        self.batch_size = int(batch_size)

    @torch.no_grad()
    def encode(self, contexts: np.ndarray, freq: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if contexts.ndim != 2:
            raise ValueError(f"contexts must be 2D, got shape={contexts.shape}")
        if contexts.shape[1] > 512:
            raise ValueError(f"TimesFM v1 supports max context 512, got {contexts.shape[1]}")
        outputs: list[torch.Tensor] = []
        mus: list[torch.Tensor] = []
        sigmas: list[torch.Tensor] = []
        freq_ids = [freq_bucket(freq)] * len(contexts)
        for start in tqdm(range(0, len(contexts), self.batch_size), desc=f"v1 encode {freq}", leave=False):
            chunk = [row.astype(np.float32, copy=False) for row in contexts[start:start + self.batch_size]]
            input_ts, input_padding, inp_freq, pmap_pad = self.tfm._preprocess(chunk, freq_ids[start:start + len(chunk)])
            t_input_ts = torch.as_tensor(input_ts, dtype=torch.float32, device=self.device)
            t_padding = torch.as_tensor(input_padding[:, : input_ts.shape[1]], dtype=torch.float32, device=self.device)
            t_freq = torch.as_tensor(inp_freq, dtype=torch.long, device=self.device)
            model = self.tfm._model
            model_input, patched_padding, stats, _ = model._preprocess_input(t_input_ts, t_padding)
            model_input = model_input + model.freq_emb(t_freq)
            hidden = model.stacked_transformer(model_input, patched_padding)
            emb = hidden[:, -1, :].detach().float().cpu()
            if pmap_pad > 0:
                emb = emb[:-pmap_pad]
            outputs.append(emb)
            mu, sigma = stats
            mu_last = mu.detach().float().cpu().view(-1, 1)
            sigma_last = sigma.detach().float().cpu().view(-1, 1)
            if pmap_pad > 0:
                mu_last = mu_last[:-pmap_pad]
                sigma_last = sigma_last[:-pmap_pad]
            mus.append(mu_last)
            sigmas.append(sigma_last)
        return torch.cat(outputs, dim=0), torch.cat(mus, dim=0), torch.cat(sigmas, dim=0)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_contexts_from_lotsa_cache(
    cache_dir: Path,
    union_indices: list[int],
    hf_cache_dir: str | None,
) -> tuple[np.ndarray, str, int, dict[str, Any]]:
    backbone_path = sorted(cache_dir.glob("backbone_emb_c*_*.pt"))[0]
    payload = torch.load(backbone_path, map_location="cpu", weights_only=False)
    freq = str(payload.get("freq") or payload.get("frequency") or "")
    context_len = int(payload.get("context_len") or re.search(r"backbone_emb_c(\d+)_", backbone_path.name).group(1))
    series_ids = payload.get("series_ids")
    win_starts = payload.get("win_starts")
    if series_ids is None or win_starts is None:
        raise ValueError(f"source cache lacks series_ids/win_starts: {backbone_path}")
    subset = cache_dir.name
    dataset = load_dataset("Salesforce/lotsa_data", subset, split="train", streaming=False, cache_dir=hf_cache_dir)
    series_cache: dict[int, np.ndarray] = {}
    contexts: list[np.ndarray] = []
    union_idx = torch.as_tensor(union_indices, dtype=torch.long)
    for idx in union_indices:
        series_id = int(series_ids[idx])
        if series_id not in series_cache:
            series_cache[series_id] = target_values(dataset[series_id])
        sliced = slice_context(series_cache[series_id], int(win_starts[idx]), context_len)
        if sliced is None:
            raise ValueError(f"invalid context slice: {cache_dir} idx={idx}")
        contexts.append(sliced[0])
    meta = {
        "source_backbone": str(backbone_path),
        "series_ids": [int(series_ids[idx]) for idx in union_indices],
        "win_starts": torch.as_tensor([int(win_starts[idx]) for idx in union_indices], dtype=torch.long),
        "mu": payload["mu"].index_select(0, union_idx).contiguous(),
        "sigma": payload["sigma"].index_select(0, union_idx).contiguous(),
        "freq": freq,
        "frequency": freq,
        "context_len": context_len,
    }
    del dataset
    return np.stack(contexts, axis=0), freq, context_len, meta


def find_synth_source_npz(ds_dir: Path) -> Path:
    root = ds_dir.parents[1]
    name = root.name
    if name.startswith("func_dec_syn_cent_fourier_all_train_cache"):
        source_root = root.parent / "func_dec_syn_cent_fourier_all_train"
    elif name.startswith("func_dec_syn_cent_fourier_all_eval_cache"):
        source_root = root.parent / "func_dec_syn_cent_fourier_all_eval"
    else:
        source_root = root.parent / name.replace("_cache_10_4_2_8", "")
    candidates = [
        source_root / "complex" / f"{ds_dir.name}.npz",
        source_root / f"{ds_dir.name}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"synthetic source npz not found for {ds_dir}; tried {candidates}")


def load_synth_contexts(ds_dir: Path, indices: list[int]) -> tuple[np.ndarray, str, int]:
    meta_match = re.match(
        r"^.+_(?P<granularity>\w+)_seed\d+_c(?P<context_len>\d+)_h(?P<horizon>\d+)$",
        ds_dir.name,
    )
    if not meta_match:
        raise ValueError(f"cannot parse synthetic cache name: {ds_dir.name}")
    context_len = int(meta_match.group("context_len"))
    freq = str(meta_match.group("granularity"))
    source_path = find_synth_source_npz(ds_dir)
    with np.load(source_path, allow_pickle=False) as data:
        signal = data["signal"].astype(np.float32, copy=False)
    return signal[indices, :context_len], freq, context_len


def save_v1_backbone(
    dst_path: Path,
    contexts: np.ndarray,
    freq: str,
    context_len: int,
    encoder: TimesFmV1Encoder,
    extra: dict[str, Any] | None = None,
) -> None:
    emb, mu, sigma = encoder.encode(contexts, freq)
    if extra and "mu" in extra and "sigma" in extra:
        mu = extra["mu"]
        sigma = extra["sigma"]
    payload: dict[str, Any] = {
        "embeddings": emb,
        "mu": mu.float() if torch.is_tensor(mu) else mu,
        "sigma": sigma.float() if torch.is_tensor(sigma) else sigma,
        "freq": str(freq),
        "frequency": str(freq),
        "context_len": int(context_len),
        "backbone": "timesfm-1.0-200m-pytorch",
    }
    if extra:
        payload.update(extra)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst_path)
    print(f"[v1 saved] {dst_path} rows={emb.shape[0]}", flush=True)


def _missing_backbone_dsts(cache_dir: Path, src_root: Path, dst_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(cache_dir.glob("backbone_emb_c*_*.pt"))
        if not compact.dst_for(path, src_root, dst_root).exists()
    ]


def copy_real_group_v1(
    cache_dir_s: str,
    indices_by_horizon: dict[int, set[int]],
    src_root: Path,
    dst_root: Path,
    encoder: TimesFmV1Encoder,
    hf_cache_dir: str | None,
    prefetched: tuple | None = None,
) -> None:
    cache_dir = Path(cache_dir_s)
    missing_backbones = _missing_backbone_dsts(cache_dir, src_root, dst_root)
    union_indices = sorted({idx for indices in indices_by_horizon.values() for idx in indices})
    compact_pos = {original_idx: pos for pos, original_idx in enumerate(union_indices)}
    if missing_backbones:
        if prefetched is not None:
            contexts, freq, context_len, meta = prefetched
        else:
            contexts, freq, context_len, meta = load_contexts_from_lotsa_cache(cache_dir, union_indices, hf_cache_dir)
        for path in missing_backbones:
            save_v1_backbone(compact.dst_for(path, src_root, dst_root), contexts, freq, context_len, encoder, meta)
    else:
        print(f"[v1 reuse] {cache_dir.name} backbone already encoded; refreshing non-backbone files", flush=True)
    for path in sorted(cache_dir.glob("futures_c*_h*_*.pt")):
        horizon = compact.horizon_from_future_path(path)
        dst = compact.dst_for(path, src_root, dst_root)
        if horizon is None or horizon not in indices_by_horizon:
            copy_file(path, dst)
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        selected = sorted(indices_by_horizon[horizon])
        selected_idx = torch.as_tensor(selected, dtype=torch.long)
        selected_futures = payload["futures_n"].index_select(0, selected_idx).contiguous()
        compact_futures = torch.zeros((len(union_indices), *selected_futures.shape[1:]), dtype=selected_futures.dtype)
        compact_valid = torch.zeros(len(union_indices), dtype=torch.bool)
        compact_rows = torch.as_tensor([compact_pos[i] for i in selected], dtype=torch.long)
        compact_futures.index_copy_(0, compact_rows, selected_futures)
        compact_valid.index_fill_(0, compact_rows, True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"futures_n": compact_futures, "valid_mask": compact_valid}, dst)
    sliced = set(cache_dir.glob("backbone_emb_c*_*.pt")) | set(cache_dir.glob("futures_c*_h*_*.pt"))
    for path in sorted(p for p in cache_dir.iterdir() if p.is_file() and p not in sliced and p.name != "valid_mask.pt"):
        copy_file(path, compact.dst_for(path, src_root, dst_root))


def copy_synth_group_v1(ds_dir_s: str, indices_s: set[int], src_root: Path, dst_root: Path, encoder: TimesFmV1Encoder) -> None:
    ds_dir = Path(ds_dir_s)
    backbone_paths = sorted(ds_dir.glob("backbone_emb_c*_h*_stride1.pt"))
    missing_backbones = [
        path
        for path in backbone_paths
        if not compact.dst_for(path, src_root, dst_root).exists()
    ]
    indices = sorted(indices_s)
    idx = torch.as_tensor(indices, dtype=torch.long)
    if missing_backbones:
        contexts, freq, context_len = load_synth_contexts(ds_dir, indices)
        for path in missing_backbones:
            src_backbone = torch.load(path, map_location="cpu", weights_only=False)
            extra = compact.tensor_index_payload(src_backbone, idx, ["mu", "sigma"])
            save_v1_backbone(compact.dst_for(path, src_root, dst_root), contexts, freq, context_len, encoder, extra)
    elif backbone_paths:
        print(f"[v1 reuse] {ds_dir.name} backbone already encoded; refreshing target files", flush=True)
    for paths, keys in [
        (sorted(ds_dir.glob("raw_futures_h*.pt")), ["futures_n", "valid_mask"]),
        (sorted(ds_dir.glob("component_targets_h*.pt")), ["trend_n", "seasonal_n"]),
    ]:
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            out = compact.tensor_index_payload(payload, idx, keys)
            dst = compact.dst_for(path, src_root, dst_root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.save(out, dst)
    coeff_paths = sorted(ds_dir.glob("seasonal_coefficients*.pt"))
    for path in coeff_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        out = compact._slice_tensor_dict(payload, idx)
        dst = compact.dst_for(path, src_root, dst_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, dst)
    sliced = (
        set(ds_dir.glob("backbone_emb_c*_h*_stride1.pt"))
        | set(ds_dir.glob("raw_futures_h*.pt"))
        | set(ds_dir.glob("component_targets_h*.pt"))
        | set(coeff_paths)
    )
    for path in sorted(p for p in ds_dir.iterdir() if p.is_file() and p not in sliced):
        copy_file(path, compact.dst_for(path, src_root, dst_root))


def prepare_train_cache(args: argparse.Namespace, encoder: TimesFmV1Encoder) -> None:
    base = compact.import_experiment_train(args.src_data_root, args.exp_dir)
    plan = compact.simulate_schedule(base, args)
    compact.print_row_summary(plan)
    args.dst_data_root.mkdir(parents=True, exist_ok=True)

    real_items = [
        (cache_dir, rows)
        for cache_dir, rows in sorted(compact.grouped_real_needed(plan.real_needed).items())
        if rows
    ]

    num_workers = max(1, int(getattr(args, "num_workers", 1)))

    def _load_real(cache_dir_s: str, rows: dict) -> tuple:
        cache_dir = Path(cache_dir_s)
        union_indices = sorted({idx for indices in rows.values() for idx in indices})
        return load_contexts_from_lotsa_cache(cache_dir, union_indices, args.hf_cache_dir)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        # Find the first item that actually needs encoding to pre-submit
        pending: Future | None = None
        pending_idx: int = -1
        for j, (cd, _) in enumerate(real_items):
            if _missing_backbone_dsts(Path(cd), args.src_data_root, args.dst_data_root):
                pending = pool.submit(_load_real, cd, real_items[j][1])
                pending_idx = j
                break

        for i, (cache_dir, rows) in enumerate(real_items):
            missing_current = bool(_missing_backbone_dsts(Path(cache_dir), args.src_data_root, args.dst_data_root))
            print(f"[train real] {Path(cache_dir).name}", flush=True)

            # Get prefetched data (blocks until ready — overlapped with previous GPU encode)
            if missing_current and pending is not None and pending_idx == i:
                prefetched = pending.result()
                pending = None
            else:
                prefetched = None  # fallback: load synchronously inside copy_real_group_v1

            # Pre-submit NEXT item's I/O load BEFORE starting GPU encode
            # so data loading overlaps with the GPU work below
            if pending is None:
                for j in range(i + 1, len(real_items)):
                    next_cd, next_rows = real_items[j]
                    if _missing_backbone_dsts(Path(next_cd), args.src_data_root, args.dst_data_root):
                        pending = pool.submit(_load_real, next_cd, next_rows)
                        pending_idx = j
                        break

            copy_real_group_v1(cache_dir, rows, args.src_data_root, args.dst_data_root, encoder, args.hf_cache_dir, prefetched)
            gc.collect()

        if pending is not None:
            pending.cancel()

    for ds_dir, rows in sorted(plan.synth_needed.items()):
        if rows:
            print(f"[train synth] {Path(ds_dir).name}", flush=True)
            copy_synth_group_v1(ds_dir, rows, args.src_data_root, args.dst_data_root, encoder)
            gc.collect()


def copy_tree_without_backbones(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        print(f"[eval copy] destination exists; files may be overwritten: {dst_root}", flush=True)
    for path in src_root.rglob("*"):
        rel = path.relative_to(src_root)
        dst = dst_root / rel
        if path.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif not path.name.startswith("backbone_emb"):
            copy_file(path, dst)


def recache_eval_backbone(cache_dir: Path, real_root: Path, dst_root: Path, encoder: TimesFmV1Encoder, hf_cache_dir: str | None) -> None:
    backbone_paths = sorted(cache_dir.glob("backbone_emb*.pt"))
    if not backbone_paths:
        return
    backbone = torch.load(backbone_paths[0], map_location="cpu", weights_only=False)
    context_len = int(backbone.get("context_len", 512))
    freq = str(backbone.get("freq") or backbone.get("frequency") or "hourly")
    row_count = int(backbone["embeddings"].shape[0])
    contexts: np.ndarray | None = None

    raw_path = cache_dir / "raw.parquet"
    if raw_path.exists():
        raw_df = pd.read_parquet(raw_path)
        numeric_cols = [c for c in raw_df.columns if c != "date"]
        col_ids = backbone.get("col_ids", numeric_cols)
        win_starts = backbone.get("win_starts")
        contexts = np.stack([
            raw_df[col_ids[i] if col_ids[i] in raw_df.columns else numeric_cols[i % len(numeric_cols)]]
            .iloc[int(win_starts[i]): int(win_starts[i]) + context_len]
            .to_numpy(dtype=np.float32)
            for i in range(row_count)
        ], axis=0)
    elif (cache_dir / "sample_indices.pt").exists():
        sample = torch.load(cache_dir / "sample_indices.pt", map_location="cpu", weights_only=False)
        _src_bb_path = Path(str(sample["source_backbone"]))
        _old_prefix = Path("/home/sia2/project/data")
        try:
            _src_bb_path = DEFAULT_SRC_DATA_ROOT / _src_bb_path.relative_to(_old_prefix)
        except ValueError:
            pass
        source_backbone = torch.load(_src_bb_path, map_location="cpu", weights_only=False)
        source_indices = sample["source_indices"].long()
        series_ids = source_backbone["series_ids"]
        win_starts = source_backbone["win_starts"]
        dataset_name = cache_dir.name
        dataset = load_dataset("Salesforce/lotsa_data", dataset_name, split="train", streaming=False, cache_dir=hf_cache_dir)
        series_cache: dict[int, np.ndarray] = {}
        rows = []
        for source_idx_t in source_indices:
            source_idx = int(source_idx_t)
            series_id = int(series_ids[source_idx])
            if series_id not in series_cache:
                series_cache[series_id] = target_values(dataset[series_id])
            sliced = slice_context(series_cache[series_id], int(win_starts[source_idx]), context_len)
            if sliced is None:
                raise ValueError(f"invalid eval source slice: {cache_dir} source_idx={source_idx}")
            rows.append(sliced[0])
        contexts = np.stack(rows, axis=0)
    elif (real_root / "cloudops_index.parquet").exists() and cache_dir.name == "alibaba_cluster_trace_2018":
        index = pd.read_parquet(real_root / "cloudops_index.parquet")
        dataset = load_dataset("Salesforce/lotsa_data", cache_dir.name, split="train", streaming=False, cache_dir=hf_cache_dir)
        series_cache = {}
        rows = []
        for i in range(row_count):
            row = index.iloc[i]
            series_id = int(row["series_id"])
            if series_id not in series_cache:
                series_cache[series_id] = target_values(dataset[series_id])
            sliced = slice_context(series_cache[series_id], int(row["win_start"]), context_len)
            if sliced is None:
                raise ValueError(f"invalid cloudops eval slice: {cache_dir} idx={i}")
            rows.append(sliced[0])
        contexts = np.stack(rows, axis=0)
    if contexts is None:
        raise ValueError(f"cannot reconstruct eval contexts for {cache_dir}")

    extra = {
        key: value
        for key, value in backbone.items()
        if key not in {"embeddings", "freq", "frequency", "context_len"}
    }
    for src_path in backbone_paths:
        dst_path = dst_root / src_path.relative_to(real_root)
        save_v1_backbone(dst_path, contexts, freq, context_len, encoder, extra)
        context_path = dst_path.parent / f"raw_contexts_c{context_len}.pt"
        if not context_path.exists():
            torch.save({"contexts": torch.from_numpy(contexts).float(), "context_len": context_len}, context_path)


def prepare_real_eval_cache(args: argparse.Namespace, encoder: TimesFmV1Encoder) -> None:
    if not args.real_eval_root.exists():
        raise FileNotFoundError(args.real_eval_root)
    copy_tree_without_backbones(args.real_eval_root, args.real_eval_dst_root)
    for cache_dir in sorted({path.parent for path in args.real_eval_root.rglob("backbone_emb*.pt")}):
        print(f"[eval recache] {cache_dir.relative_to(args.real_eval_root)}", flush=True)
        recache_eval_backbone(cache_dir, args.real_eval_root, args.real_eval_dst_root, encoder, args.hf_cache_dir)
        gc.collect()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    args.src_data_root = args.src_data_root.resolve()
    args.dst_data_root = args.dst_data_root.resolve()
    args.domain_config = args.domain_config.resolve()
    args.exp_dir = args.exp_dir.resolve()
    args.real_eval_root = args.real_eval_root.resolve()
    args.real_eval_dst_root = args.real_eval_dst_root.resolve()
    if args.fourier_batch_size is None:
        args.fourier_batch_size = args.batch_size
    if args.lotsa_cache_root is None:
        args.lotsa_cache_root = args.src_data_root / "data_lotsa" / "lotsa_cache"
    else:
        args.lotsa_cache_root = args.lotsa_cache_root.resolve()
    if args.src_data_root == args.dst_data_root:
        raise SystemExit("--src_data_root and --dst_data_root must differ")
    if not args.src_data_root.exists():
        raise SystemExit(f"src_data_root not found: {args.src_data_root}")
    if not args.lotsa_cache_root.exists():
        raise SystemExit(f"lotsa_cache_root not found: {args.lotsa_cache_root}")
    if not args.exp_dir.exists():
        raise SystemExit(f"exp_dir not found: {args.exp_dir}")
    if args.checkpoint_path.exists():
        print(f"[v1] checkpoint={args.checkpoint_path}", flush=True)
    else:
        print(f"[v1] checkpoint path missing; will download {MODEL_REPO}", flush=True)

    encoder = TimesFmV1Encoder(
        args.v1_src,
        args.checkpoint_path.resolve(),
        args.hf_cache_dir,
        resolve_device(args.device),
        args.encode_batch_size,
    )
    if not args.skip_train_cache:
        prepare_train_cache(args, encoder)
    if not args.skip_real_eval_cache:
        prepare_real_eval_cache(args, encoder)
    print(f"[done] elapsed={time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
