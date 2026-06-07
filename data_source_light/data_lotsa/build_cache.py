from __future__ import annotations

import argparse
import os
import gc
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from shared_utils import IndexRow, format_bytes, group_sorted_rows, load_sorted_index_rows, target_values


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/workspace")) / "4.28basis"
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

VARIATE_IDX = 0
MODEL_ID = "google/timesfm-2.5-200m-pytorch"
EMBED_DIM = 1280
PATCH_LEN = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="lotsa_index.parquet")
    parser.add_argument("--output_dir", type=str, default="lotsa_cache/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is not available.")
    return torch.device(device_name)


def load_backbone(device: torch.device, hf_cache_dir: str | None):
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    from timesfm.torch.util import revin, update_running_stats

    pretrained = TimesFM_2p5_200M_torch.from_pretrained(
        MODEL_ID,
        torch_compile=False,
        cache_dir=hf_cache_dir,
    )
    backbone = pretrained.model.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    if backbone.p != PATCH_LEN or backbone.md != EMBED_DIM:
        raise ValueError(f"Unexpected TimesFM shape: patch_len={backbone.p}, embed_dim={backbone.md}")
    return backbone, revin, update_running_stats


def load_subset(subset_name: str, hf_cache_dir: str | None):
    return load_dataset(
        "Salesforce/lotsa_data",
        subset_name,
        split="train",
        streaming=False,
        cache_dir=hf_cache_dir,
    )


def cache_path(output_dir: Path, subset_name: str, freq: str, context_len: int) -> Path:
    return output_dir / subset_name / f"backbone_emb_c{context_len}_{freq}_lotsa.pt"


def slice_context(series: np.ndarray, win_start: int, context_len: int) -> tuple[np.ndarray, np.ndarray] | None:
    context_end = win_start + context_len
    if context_end > len(series):
        return None

    context = series[max(0, win_start):context_end]
    mask = np.zeros(context_len, dtype=np.bool_)
    if len(context) < context_len:
        pad_len = context_len - len(context)
        context = np.pad(context, (pad_len, 0), mode="constant")
        mask[:pad_len] = True
    return context.astype(np.float32, copy=False), mask


def encode_batch(
    contexts: torch.Tensor,
    masks: torch.Tensor,
    backbone,
    revin_fn,
    update_stats_fn,
    device: torch.device,
):
    batch_n, context_len = contexts.shape
    if context_len % PATCH_LEN != 0:
        raise ValueError(f"context_len={context_len} must be divisible by patch_len={PATCH_LEN}")

    contexts = contexts.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)
    patched_inputs = contexts.reshape(batch_n, -1, PATCH_LEN)
    patched_masks = masks.reshape(batch_n, -1, PATCH_LEN)
    n = torch.zeros(batch_n, device=device)
    mu = torch.zeros(batch_n, device=device)
    sigma = torch.zeros(batch_n, device=device)
    patch_mu = []
    patch_sigma = []

    for patch_idx in range(context_len // PATCH_LEN):
        (n, mu, sigma), _ = update_stats_fn(
            n,
            mu,
            sigma,
            patched_inputs[:, patch_idx],
            patched_masks[:, patch_idx],
        )
        patch_mu.append(mu)
        patch_sigma.append(sigma)

    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)
    normed_inputs = revin_fn(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

    with torch.no_grad():
        (_, output_embeddings, _, _), _ = backbone(normed_inputs, patched_masks)

    return output_embeddings[:, -1, :].float().cpu(), context_mu[:, -1:].float().cpu(), context_sigma[:, -1:].float().cpu()


def flush_batch(
    contexts: list[np.ndarray],
    masks: list[np.ndarray],
    embeddings_out: list[torch.Tensor],
    mu_out: list[torch.Tensor],
    sigma_out: list[torch.Tensor],
    backbone,
    revin_fn,
    update_stats_fn,
    device: torch.device,
) -> None:
    if not contexts:
        return
    context_tensor = torch.from_numpy(np.stack(contexts, axis=0))
    mask_tensor = torch.from_numpy(np.stack(masks, axis=0))
    emb, mu, sigma = encode_batch(
        context_tensor,
        mask_tensor,
        backbone,
        revin_fn,
        update_stats_fn,
        device,
    )
    embeddings_out.append(emb)
    mu_out.append(mu)
    sigma_out.append(sigma)
    contexts.clear()
    masks.clear()


def process_context_group(
    subset_name: str,
    freq: str,
    context_len: int,
    rows: list[IndexRow],
    dataset,
    output_path: Path,
    batch_size: int,
    backbone,
    revin_fn,
    update_stats_fn,
    device: torch.device,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"[skip] {output_path}")
        return 0
    if any(row.freq != freq for row in rows):
        raise ValueError(f"Mixed freq rows passed to {output_path}")

    embeddings: list[torch.Tensor] = []
    mus: list[torch.Tensor] = []
    sigmas: list[torch.Tensor] = []
    contexts: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    win_starts: list[int] = []
    series_ids: list[int] = []

    current_series_id: int | None = None
    current_series: np.ndarray | None = None
    skipped = 0

    for row in tqdm(rows, desc=f"{subset_name} c{context_len}", leave=False):
        if row.series_id != current_series_id:
            current_series_id = row.series_id
            current_series = target_values(dataset[row.series_id], VARIATE_IDX)
        assert current_series is not None
        sliced = slice_context(current_series, row.win_start, context_len)
        if sliced is None:
            skipped += 1
            continue

        context, mask = sliced
        contexts.append(context)
        masks.append(mask)
        win_starts.append(row.win_start)
        series_ids.append(row.series_id)

        if len(contexts) >= batch_size:
            flush_batch(
                contexts,
                masks,
                embeddings,
                mus,
                sigmas,
                backbone,
                revin_fn,
                update_stats_fn,
                device,
            )

    flush_batch(
        contexts,
        masks,
        embeddings,
        mus,
        sigmas,
        backbone,
        revin_fn,
        update_stats_fn,
        device,
    )

    if embeddings:
        emb_tensor = torch.cat(embeddings, dim=0)
        mu_tensor = torch.cat(mus, dim=0)
        sigma_tensor = torch.cat(sigmas, dim=0)
    else:
        emb_tensor = torch.empty((0, EMBED_DIM), dtype=torch.float32)
        mu_tensor = torch.empty((0, 1), dtype=torch.float32)
        sigma_tensor = torch.empty((0, 1), dtype=torch.float32)

    torch.save(
        {
            "embeddings": emb_tensor,
            "mu": mu_tensor,
            "sigma": sigma_tensor,
            "win_starts": torch.tensor(win_starts, dtype=torch.long),
            "series_ids": series_ids,
            "col_ids": series_ids,
            "freq": str(freq),
            "frequency": str(freq),
            "context_len": int(context_len),
        },
        output_path,
    )
    print(f"[saved] {output_path} windows={len(series_ids)} skipped={skipped}")
    return len(series_ids)


def total_cache_bytes(output_dir: Path) -> int:
    return sum(path.stat().st_size for path in output_dir.rglob("backbone_emb_c*_lotsa.pt"))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    index_path = Path(args.index)
    output_dir = Path(args.output_dir)
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    rows = load_sorted_index_rows(index_path)
    if not rows:
        raise ValueError(f"No index rows found in {index_path}")
    grouped = group_sorted_rows(rows)

    device = resolve_device(args.device)
    backbone, revin_fn, update_stats_fn = load_backbone(device, args.hf_cache_dir)

    total_index_windows = len(rows)
    windows_written = 0
    indexed_by_subset: Counter[str] = Counter(row.subset_name for row in rows)
    written_by_subset: Counter[str] = Counter()

    subsets = sorted({subset_name for subset_name, _freq, _context_len in grouped})
    for subset_name in tqdm(subsets, desc="subsets"):
        subset_groups = {
            (freq, context_len): group_rows
            for (group_subset, freq, context_len), group_rows in grouped.items()
            if group_subset == subset_name
        }
        pending = {
            group_key: group_rows
            for group_key, group_rows in subset_groups.items()
            if not cache_path(output_dir, subset_name, group_key[0], group_key[1]).exists()
        }
        if not pending:
            print(f"[skip] {subset_name}: cache exists")
            continue

        dataset = load_subset(subset_name, args.hf_cache_dir)
        try:
            for (freq, context_len), group_rows in sorted(pending.items()):
                out_path = cache_path(output_dir, subset_name, freq, context_len)
                count = process_context_group(
                    subset_name,
                    freq,
                    context_len,
                    group_rows,
                    dataset,
                    out_path,
                    args.batch_size,
                    backbone,
                    revin_fn,
                    update_stats_fn,
                    device,
                )
                windows_written += count
                written_by_subset[subset_name] += count
        finally:
            del dataset
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    total_size = total_cache_bytes(output_dir)
    print("\nCache build summary:")
    print(f"  index_windows: {total_index_windows}")
    print(f"  windows_written: {windows_written}")
    print(f"  cache_size: {format_bytes(total_size)}")
    print("  subset_index_windows:")
    for subset_name, count in sorted(indexed_by_subset.items()):
        written = written_by_subset[subset_name]
        print(f"    {subset_name}: indexed={count} written={written}")


if __name__ == "__main__":
    main()
