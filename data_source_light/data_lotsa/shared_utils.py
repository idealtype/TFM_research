from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SORT_KEYS = ["subset_name", "freq", "context_len", "series_id", "win_start"]
INDEX_COLUMNS = ("subset_name", "series_id", "win_start", "freq", "context_len", "max_horizon")
CONTEXT_BY_FREQ = {
    "4_seconds": 512,
    "minutely": 512,
    "5_minutes": 512,
    "10_minutes": 512,
    "15_minutes": 512,
    "half_hourly": 512,
    "hourly": 512,
    "daily": 512,
    "weekly": 256,
    "monthly": 64,
    "quarterly": 64,
    "yearly": 64,
}
FREQ_DAYS = {
    "4_seconds": 1 / 21600,
    "minutely": 1 / 1440,
    "5_minutes": 1 / 288,
    "10_minutes": 1 / 144,
    "15_minutes": 1 / 96,
    "half_hourly": 1 / 48,
    "hourly": 1 / 24,
    "daily": 1.0,
    "weekly": 7.0,
    "monthly": 30.4375,
    "quarterly": 91.3125,
    "yearly": 365.25,
}
HF_FREQ_ALIASES = {
    "4S": "4_seconds",
    "4s": "4_seconds",
    "T": "minutely",
    "min": "minutely",
    "1T": "minutely",
    "1min": "minutely",
    "5T": "5_minutes",
    "5min": "5_minutes",
    "10T": "10_minutes",
    "10min": "10_minutes",
    "15T": "15_minutes",
    "15min": "15_minutes",
    "30T": "half_hourly",
    "30min": "half_hourly",
    "H": "hourly",
    "h": "hourly",
    "D": "daily",
    "W": "weekly",
    "W-SUN": "weekly",
    "W-MON": "weekly",
    "M": "monthly",
    "MS": "monthly",
    "ME": "monthly",
    "Q": "quarterly",
    "QS": "quarterly",
    "QE": "quarterly",
    "Q-DEC": "quarterly",
    "QS-DEC": "quarterly",
    "A": "yearly",
    "Y": "yearly",
    "AS": "yearly",
    "YS": "yearly",
    "A-DEC": "yearly",
    "Y-DEC": "yearly",
}


@dataclass(frozen=True)
class IndexRow:
    subset_name: str
    series_id: int
    win_start: int
    freq: str
    context_len: int
    max_horizon: int


def index_schema() -> pa.Schema:
    return pa.schema(
        [
            ("subset_name", pa.string()),
            ("series_id", pa.int64()),
            ("win_start", pa.int64()),
            ("freq", pa.string()),
            ("context_len", pa.int32()),
            ("max_horizon", pa.int32()),
        ]
    )


def row_sort_key(row: IndexRow) -> tuple:
    return tuple(getattr(row, key) for key in SORT_KEYS)


def rows_to_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=index_schema())


def normalize_freq(value: Any) -> str | None:
    if value is None:
        return None
    freq = str(value)
    return HF_FREQ_ALIASES.get(freq, freq)


def row_freq(row: dict[str, Any]) -> str | None:
    return normalize_freq(row.get("freq", row.get("frequency")))


def target_values(row: dict[str, Any], variate_idx: int = 0) -> np.ndarray:
    target = row["target"]
    if len(target) == 0:
        return np.asarray([], dtype=np.float32)
    first = target[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        if len(target) <= variate_idx:
            raise ValueError(f"target has no variate index {variate_idx}")
        values = target[variate_idx]
    else:
        values = target
    return np.asarray(values, dtype=np.float32)


def series_len(row: dict[str, Any]) -> int:
    return int(len(target_values(row)))


def iter_index_rows(index_path: Path, columns: tuple[str, ...] = INDEX_COLUMNS) -> Iterator[IndexRow]:
    parquet_file = pq.ParquetFile(index_path)
    for batch in parquet_file.iter_batches(columns=list(columns)):
        data = pa.Table.from_batches([batch]).to_pydict()
        for i in range(len(data["subset_name"])):
            yield IndexRow(
                subset_name=str(data["subset_name"][i]),
                series_id=int(data["series_id"][i]),
                win_start=int(data["win_start"][i]),
                freq=str(data["freq"][i]),
                context_len=int(data["context_len"][i]),
                max_horizon=int(data["max_horizon"][i]),
            )


def load_sorted_index_rows(index_path: Path) -> list[IndexRow]:
    rows = list(iter_index_rows(index_path))
    rows.sort(key=row_sort_key)
    return rows


def group_sorted_rows(rows: list[IndexRow]) -> dict[tuple[str, str, int], list[IndexRow]]:
    grouped: dict[tuple[str, str, int], list[IndexRow]] = {}
    for row in rows:
        grouped.setdefault((row.subset_name, row.freq, row.context_len), []).append(row)
    return grouped


def subset_names_from_rows(rows: list[IndexRow]) -> list[str]:
    return sorted({row.subset_name for row in rows})


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"
