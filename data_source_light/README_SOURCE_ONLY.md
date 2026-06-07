# Data Source-Only Transfer Folder

This folder is a lightweight copy of `/workspace/data` for moving data
generation and cache-building code without large generated artifacts.

Included:
- Python generation/cache scripts
- shell runners
- YAML/JSON/TXT/MD config and manifest files
- original relative folder layout for source-bearing directories

Excluded:
- `.npz`, `.pt`, `.parquet`, `.png`, logs, caches, `__pycache__`
- generated synthetic datasets and TimesFM backbone cache tensors
- large real-data cache trees under `data_lotsa`, `synthetic`, and `synthetic_nonF`

Important path assumptions:
- Several cache scripts import TimesFM from `/workspace/4.28basis/src`.
- Many scripts have default roots under `/workspace/data/...`.
- In a new environment, either place this folder back at `/workspace/data`
  or pass explicit `--config`, `--input_root`, `--output_root`, `--data_root`,
  and `--cache_root` arguments.

Common dependencies:
- python
- numpy
- torch
- scipy
- pyyaml
- tqdm
- datasets
- pyarrow
- matplotlib
- PyWavelets, for `synthetic_nonF` wavelet models

Useful entry points:
- `synthetic/generate_all_synth.py`
- `synthetic/generate_complex_synth.py`
- `synthetic/prepare_synth_cache.py`
- `synthetic/prepare_complex_synth_cache.py`
- `synthetic_nonF/nonfourier_generator.py`
- `synthetic_nonF/generate_nonfourier_ts.py`
- `synthetic_nonF/generate_nonfourier_tsr.py`
- `synthetic_nonF/prepare_nonfourier_synth_cache.py`
- `synthetic_nonF/prepare_nonfourier_ts_cache.py`
- `synthetic_nonF/prepare_nonfourier_tsr_cache.py`
- `data_lotsa/build_index.py`
- `data_lotsa/build_futures.py`
- `data_lotsa/build_cache.py`

Minimal sanity checks after transfer:

```bash
cd /workspace/data
python synthetic/synth_generator.py --help
python synthetic_nonF/nonfourier_generator.py --help
python data_lotsa/build_index.py --help
```

For cache building with TimesFM, also verify:

```bash
python - <<'PY'
import sys
sys.path.insert(0, "/workspace/4.28basis/src")
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
print("timesfm import ok")
PY
```
