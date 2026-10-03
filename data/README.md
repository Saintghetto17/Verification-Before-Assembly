# Data

Scientific figure corpus used by **Verification Before Assembly**.

## Download (Hugging Face)

Full public release:

**[Saintghetto17/verification-before-assembly](https://huggingface.co/datasets/Saintghetto17/verification-before-assembly)**

```bash
# from repository root
pip install huggingface_hub
python scripts/download_data.py
```

This installs:

```text
data/figures/   ← HF `figures/`  (golden + train splits)
data_router/    ← HF `router/`   (domain-classifier images)
```

Or with the Hub API directly:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Saintghetto17/verification-before-assembly",
    repo_type="dataset",
    local_dir="hf_data",
)
```

## Local layout after download

```text
data/
  figures/
    golden/          # held-out evaluation only
    ml_orig/         # train match
    ml_l2_fail/      # train mismatch
    juri_orig/
    juri_l2_fail/
    bt_l2_fail/      # mismatches without paired originals
```

| Purpose | Path | Notes |
|---------|------|-------|
| Train / CV pool | `figures/**` excluding `golden` | 691 figures |
| Held-out eval | `figures/golden/` | 294 figures; never used for training or threshold search |
| Flat index | HF `metadata/figures_index.jsonl` | `sample_id`, label, caption, lie, paths |

A compact paired demo pack (independent of the full dump) is also available at [`../examples/check_generation/`](../examples/check_generation/).

See [`../docs/REPRODUCTION.md`](../docs/REPRODUCTION.md) and the [dataset card](https://huggingface.co/datasets/Saintghetto17/verification-before-assembly).
