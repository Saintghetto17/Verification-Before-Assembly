# Public sample pairs

Compact paired set for qualitative inspection (**5 originals + 5 failures**).

```text
01_l2_fail/          # corrupted figures
02_orig_paired/      # matching correct originals (same directory names)
```

Each sample directory contains only:

- `figure_*.png` — the image
- `figure_*.json` — caption / context, visual description & claims, and `gold.lie` for failures

These samples are intended for demos and prompt/engineering checks; they are **not** the full training or golden evaluation corpora used in the paper tables.

For the full public release (985 figures + router images), see:

**[Saintghetto17/verification-before-assembly](https://huggingface.co/datasets/Saintghetto17/verification-before-assembly)**

```bash
python scripts/download_data.py
```
