# Reproduction notes

## Metrics

Reported class-wise F1 values are computed after binarizing out-of-fold (OOF) probabilities with a selected threshold:

- `macro-F1 = (F1_match + F1_mismatch) / 2`
- selection objective: maximize `min(F1_match, F1_mismatch)`

Non-golden arbiter tuning uses **5 repeats × 5 folds** of `StratifiedGroupKFold`, keeping paired original/failure figures in the same fold. Per-sample OOF scores are averaged across repeats before threshold selection. See `balanced_arbiter_tuning.py` and `TRAIN_METRICS.md`.

## Historical vs current numbers

Some older notes mention ~5,465 feature rows and a 300-example golden snapshot. The current figure corpus and golden holdout sizes may differ (see `ablations_notes.md` and `data/README.md`). Prefer the tables in `TRAIN_METRICS.md` when citing quantitative results from this codebase.

## Suggested local checklist

1. Install dependencies (`requirements.txt` + PyTorch + `en_core_web_sm`).
2. Download the public dataset from Hugging Face:
   ```bash
   pip install huggingface_hub
   python scripts/download_data.py
   ```
   Dataset: [Saintghetto17/verification-before-assembly](https://huggingface.co/datasets/Saintghetto17/verification-before-assembly)
3. Place `router.pth` and `sem.pt` under `checkpoints/`.
4. Point a config in `configs/` at your data root.
5. Run `./run_agent.sh --mode eval --exp-name <exp>`.
6. Optionally retune the arbiter with `balanced_arbiter_tuning.py` on non-golden features.
