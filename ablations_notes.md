# Ablation and evaluation notes

## Protocol

- Golden evaluation is held out and never used for training, threshold search, or model selection.
- Non-golden tuning uses repeated **5×5 Stratified Group K-Fold** (paired orig/fail stay together).
- Primary selection objective: maximize `min(F1_match, F1_mismatch)`, then macro-F1 / accuracy.
- Macro-F1 is the unweighted mean of the two class F1 scores.
- Soft router probabilities reweight judge scores; they do not hard-gate judges.

## Dataset caveats

- `bt` mismatches exist without paired originals.
- Some historical artifacts used different golden sizes (e.g. n=300) or larger feature dumps; prefer current tables in `TRAIN_METRICS.md`.
- Irreducible label contradictions and domain leakage risks are audited in the research workspace (`data_audit_non_golden.py`).

## Selected direction

The selected non-golden configuration uses a pair-aware semantic judge plus a balanced arbiter ensemble (shallow XGBoost + logistic regression), retaining router and Judge C features when they improve min-class F1. See `TRAIN_METRICS.md` (J03) and `balanced_arbiter_tuning.py`.

## Baselines

Standalone VLM / VQAScore baselines are documented in `BASELINE_EVAL.md` and `baseline_eval.py`.
