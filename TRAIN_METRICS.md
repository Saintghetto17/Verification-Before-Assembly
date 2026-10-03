# Experiment metrics summary

Notation: `M` = `match/PASS`, `R` = `mismatch/REGENERATE`.
Columns `P/R/F1` are precision/recall/F1 for that class.
Golden has 294 examples; robust non-golden CV uses 558 `juri`/`ml` examples.
The same `id` in both tables refers to the same experiment on different evaluation sets.
`Group` lists which judges contributed arbiter features; `standalone` means no arbiter.

Judge A variants: `A_orig` — original retrained Judge A; `A_bce` — weighted BCE, 128 tokens; `A_focal` — focal loss, 128 tokens; `A_pair` — pair-aware BCE + ranking, 128 tokens. `B` = Judge B (OCR/struct), `C` = Judge C (objects), `router` = image-type router.

## Golden experiments and baselines

| id | Group (judges for features) | Experiment | Eval | Accuracy | M: P/R/F1 | R: P/R/F1 | Macro-F1 | Min-F1 | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| H1 | A,B,C,router (old snapshot) | agent_v3 | Golden, n=300 | 0.6467 | 0.6009/0.8733/0.7120 | — | — | — | — |
| H2 | A,B,C,router (old snapshot) | agent_v5 | Golden, n=300 | 0.6300 | 0.5915/0.8400/0.6942 | — | — | — | — |
| F01 | A_orig,B,C,router | Fresh agent_v5 | Golden | 0.6327 | 0.5685/0.9930/0.7231 | 0.9783/0.2961/0.4545 | 0.5888 | 0.4545 | 0.7234 |
| A01 | A_orig,B,C,router | Full controlled pipeline | Golden | 0.6565 | 0.5936/0.9155/0.7202 | 0.8400/0.4145/0.5551 | 0.6377 | 0.5551 | 0.7037 |
| A02 | A_orig,B,C | Without router | Golden | 0.6497 | 0.5809/0.9859/0.7311 | 0.9623/0.3355/0.4976 | 0.6144 | 0.4976 | 0.7199 |
| A03 | B,C,router | Without Judge A | Golden | 0.6327 | 0.5850/0.8239/0.6842 | 0.7340/0.4539/0.5610 | 0.6226 | 0.5610 | 0.6935 |
| A04 | A_orig,C,router | Without Judge B | Golden | 0.6599 | 0.5921/0.9507/0.7297 | 0.8939/0.3882/0.5413 | 0.6355 | 0.5413 | 0.7049 |
| A05 | A_orig,B,router | Without Judge C | Golden | 0.6735 | 0.6045/0.9366/0.7348 | 0.8784/0.4276/0.5752 | 0.6550 | 0.5752 | 0.7158 |
| S01 | router only | Router only, no arbiter | Golden | 0.4830 | 0.4830/1.0000/0.6514 | 0/0/0 | 0.3257 | 0 | 0.5331 |
| S02 | A_orig only | Judge A only, no arbiter | Golden | 0.6667 | 0.6009/0.9225/0.7278 | 0.8553/0.4276/0.5702 | 0.6490 | 0.5702 | 0.6778 |
| S03 | B only | Judge B only, no arbiter | Golden | 0.4830 | 0.4830/1.0000/0.6514 | 0/0/0 | 0.3257 | 0 | 0.5104 |
| S04 | C only | Judge C only, no arbiter | Golden | 0.4966 | 0.4894/0.9789/0.6526 | 0.7000/0.0461/0.0864 | 0.3695 | 0.0864 | 0.5394 |
| S05 | A_orig,B,C (majority) | Majority vote A/B/C, no arbiter | Golden | 0.4932 | 0.4879/0.9930/0.6543 | 0.8000/0.0263/0.0510 | 0.3527 | 0.0510 | 0.6775 |
| B01 | Qwen-3B standalone | Qwen2.5-VL-3B-Instruct | Golden | 0.5272 | 0.5081/0.6620/0.5749 | 0.5596/0.4013/0.4674 | 0.5212 | 0.4674 | — |
| B02 | Qwen-4B standalone | Qwen3-VL-4B-Instruct | Golden | 0.5442 | 0.5156/0.9296/0.6633 | 0.7368/0.1842/0.2947 | 0.4790 | 0.2947 | — |
| B03 | Qwen-7B standalone | Qwen2.5-VL-7B-Instruct | Golden | 0.5544 | 0.5314/0.6549/0.5868 | 0.5882/0.4605/0.5166 | 0.5517 | 0.5166 | — |
| B04 | VQAScore standalone | VQAScore `clip-flant5-xxl` | Golden | 0.4796 | 0.4811/0.9859/0.6467 | 0.3333/0.0066/0.0129 | 0.3298 | 0.0129 | — |
| J03 | A_pair,B,C,router | Pair-aware A + XGB/LogReg 0.5/0.5 | Locked Golden | 0.5850 | 0.5806/0.5070/0.5414 | 0.5882/0.6579/0.6211 | 0.5812 | 0.5414 | 0.6707 |

`GenEval` produced no numeric result: 0 of 294 scientific captions match its
official simple photo-prompt format.

## Non-golden CV experiments

| id | Group (judges for features) | Experiment | Protocol | Accuracy | M: P/R/F1 | R: P/R/F1 | Macro-F1 | Min-F1 | ROC-AUC |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| F01 | A_orig,B,C,router | Original 37 features, including `bt` | 5×5 grouped CV | — | — | — | — | 0.7054 | — |
| F02 | A_orig,B,C | Original features, without `bt` and without router | 5×5 grouped CV, no `bt` | — | — | — | — | 0.6069 | — |
| J01 | A_bce,B,C | Weighted BCE Judge A, 128 tokens | 5×5 grouped CV, no `bt` | — | — | — | — | 0.6841 | — |
| J02 | A_focal,B,C | Focal-loss Judge A, 128 tokens | 5×5 grouped CV, no `bt` | — | — | — | — | 0.6777 | — |
| J03 | A_pair,B,C,router | Pair-aware A + XGB/LogReg 0.5/0.5 | 5×5 grouped CV, no `bt` | 0.8082 | 0.8851/0.7821/0.8304 | 0.7214/0.8475/0.7794 | 0.8049 | **0.7794** | 0.8729 |
| J04 | A_pair,B,C | Pair-aware A, without router features | 5×5 grouped CV, no `bt` | — | — | — | — | 0.7771 | — |
| J05 | A_pair,B,router | Pair-aware A, without Judge C features | 5×5 grouped CV, no `bt` | — | — | — | — | 0.7759 | — |
| J06 | A_pair,B,C,router + derived | Pair-aware A + derived interactions | 5×5 grouped CV, no `bt` | — | — | — | — | 0.7745 | — |
| J07 | A_pair,B,C,router + Qwen-7B | Pair-aware A + frozen Qwen-7B feature | 5×5 grouped CV, no `bt` | — | — | — | — | 0.7780 | — |
| B03 | Qwen-7B standalone | Qwen2.5-VL-7B-Instruct | Non-golden, n=691 | 0.5297 | 0.5117/0.6537/0.5740 | 0.5589/0.4129/0.4750 | 0.5245 | 0.4750 | — |
| B04 | VQAScore standalone | VQAScore at threshold 0.263958 | Non-golden, n=691 | 0.4877 | 0.4862/1.0000/0.6543 | 1.0000/0.0056/0.0112 | 0.3327 | 0.0112 | — |

## Arbiter search

| Candidate family | Count |
|---|---:|
| Logistic Regression, `C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10}` | 7 |
| Shallow XGBoost grids | 8 |
| XGB/LogReg ensembles with XGB weights 0.25, 0.5, 0.75 | 168 |
| Total candidates | **183** |

Winning configuration:

```text
P(match) = 0.5 × P(XGBoost xgb_05)
         + 0.5 × P(Logistic Regression, C=0.03)

PASS       if P(match) >= 0.5674605586
REGENERATE otherwise
```

CV fold models are not used at inference. XGBoost and Logistic Regression are
refit on all non-golden features; runtime loads these two models, mixes their
probabilities 0.5/0.5, and applies the frozen threshold.
The final arbiter does **not** include Qwen features.

## Where Qwen prompts and outputs live

| Artifact | Location |
|---|---|
| Qwen prompt | `baseline_eval.py`, constant `JUDGE_PROMPT`, lines 32–52 |
| Prompt construction / generation code | `baseline_eval.py`, lines 453–466 |
| Qwen2.5-VL-3B answers on Golden | `outputs/baselines/qwen25_3b_golden.jsonl` |
| Qwen3-VL-4B answers on Golden | `outputs/baselines/qwen3_4b_golden.jsonl` |
| Qwen2.5-VL-7B answers on Golden | `outputs/baselines/qwen25_7b_golden.jsonl` |
| Qwen2.5-VL-7B answers on non-golden | `outputs/baselines/qwen25_7b_dev_golden.jsonl` |
| Qwen2.5-VL-3B report | `outputs/baselines/qwen25_3b_report.json` |
| Qwen3-VL-4B report | `outputs/baselines/qwen3_4b_report.json` |
| Qwen2.5-VL-7B report | `outputs/baselines/qwen25_7b_report.json` |
| VQAScore raw scores on non-golden | `outputs/baselines/vqa_validation.jsonl` |
| VQAScore raw scores on Golden | `outputs/baselines/vqa_golden.jsonl` |
| VQAScore final report | `outputs/baselines/vqa_report.json` |

Qwen is asked to return JSON:

```json
{
  "correct_things": ["visible agreements"],
  "mistake_things": ["mismatches or unsupported claims"],
  "critical_things": ["mismatches severe enough to reject"],
  "verdict": "pass or regenerate"
}
```

Malformed JSON was treated as `REGENERATE`: 1 case for Qwen2.5-3B, 6 for
Qwen3-4B, and 0 for Qwen2.5-7B.
