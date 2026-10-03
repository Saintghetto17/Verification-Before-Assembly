# Architecture

This document describes the released **Approach A** verifier (Router → Judges → Arbiter).

## Pipeline

1. **Router** (`agent_backend/router.py`)  
   ConvNeXt-Tiny classifier over four image domains: `byt`, `video`, `graph`, `scheme`. Soft probabilities reweight judge features; they do not disable judges.

2. **Judge A — Semantic** (`agent_backend/judges/semantic.py`, `match_head.py`)  
   Shared `google/siglip-base-patch16-224` encodes image and text (64 tokens). A small MLP MatchHead consumes `[image, text, image×text, |image−text|]` and outputs `score_sem ∈ [0,1]`.

3. **Judge B — Structural / OCR** (`agent_backend/judges/structural.py`)  
   EasyOCR over the figure; Jaccard overlap with informative caption words; optional number-recall term when digits are present. Missing OCR → neutral `0.5`.

4. **Judge C — Objects** (`agent_backend/judges/object.py`)  
   Heuristic noun extraction → prompts like `"a figure showing encoder"` → zero-shot SigLIP similarities → average of top-3 as `score_obj`.

5. **Features** (`agent_backend/features.py`)  
   37 dimensions: raw judge scores, router probs, router-weighted scores, router confidence, image stats, and light text stats.

6. **Arbiter** (`agent_backend/arbiter.py`, `balanced_arbiter_tuning.py`)  
   Meta-classifier over features → `P(keep)`. Threshold maps probability to `PASS` / `REGENERATE`.

## Orchestration

`agent_backend/agent.py` implements `VerificationAgent`: feature extraction and prediction.  
`run_agent.sh` / `agent_backend/run.py` expose training and evaluation modes driven by YAMLs in `configs/`.

## Research track (Approach B)

`agent_backend/graphs/` contains claim → scene-graph SFT utilities, schema, and grounding hooks (LLMDet). This path is experimental and is not the default decision path in `VerificationAgent.predict`.
