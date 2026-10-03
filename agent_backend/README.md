# `agent_backend`

Core implementation of **Verification Before Assembly: A Low-Cost Verifier for Generated Scientific Video Frames**.

Given a figure and its caption / text context, the agent decides:

- **PASS** — keep the figure  
- **REGENERATE** — regenerate the figure

Pipeline: **Router → Judges A/B/C → 37 features → Arbiter**.

```text
image + text
    │
    ├─ Judge A (semantic)   SigLIP + MatchHead     → score_sem
    ├─ Judge B (structural) OCR / caption overlap  → score_struct
    ├─ Judge C (object)     zero-shot SigLIP nouns → score_obj
    └─ Router               ConvNeXt-Tiny          → p_byt, p_video, p_graph, p_scheme
            │
            ▼
      37-dim feature vector
            │
            ▼
      Arbiter → P(keep) ≷ threshold → PASS | REGENERATE
```

| Component | Module | Role |
|-----------|--------|------|
| Judges A/B/C | [`judges/`](judges/README.md) | `score_sem` / `score_struct` / `score_obj` |
| Router | `router.py` | Image-type probabilities |
| Features | `features.py` | 37-dimensional feature vector |
| Arbiter | `arbiter.py` | Meta-classifier + decision threshold |
| Agent | `agent.py` | Orchestration (`compute_features` / `predict`) |
| Graphs (research) | `graphs/` | Claim → scene-graph utilities |

## Entrypoints

```bash
# from repository root
./run_agent.sh --mode train_router
./run_agent.sh --mode train_sem
./run_agent.sh --mode train_arbiter
./run_agent.sh --mode eval --exp-name agent_v5
```

Configs live in [`../configs/`](../configs/). See also [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).
