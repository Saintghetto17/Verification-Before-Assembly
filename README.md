<p align="center">
  <h1 align="center">Verification Before Assembly</h1>
  <p align="center">
    <b>A Low-Cost Verifier for Generated Scientific Video Frames</b>
  </p>
  <p align="center">
    <a href="https://openreview.net/forum?id=cNyEkRsJgC"><img alt="OpenReview" src="https://img.shields.io/badge/OpenReview-cNyEkRsJgC-b31b1b?style=for-the-badge&logo=openaccess&logoColor=white"></a>
    <a href="https://huggingface.co/datasets/Saintghetto17/verification-before-assembly"><img alt="Hugging Face Dataset" src="https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"></a>
    <a href="https://ai4metascience.org/"><img alt="NeurIPS 2026 Workshop" src="https://img.shields.io/badge/NeurIPS%202026-AI%20for%20Meta--Science-3b82f6?style=for-the-badge"></a>
    <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge"></a>
  </p>
  <p align="center">
    Official implementation of our paper accepted at the<br/>
    <b>NeurIPS 2026 Workshop on AI for Meta‑Science:<br/>Scaling and Organizing Science in the Age of AI</b>
  </p>
  <p align="center">
    <a href="https://openreview.net/forum?id=cNyEkRsJgC"><b>📄 Paper (OpenReview)</b></a>
    ·
    <a href="https://huggingface.co/datasets/Saintghetto17/verification-before-assembly"><b>🤗 Dataset</b></a>
    ·
    <a href="#quickstart"><b>🚀 Quickstart</b></a>
    ·
    <a href="#method"><b>🧠 Method</b></a>
    ·
    <a href="#citation"><b>📚 Citation</b></a>
  </p>
</p>

---

## Overview

Modern AI systems can generate scientific figures at scale. Quality control has not kept up: a locally edited diagram, a flipped axis, or a swapped label can look plausible while contradicting its caption.

**Verification Before Assembly: A Low-Cost Verifier for Generated Scientific Video Frames** asks a practical meta-science question:

> Should this figure be **kept** or **regenerated** before it is assembled into a paper, video, or review artifact?

This repository releases the embedding-based verifier used in the paper (Router → Judges → Arbiter), together with evaluation tooling, metrics, and a compact public sample set.

| Decision | Meaning |
|----------|---------|
| **PASS** | Caption and image are consistent enough to keep |
| **REGENERATE** | Likely mismatch — regenerate the figure |

We prioritize catching faulty figures: false accepts are more costly than extra regenerations.

---

## Paper

| Field | Value |
|-------|-------|
| **Title** | Verification Before Assembly: A Low-Cost Verifier for Generated Scientific Video Frames |
| **Venue** | NeurIPS 2026 Workshop on AI for Meta‑Science |
| **OpenReview** | [https://openreview.net/forum?id=cNyEkRsJgC](https://openreview.net/forum?id=cNyEkRsJgC) |
| **Dataset** | [Saintghetto17/verification-before-assembly](https://huggingface.co/datasets/Saintghetto17/verification-before-assembly) |
| **Workshop site** | [ai4metascience.org](https://ai4metascience.org/) |
| **Code** | this repository |

> If you use this code or ideas, please cite the paper (see [Citation](#citation)).

---

## Method

We study two complementary approaches. **Approach A** is the released production-style agent. **Approach B** is an ongoing visual-spec / scene-graph research track included under `agent_backend/graphs/`.

### Approach A — Embedding-based verification agent (released)

```text
image + caption / text block
        │
        ├─ Router  (ConvNeXt-Tiny)     → p_byt, p_video, p_graph, p_scheme
        ├─ Judge A (SigLIP + MatchHead)→ score_sem
        ├─ Judge B (OCR / structural)  → score_struct
        └─ Judge C (zero-shot objects) → score_obj
                │
                ▼
        37-dimensional feature vector
                │
                ▼
        Arbiter (XGBoost / ensemble) → P(keep) ≷ threshold → PASS | REGENERATE
```

**Soft routing.** The router does **not** hard-gate judges. Domain probabilities reweight judge scores:

```text
sem_w    = score_sem    × (p_byt + p_video)
struct_w = score_struct × (p_graph + p_scheme)
obj_w    = score_obj    × (p_scheme + p_byt)
```

Motivation: SigLIP is stronger on photo/video-like frames; OCR/structure matters for graphs and schemes; object presence helps schemes and photographic content.

| Stage | Role | Trainable? |
|-------|------|------------|
| **Router** | ConvNeXt-Tiny domain classifier (`byt` / `video` / `graph` / `scheme`) | Yes |
| **Judge A** | SigLIP-base-patch16-224 + binary MatchHead → semantic match score | Yes |
| **Judge B** | EasyOCR + caption Jaccard / number recall | No (rule-based) |
| **Judge C** | Noun prompts via shared SigLIP (top-3 avg) | No separate head |
| **Arbiter** | Meta-classifier over 37 features | Yes |

### Approach B — Claim → scene graph → geometry (research)

Experimental pipeline that converts visual claims into scene graphs (entities + relations), grounds nodes with an open-vocabulary detector (LLMDet), and checks spatial / semantic conditions on crops. Code lives in `agent_backend/graphs/`; it is **not** yet wired into the default `VerificationAgent.predict` path.

---

## Repository layout

```text
Verification-Before-Assembly/
├── README.md                 # you are here
├── CITATION.cff              # machine-readable citation
├── LICENSE                   # Apache-2.0
├── requirements.txt
├── run_agent.sh              # main CLI entrypoint
├── agent_backend/            # Router, Judges A/B/C, features, arbiter, graphs
├── configs/                  # experiment YAMLs (agent_v5, pair128, …)
├── checkpoints/              # small arbiter artifacts (see Weights)
├── examples/check_generation # 5 orig + 5 fail paired samples (PNG + JSON)
├── tests/                    # unit tests
├── scripts/                  # data rebuild helpers
├── docs/                     # architecture & reproduction notes
├── TRAIN_METRICS.md          # tabulated experiment metrics
├── BASELINE_EVAL.md          # baseline protocol
└── ablations_notes.md        # ablation protocol & findings
```

---

## Quickstart

### 1. Environment

```bash
git clone https://github.com/Saintghetto17/Verification-Before-Assembly.git
cd Verification-Before-Assembly

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# Install a matching PyTorch build for your platform, then:
# python -m spacy download en_core_web_sm
```

### 2. Inspect the public sample pairs

```bash
ls examples/check_generation/01_l2_fail
ls examples/check_generation/02_orig_paired
```

Each sample folder contains a figure PNG and a JSON with caption, visual claims/description, and gold lie annotation for failures.

### 3. Run the agent (after placing weights)

Large neural weights (`router.pth`, `sem.pt`) are **not** shipped in git (size limits). Place them under `checkpoints/` (or point your YAML `ckpt` paths accordingly), then:

```bash
chmod +x run_agent.sh
./run_agent.sh --mode eval --exp-name agent_v5
```

Common modes:

```bash
./run_agent.sh --mode train_router
./run_agent.sh --mode train_sem
./run_agent.sh --mode train_arbiter
./run_agent.sh --mode all
```

Or launch the same modes through the local PyTorch helpers:

```bash
python submit_agent.py train_sem --config-name agent_v5 --exp-name agent_v5
python submit_agent.py train_sem --n-gpus 2          # multi-GPU via torchrun
python submit_baseline.py qwen25_7b --dry-run
```

See [`agent_backend/README.md`](agent_backend/README.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.

---

## Data

Public dataset on Hugging Face:

**[🤗 Saintghetto17/verification-before-assembly](https://huggingface.co/datasets/Saintghetto17/verification-before-assembly)**

| Split | Role | Size |
|-------|------|-----:|
| Non-golden train pool | `ml` / `juri` / `bt` figures for training & CV | 691 |
| Golden holdout | held-out evaluation (never used for training / threshold search) | 294 |
| Router dataset | image-type labels (`byt` / `video` / `graph` / `scheme`) | ~1860 |

Mismatches are typically **local edits** of real figures (labels, arrows, blocks, axes)—not random negatives.

```bash
pip install huggingface_hub
python scripts/download_data.py
# → data/figures/  and  data_router/
```

This repository also ships:

- docs under [`data/README.md`](data/README.md) and [`data_router/README.md`](data_router/README.md)
- a compact paired demo pack under [`examples/check_generation/`](examples/check_generation/)

---

## Evaluation protocol (summary)

Primary balanced metrics:

- class-wise Precision / Recall / F1 for **match** and **mismatch**
- **macro-F1** = mean of the two class F1s
- **min-class F1** = `min(F1_match, F1_mismatch)` (selection objective)

Non-golden arbiter selection uses **repeated 5×5 Stratified Group K-Fold** (paired orig/fail stay in the same fold). Out-of-fold probabilities are averaged over repeats; a single threshold is chosen to maximize min-class F1. Tabulated numbers live in [`TRAIN_METRICS.md`](TRAIN_METRICS.md).

---

## Weights

| Artifact | Included in git? | Notes |
|----------|------------------|-------|
| `checkpoints/arbiter.pkl` | ✅ | Arbiter model |
| `checkpoints/scaler.pkl` | ✅ | Feature scaler |
| `checkpoints/decision_threshold.json` | ✅ | Decision threshold |
| `checkpoints/router.pth` | ❌ (~107 MB) | ConvNeXt-Tiny router |
| `checkpoints/sem.pt` | ❌ (~786 MB) | SigLIP + MatchHead |

Provide large weights via your preferred release channel (Hugging Face, Zenodo, internal storage) and place them under `checkpoints/`.

---

## Citation

If you find this repository or paper useful, please cite:

```bibtex
@inproceedings{verification_before_assembly_2026,
  title     = {Verification Before Assembly: A Low-Cost Verifier for Generated Scientific Video Frames},
  author    = {Gromov, Egor},
  booktitle = {NeurIPS 2026 Workshop on AI for Meta-Science: Scaling and Organizing Science in the Age of AI},
  year      = {2026},
  url       = {https://openreview.net/forum?id=cNyEkRsJgC}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided.

> Please update the author list in your citation to match the final OpenReview / camera-ready author string.

---

## Acknowledgments

This work was developed in the context of AI-assisted scientific media generation and verification. Workshop context: [AI for Meta‑Science @ NeurIPS 2026](https://ai4metascience.org/).

---

## License

This project is released under the [Apache License 2.0](LICENSE).
