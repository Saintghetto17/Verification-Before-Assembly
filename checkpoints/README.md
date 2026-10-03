# Checkpoints

Shipped with this repository (small, inference-side arbiter artifacts):

| File | Description |
|------|-------------|
| `arbiter.pkl` | Trained arbiter |
| `scaler.pkl` | Feature `StandardScaler` |
| `decision_threshold.json` | Probability threshold for PASS / REGENERATE |

**Not shipped** (place manually under this directory):

| File | Approx. size | Description |
|------|-------------:|-------------|
| `router.pth` | ~107 MB | ConvNeXt-Tiny domain router |
| `sem.pt` | ~786 MB | Fine-tuned SigLIP + MatchHead |

Configure paths via the YAML files in `configs/` if your layout differs.
