# Configs

Experiment YAML files for `run_agent.sh` / `submit_agent.py` (local PyTorch launchers).

| File | Notes |
|------|-------|
| `agent_v5.yaml` | Main embedding-agent configuration |
| `agent_v5_sem_pair128.yaml` | Pair-aware semantic judge variant |
| `agent_v5_balanced_final.yaml` | Balanced arbiter / final settings |
| `agent_v6.yaml` | Research / visual-spec oriented settings |
| older `agent_v1`–`v4` | Historical snapshots |

Typical path conventions:

| Location | Role |
|----------|------|
| `checkpoints/` | Load router / semantic / arbiter weights |
| `train_ckpt/<exp>/` | Per-experiment training outputs (promote into `checkpoints/`) |
| `outputs/<exp>/` | Logs, features, metrics, predictions |

Select an experiment with `--exp-name <name>`.
