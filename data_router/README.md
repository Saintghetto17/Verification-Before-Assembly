# Router dataset

Image-type labels for the ConvNeXt-Tiny domain classifier.

## Download (Hugging Face)

Shipped as the `router/` folder inside:

**[Saintghetto17/verification-before-assembly](https://huggingface.co/datasets/Saintghetto17/verification-before-assembly)**

```bash
python scripts/download_data.py
# → populates data_router/ from the HF `router/` tree
```

## Classes

| Class | Content |
|-------|---------|
| `byt` | biotechnology / photo-like scientific imagery |
| `video` | video-like / rendered frames |
| `graph` | plots and charts |
| `scheme` | diagrams / architectures / schemes |

There are **no** match/mismatch labels here — only image domain for soft routing.

Typical documented split sizes:

| split | byt | video | graph | scheme | Σ |
|-------|----:|------:|------:|-------:|--:|
| train | 250 | 800 | 250 | 250 | 1550 |
| val | 50 | 160 | 50 | 50 | 310 |

After download, train the router with:

```bash
./run_agent.sh --mode train_router --config-name agent_v5 --exp-name agent_v5
```
