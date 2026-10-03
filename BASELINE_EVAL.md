# Standalone baseline evaluation

`baseline_eval.py` evaluates baselines without training or tuning on
`data/figures/golden/golden.jsonl`. Heavy dependencies are imported only by the
subcommand that needs them. None of these commands submits a job.

All headline classification reports use **match / pass** as the positive class,
matching the existing agent metrics. They include accuracy, precision, recall,
F1, macro-F1, per-class metrics, and confusion counts. Reports also record
input hashes, package versions, model names, command arguments, runner hash,
and the leakage policy.

## Structured Qwen VL judge

The judge prompt requires `correct_things`, `mistake_things`,
`critical_things`, and a `pass`/`regenerate` verdict. Output JSONL is appended
and fsynced one record at a time; rerunning the same command resumes by a hash
of sample, image, caption, and model. A partial final JSONL record left by an
interruption is repaired on resume.

```bash
python baseline_eval.py judge \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --output outputs/baselines/qwen25_3b_golden.jsonl \
  --report outputs/baselines/qwen25_3b_report.json
```

The other supported judge model names are
`Qwen/Qwen3-VL-4B-Instruct` and `Qwen/Qwen2.5-VL-7B-Instruct`.
Use a sufficiently recent `transformers` release for the selected architecture.
Malformed model output is retained with its parse error. By default an invalid
verdict is conservatively evaluated as `regenerate`; change this explicitly
with `--invalid-verdict pass` or `--invalid-verdict exclude`.

## VQAScore

```bash
python baseline_eval.py vqascore \
  --model clip-flant5-xxl \
  --validation-scores outputs/baselines/vqa_validation.jsonl \
  --golden-scores outputs/baselines/vqa_golden.jsonl \
  --report outputs/baselines/vqa_report.json
```

By default, validation is the labeled `data/figures/**` pool excluding the
`golden/` directory. An explicit non-golden JSONL can be supplied with
`--validation-input`. The score is interpreted as image-caption match
confidence, so `score >= threshold` means pass. The continuous threshold is
selected on non-golden validation only (F1 by default), frozen, and then
applied to golden. Golden labels never participate in threshold selection.

The model name is passed unchanged to `t2v_metrics.VQAScore`. To reproduce the
paper-era CLIP-FlanT5 setup, install the August 2024 release. It predates the
unrelated DeepSpeed/Apex dependencies added later:

```bash
pip install 't2v-metrics==1.2'
```

Current `t2v_metrics` releases have a different model roster; use their model
names via `--model` when exact legacy reproduction is not required.

## GenEval applicability and preparation

```bash
python baseline_eval.py geneval-prepare \
  --metadata-output outputs/baselines/geneval_metadata.jsonl \
  --mapping-output outputs/baselines/geneval_image_mapping.jsonl \
  --applicability-output outputs/baselines/geneval_applicability.jsonl \
  --report outputs/baselines/geneval_report.json
```

This mode does **not** claim to run the official detector or produce a GenEval
score. It conservatively recognizes only simple official-style prompts over
the GenEval/COCO vocabulary: one object, two objects, counts two through four,
one colored object, or one of the supported spatial relations. The metadata
JSONL contains only official-compatible `tag`, `include`, optional `exclude`,
and `prompt` fields. Image paths and sample IDs are kept in a separate mapping.

Scientific captions that cannot be represented honestly are marked unsupported
in the applicability JSONL. The report gives supported-caption coverage and
sets `score` to `null`; it never fabricates a score for unsupported captions.

## Smoke tests

The parsing, metric, threshold-direction, and GenEval applicability tests have
no ML dependencies:

```bash
python -m unittest -v test_baseline_eval.py
```
