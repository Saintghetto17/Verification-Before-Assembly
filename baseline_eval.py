#!/usr/bin/env python3
"""Standalone, leakage-safe baselines for figure/caption verification.

Golden labels are evaluation-only.  The only fitted parameter in this file is
the VQAScore decision threshold, and it is selected exclusively from a
non-golden validation set.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_GOLDEN = ROOT / "data/figures/golden/golden.jsonl"
QWEN_MODELS = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
)
VERDICTS = {"pass", "regenerate"}
JUDGE_PROMPT = """You are a strict scientific figure verification judge.
Compare the supplied image with its caption and optional surrounding context.
Check visible objects, text, labels, quantities, trends, relations, and whether
the image supports the caption. Do not assume facts that are not visible.

Return exactly one JSON object, with no markdown and no extra keys:
{{
  "correct_things": ["specific visible agreements"],
  "mistake_things": ["specific mismatches or unsupported claims"],
  "critical_things": ["mismatches severe enough to reject the figure"],
  "verdict": "pass" or "regenerate"
}}

Use "pass" only when there is no critical image-caption mismatch. Use empty
arrays when a category has no items.

CAPTION:
{caption}

SURROUNDING CONTEXT (secondary evidence only):
{context}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def iter_valid_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read complete records and tolerate one interrupted trailing write."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                return
            raise ValueError(f"Malformed non-trailing JSONL record in {path}:{index + 1}")
        if isinstance(value, dict):
            yield value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sample_id(row: dict[str, Any], index: int = 0) -> str:
    return str(row.get("sample_id") or row.get("id") or f"row-{index}")


def caption_of(row: dict[str, Any]) -> str:
    return str(row.get("caption") or row.get("premise") or row.get("text_block") or "").strip()


def image_of(row: dict[str, Any]) -> str:
    value = row.get("image_path") or row.get("image")
    if not value:
        raise ValueError(f"Missing image_path for {row.get('sample_id', '<unknown>')}")
    return str(Path(str(value)).expanduser().resolve())


def target_of(row: dict[str, Any]) -> int:
    """Return 1 for mismatch/regenerate and 0 for match/pass."""
    if "gold_is_corrupted" in row:
        return int(bool(row["gold_is_corrupted"]))
    gold = row.get("gold")
    if isinstance(gold, dict) and "is_corrupted" in gold:
        return int(bool(gold["is_corrupted"]))
    label = str(row.get("label", "")).lower()
    if label in {"mismatch", "regenerate", "corrupted", "fail", "1"}:
        return 1
    if label in {"match", "pass", "clean", "0"}:
        return 0
    if "label_id" in row:
        return int(row["label_id"])
    raise ValueError(f"Missing binary label for {row.get('sample_id', '<unknown>')}")


def discover_non_golden(figures_dir: Path) -> list[dict[str, Any]]:
    """Load labeled sidecars below data/figures, explicitly excluding golden."""
    rows: list[dict[str, Any]] = []
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for split_dir in sorted(figures_dir.iterdir()):
        if not split_dir.is_dir() or split_dir.name == "golden" or split_dir.name.startswith("_"):
            continue
        folder_corrupted = "l2_fail" in split_dir.name.lower()
        for image in sorted(split_dir.rglob("*")):
            if not image.is_file() or image.suffix.lower() not in suffixes:
                continue
            sidecar = image.with_suffix(".json")
            if not sidecar.is_file():
                continue
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            gold = data.get("gold") if isinstance(data.get("gold"), dict) else {}
            corrupted = bool(gold.get("is_corrupted", folder_corrupted))
            caption = str(data.get("caption") or data.get("text_block") or data.get("premise") or "").strip()
            if not caption:
                continue
            rows.append(
                {
                    "sample_id": data.get("sample_id")
                    or f"{split_dir.name}/{image.parent.name}/{image.stem}",
                    "split": split_dir.name,
                    "image_path": str(image.resolve()),
                    "caption": caption,
                    "text_block": str(data.get("text_block") or data.get("premise") or ""),
                    "gold_is_corrupted": corrupted,
                    "label": "mismatch" if corrupted else "match",
                }
            )
    if not rows:
        raise RuntimeError(f"No labeled non-golden validation rows found under {figures_dir}")
    return rows


def load_validation(path: Path | None, figures_dir: Path) -> tuple[list[dict[str, Any]], str]:
    if path is not None:
        resolved = path.resolve()
        if DEFAULT_GOLDEN.resolve() == resolved or "golden" in resolved.parts:
            raise ValueError("Validation input must be non-golden; refusing a path containing 'golden'")
        return read_jsonl(resolved), str(resolved)
    return discover_non_golden(figures_dir), f"{figures_dir.resolve()} excluding golden/"


def binary_metrics(targets: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    """Metrics compatible with the agent: match/pass is the positive class.

    Inputs remain encoded as 1=mismatch and 0=match because that is the native
    golden label convention used by this standalone runner.
    """
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions have different lengths")
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    tn = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    n = len(targets)
    mismatch_precision = tp / (tp + fp) if tp + fp else 0.0
    mismatch_recall = tp / (tp + fn) if tp + fn else 0.0
    mismatch_f1 = (
        2 * mismatch_precision * mismatch_recall / (mismatch_precision + mismatch_recall)
        if mismatch_precision + mismatch_recall
        else 0.0
    )
    match_tp, match_tn, match_fp, match_fn = tn, tp, fn, fp
    precision = match_tp / (match_tp + match_fp) if match_tp + match_fp else 0.0
    recall = match_tp / (match_tp + match_fn) if match_tp + match_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "positive_class": "match/pass",
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": match_tp,
        "tn": match_tn,
        "fp": match_fp,
        "fn": match_fn,
        "macro_f1": (f1 + mismatch_f1) / 2,
        "min_class_f1": min(f1, mismatch_f1),
        "f1_match": f1,
        "f1_mismatch": mismatch_f1,
        "by_class": {
            "match/pass": {"precision": precision, "recall": recall, "f1": f1},
            "mismatch/regenerate": {
                "precision": mismatch_precision,
                "recall": mismatch_recall,
                "f1": mismatch_f1,
            },
        },
    }


def select_threshold(
    match_scores: Sequence[float], targets: Sequence[int], objective: str = "f1"
) -> tuple[float, dict[str, Any]]:
    """Select pass iff score >= threshold; mismatch is the positive class."""
    if not match_scores or len(match_scores) != len(targets):
        raise ValueError("Need equally sized, non-empty validation scores and targets")
    unique = sorted(set(float(score) for score in match_scores))
    candidates = [unique[0] - 1e-12]
    candidates.extend((left + right) / 2 for left, right in zip(unique, unique[1:]))
    candidates.append(unique[-1] + 1e-12)
    best: tuple[tuple[float, float, float], float, dict[str, Any]] | None = None
    for threshold in candidates:
        predictions = [int(score < threshold) for score in match_scores]
        metrics = binary_metrics(targets, predictions)
        rank = (float(metrics[objective]), float(metrics["accuracy"]), -abs(threshold - 0.5))
        if best is None or rank > best[0]:
            best = (rank, threshold, metrics)
    assert best is not None
    return best[1], best[2]


def _balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


def parse_judge_json(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(item.strip() for item in fenced)
    balanced = _balanced_json_object(text)
    if balanced:
        candidates.append(balanced)
    parsed: Any = None
    errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except Exception as exc:
            errors.append(str(exc))
        try:
            parsed = ast.literal_eval(candidate)
            break
        except Exception as exc:
            errors.append(str(exc))
    if not isinstance(parsed, dict):
        raise ValueError("Could not parse a JSON object: " + "; ".join(errors[-2:]))

    aliases = {
        "correct_things": ("correct_things", "correct things", "correct_thinks"),
        "mistake_things": ("mistake_things", "mistake things", "mistake_thinks"),
        "critical_things": ("critical_things", "critical things", "critical_thinks"),
    }
    normalized: dict[str, Any] = {}
    lowered = {str(key).strip().lower(): value for key, value in parsed.items()}
    for output_key, keys in aliases.items():
        value = next((lowered[key] for key in keys if key in lowered), [])
        if value is None:
            value = []
        if isinstance(value, str):
            value = [value] if value.strip() else []
        if not isinstance(value, list):
            raise ValueError(f"{output_key} must be an array")
        normalized[output_key] = [str(item).strip() for item in value if str(item).strip()]
    verdict = str(lowered.get("verdict", "")).strip().lower().rstrip(".")
    if verdict not in VERDICTS:
        raise ValueError(f"Invalid verdict {verdict!r}")
    normalized["verdict"] = verdict
    return normalized


def provenance(
    command: str,
    *,
    sources: dict[str, Any],
    models: dict[str, Any],
    leakage_policy: str,
) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in ("transformers", "torch", "t2v-metrics"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "created_at_utc": utc_now(),
        "command": command,
        "argv": sys.argv,
        "sources": sources,
        "models": models,
        "packages": packages,
        "leakage_policy": leakage_policy,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _load_qwen(model_name: str, device_map: str, dtype: str) -> tuple[Any, Any]:
    try:
        import torch  # noqa: F401
        import transformers
    except ImportError as exc:
        raise RuntimeError("judge requires torch and transformers") from exc
    processor = transformers.AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    kwargs: dict[str, Any] = {"device_map": device_map, "trust_remote_code": True}
    kwargs["torch_dtype"] = "auto" if dtype == "auto" else getattr(torch, dtype)
    if "Qwen3-" in model_name:
        model_class = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    else:
        model_class = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
    model_class = model_class or getattr(transformers, "AutoModelForImageTextToText", None)
    if model_class is None:
        raise RuntimeError("Installed transformers does not support this Qwen VL model")
    model = model_class.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, processor


def _qwen_generate(
    model: Any, processor: Any, image_path: str, prompt: str, max_new_tokens: int
) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("judge requires Pillow") from exc
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    try:
        device = next(model.parameters()).device
        inputs = inputs.to(device)
    except (StopIteration, AttributeError):
        pass
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    prompt_length = inputs["input_ids"].shape[1]
    completion_ids = generated[:, prompt_length:]
    return processor.batch_decode(
        completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def _record_key(
    row: dict[str, Any], model_name: str, index: int, *, protocol: str
) -> str:
    return canonical_hash(
        {
            "sample_id": sample_id(row, index),
            "image": image_of(row),
            "caption": caption_of(row),
            "model": model_name,
            "protocol": protocol,
        }
    )


def _resume_keys(path: Path) -> set[str]:
    records = list(iter_valid_jsonl(path))
    # Remove an interrupted trailing fragment before opening the file in append mode.
    if path.exists():
        write_jsonl(path, records)
    return {
        str(row["record_key"])
        for row in records
        if isinstance(row.get("record_key"), str)
    }


def run_judge(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    completed = _resume_keys(args.output)
    model, processor = _load_qwen(args.model, args.device_map, args.dtype)
    protocol = canonical_hash({"kind": "structured_judge", "prompt": JUDGE_PROMPT})
    for index, row in enumerate(rows):
        key = _record_key(row, args.model, index, protocol=protocol)
        if key in completed:
            continue
        prompt = JUDGE_PROMPT.format(
            caption=caption_of(row),
            context=str(row.get("text_block") or row.get("premise") or "").strip(),
        )
        raw = _qwen_generate(model, processor, image_of(row), prompt, args.max_new_tokens)
        record: dict[str, Any] = {
            "record_key": key,
            "sample_id": sample_id(row, index),
            "model": args.model,
            "raw_output": raw,
            "target": target_of(row),
            "created_at_utc": utc_now(),
        }
        try:
            parsed = parse_judge_json(raw)
            record.update({"parsed": parsed, "verdict": parsed["verdict"], "parse_error": None})
        except ValueError as exc:
            record.update({"parsed": None, "verdict": None, "parse_error": str(exc)})
        append_jsonl(args.output, record)
        completed.add(key)

    predictions: list[int] = []
    targets: list[int] = []
    invalid = 0
    expected = [
        _record_key(row, args.model, index, protocol=protocol)
        for index, row in enumerate(rows)
    ]
    records = {str(record.get("record_key")): record for record in iter_valid_jsonl(args.output)}
    for key in expected:
        if key not in records:
            raise RuntimeError(f"Missing completed judge record {key} in {args.output}")
        record = records[key]
        verdict = record.get("verdict")
        if verdict not in VERDICTS:
            invalid += 1
            if args.invalid_verdict == "exclude":
                continue
            verdict = args.invalid_verdict
        predictions.append(int(verdict == "regenerate"))
        targets.append(int(record["target"]))
    report = {
        "baseline": "structured_qwen_vl_judge",
        "metrics": binary_metrics(targets, predictions),
        "json_validity": {
            "valid": len(rows) - invalid,
            "invalid": invalid,
            "rate": (len(rows) - invalid) / len(rows) if rows else 0.0,
            "invalid_policy": args.invalid_verdict,
        },
        "provenance": provenance(
            "judge",
            sources={"evaluation": str(args.input.resolve()), "evaluation_sha256": canonical_hash(rows)},
            models={"judge": args.model},
            leakage_policy="No training or threshold tuning; labels are read only after generation.",
        ),
    }
    write_json(args.report, report)
    print(json.dumps(report["metrics"], indent=2))


def _score_value(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list):
        if not value:
            raise ValueError("t2v_metrics returned an empty score")
        value = value[0]
    return float(value)


def score_vqa_rows(rows: Sequence[dict[str, Any]], output: Path, model_name: str) -> None:
    try:
        import t2v_metrics
    except ImportError as exc:
        raise RuntimeError(
            "VQAScore requires t2v_metrics. For the paper's exact CLIP-FlanT5 "
            "baseline install: pip install 't2v-metrics==1.2'"
        ) from exc
    scorer = t2v_metrics.VQAScore(model=model_name)
    completed = _resume_keys(output)
    protocol = "t2v_metrics_default_vqascore"
    for index, row in enumerate(rows):
        key = _record_key(row, model_name, index, protocol=protocol)
        if key in completed:
            continue
        score = _score_value(scorer(images=[image_of(row)], texts=[caption_of(row)]))
        append_jsonl(
            output,
            {
                "record_key": key,
                "sample_id": sample_id(row, index),
                "model": model_name,
                "score": score,
                "target": target_of(row),
                "created_at_utc": utc_now(),
            },
        )
        completed.add(key)


def _ordered_scored_records(
    rows: Sequence[dict[str, Any]], score_path: Path, model_name: str
) -> list[dict[str, Any]]:
    records = {str(item.get("record_key")): item for item in iter_valid_jsonl(score_path)}
    result = []
    protocol = "t2v_metrics_default_vqascore"
    for index, row in enumerate(rows):
        key = _record_key(row, model_name, index, protocol=protocol)
        if key not in records:
            raise RuntimeError(f"Missing score for {sample_id(row, index)} in {score_path}")
        result.append(records[key])
    return result


def run_vqascore(args: argparse.Namespace) -> None:
    validation, validation_source = load_validation(args.validation_input, args.figures_dir)
    golden = read_jsonl(args.golden_input)
    score_vqa_rows(validation, args.validation_scores, args.model)
    score_vqa_rows(golden, args.golden_scores, args.model)
    val_records = _ordered_scored_records(validation, args.validation_scores, args.model)
    golden_records = _ordered_scored_records(golden, args.golden_scores, args.model)
    val_scores = [float(row["score"]) for row in val_records]
    val_targets = [int(row["target"]) for row in val_records]
    threshold, val_metrics = select_threshold(val_scores, val_targets, args.objective)
    golden_scores = [float(row["score"]) for row in golden_records]
    golden_targets = [int(row["target"]) for row in golden_records]
    golden_predictions = [int(score < threshold) for score in golden_scores]
    report = {
        "baseline": "vqascore",
        "score_semantics": "higher means image-caption match; pass iff score >= threshold",
        "threshold": threshold,
        "threshold_objective": args.objective,
        "threshold_selected_on": "non-golden validation only",
        "validation_metrics_at_selected_threshold": val_metrics,
        "golden_metrics": binary_metrics(golden_targets, golden_predictions),
        "provenance": provenance(
            "vqascore",
            sources={
                "validation": validation_source,
                "validation_sha256": canonical_hash(validation),
                "golden": str(args.golden_input.resolve()),
                "golden_sha256": canonical_hash(golden),
            },
            models={"vqascore": args.model},
            leakage_policy=(
                "The continuous threshold is optimized only on labeled non-golden "
                "validation. Golden is scored once with that frozen threshold."
            ),
        ),
    }
    write_json(args.report, report)
    print(json.dumps(report["golden_metrics"], indent=2))


COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "computer mouse", "tv remote", "computer keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
}
COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"}
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4}
POSITIONS = ("left of", "right of", "above", "below")


def _object_phrase(text: str) -> str | None:
    value = re.sub(r"^(?:a|an)\s+", "", text.strip())
    if value in COCO_CLASSES:
        return value
    for item in sorted(COCO_CLASSES, key=len, reverse=True):
        if value in {item + "s", item + "es"}:
            return item
        if item.endswith("y") and value == item[:-1] + "ies":
            return item
    return None


def geneval_metadata(caption: str) -> tuple[dict[str, Any] | None, str]:
    """Conservatively recognize only official GenEval-style prompt templates."""
    prompt = caption.strip()
    core_match = re.fullmatch(r"a photo of (.+?)[.]?", prompt, flags=re.IGNORECASE)
    if not core_match:
        return None, "not_a_simple_photo_prompt"
    core = core_match.group(1).lower().strip()

    for relation in POSITIONS:
        marker = f" {relation} "
        if marker in core:
            subject_text, reference_text = core.split(marker, 1)
            subject = _object_phrase(subject_text)
            reference = _object_phrase(reference_text)
            if subject and reference and subject != reference:
                return {
                    "tag": "position",
                    "include": [
                        {"class": reference, "count": 1},
                        {"class": subject, "count": 1, "position": [relation, 0]},
                    ],
                    "prompt": prompt,
                }, "supported"
            return None, "unsupported_position_objects"

    count_match = re.fullmatch(r"(two|three|four)\s+(.+)", core)
    if count_match:
        obj = _object_phrase(count_match.group(2))
        if obj:
            count = NUMBER_WORDS[count_match.group(1)]
            return {
                "tag": "counting",
                "include": [{"class": obj, "count": count}],
                "exclude": [{"class": obj, "count": count + 1}],
                "prompt": prompt,
            }, "supported"
        return None, "unsupported_count_object"

    parts = re.split(r"\s+and\s+", core)
    if len(parts) == 2:
        objects = [_object_phrase(part) for part in parts]
        if all(objects) and objects[0] != objects[1]:
            return {
                "tag": "two_object",
                "include": [{"class": item, "count": 1} for item in objects],
                "prompt": prompt,
            }, "supported"
        return None, "unsupported_two_object_prompt"
    if len(parts) > 2:
        return None, "more_than_two_objects"

    color_match = re.fullmatch(
        rf"(?:a|an)\s+({'|'.join(sorted(COLORS))})\s+(.+)", core
    )
    if color_match:
        obj = _object_phrase(color_match.group(2))
        if obj:
            return {
                "tag": "colors",
                "include": [{"class": obj, "count": 1, "color": color_match.group(1)}],
                "prompt": prompt,
            }, "supported"
        return None, "unsupported_colored_object"

    obj = _object_phrase(core)
    if obj:
        return {
            "tag": "single_object",
            "include": [{"class": obj, "count": 1}],
            "prompt": prompt,
        }, "supported"
    return None, "not_in_geneval_coco_vocabulary"


def run_geneval(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    metadata_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    applicability_rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for index, row in enumerate(rows):
        metadata, reason = geneval_metadata(caption_of(row))
        supported = metadata is not None
        reasons[reason] = reasons.get(reason, 0) + 1
        applicability_rows.append(
            {"sample_id": sample_id(row, index), "supported": supported, "reason": reason}
        )
        if metadata is not None:
            metadata_index = len(metadata_rows)
            metadata_rows.append(metadata)
            mapping_rows.append(
                {
                    "metadata_index": metadata_index,
                    "sample_id": sample_id(row, index),
                    "image_path": image_of(row),
                }
            )
    write_jsonl(args.metadata_output, metadata_rows)
    write_jsonl(args.mapping_output, mapping_rows)
    write_jsonl(args.applicability_output, applicability_rows)
    supported = len(metadata_rows)
    report = {
        "baseline": "geneval_applicability_preparation",
        "n_total": len(rows),
        "n_supported": supported,
        "n_unsupported": len(rows) - supported,
        "supported_caption_coverage": supported / len(rows) if rows else 0.0,
        "reasons": reasons,
        "score": None,
        "score_status": (
            "not_computed; this mode prepares official-compatible metadata only. "
            "Unsupported scientific captions are never assigned a fabricated GenEval score."
        ),
        "outputs": {
            "official_metadata": str(args.metadata_output),
            "image_mapping": str(args.mapping_output),
            "applicability": str(args.applicability_output),
        },
        "provenance": provenance(
            "geneval-prepare",
            sources={"input": str(args.input.resolve()), "input_sha256": canonical_hash(rows)},
            models={"geneval": "metadata preparation only; official evaluator not run"},
            leakage_policy="No fitting, threshold selection, or score computation.",
        ),
    }
    write_json(args.report, report)
    print(json.dumps({key: report[key] for key in ("n_total", "n_supported", "supported_caption_coverage")}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    judge = subparsers.add_parser("judge", help="Evaluate a structured Qwen VL judge")
    judge.add_argument("--input", type=Path, default=DEFAULT_GOLDEN)
    judge.add_argument("--model", choices=QWEN_MODELS, required=True)
    judge.add_argument("--output", type=Path, required=True, help="Resume-safe prediction JSONL")
    judge.add_argument("--report", type=Path, required=True)
    judge.add_argument("--device-map", default="auto")
    judge.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    judge.add_argument("--max-new-tokens", type=int, default=512)
    judge.add_argument("--invalid-verdict", choices=("pass", "regenerate", "exclude"), default="regenerate")
    judge.set_defaults(func=run_judge)

    vqa = subparsers.add_parser("vqascore", help="Tune VQAScore threshold off-golden, evaluate golden")
    vqa.add_argument("--golden-input", type=Path, default=DEFAULT_GOLDEN)
    vqa.add_argument("--validation-input", type=Path)
    vqa.add_argument("--figures-dir", type=Path, default=ROOT / "data/figures")
    vqa.add_argument("--model", default="clip-flant5-xxl")
    vqa.add_argument("--validation-scores", type=Path, required=True)
    vqa.add_argument("--golden-scores", type=Path, required=True)
    vqa.add_argument("--report", type=Path, required=True)
    vqa.add_argument("--objective", choices=("f1", "accuracy", "precision", "recall"), default="f1")
    vqa.set_defaults(func=run_vqascore)

    geneval = subparsers.add_parser(
        "geneval-prepare", help="Measure applicability and prepare supported GenEval metadata"
    )
    geneval.add_argument("--input", type=Path, default=DEFAULT_GOLDEN)
    geneval.add_argument("--metadata-output", type=Path, required=True)
    geneval.add_argument("--mapping-output", type=Path, required=True)
    geneval.add_argument("--applicability-output", type=Path, required=True)
    geneval.add_argument("--report", type=Path, required=True)
    geneval.set_defaults(func=run_geneval)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
