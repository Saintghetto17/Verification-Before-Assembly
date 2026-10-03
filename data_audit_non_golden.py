#!/usr/bin/env python3
"""Deterministic quality audit of non-golden figure training data.

This script deliberately enumerates only immediate, non-hidden, non-``golden``
split directories. It never opens the held-out golden JSONL or its sidecars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FAIL_SUFFIX = "_l2_fail"
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return SPACE_RE.sub(" ", text).strip()


def tokens(text: str) -> frozenset[str]:
    return frozenset(TOKEN_RE.findall(normalize_text(text)))


def pair_group(sample_id: str) -> str:
    parts = sample_id.split("/")
    if not parts:
        return sample_id
    family = parts[0]
    if family.endswith(FAIL_SUFFIX):
        family = family[: -len(FAIL_SUFFIX)]
    elif family.endswith("_orig"):
        family = family[: -len("_orig")]
    return "/".join([family, *parts[1:]])


def domain_for(split: str) -> str:
    if split.endswith(FAIL_SUFFIX):
        return split[: -len(FAIL_SUFFIX)]
    if split.endswith("_orig"):
        return split[: -len("_orig")]
    return split


def discover_rows(figures_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read non-golden PNG/sidecar pairs without traversing golden."""
    rows: list[dict[str, Any]] = []
    split_names: list[str] = []
    for split_dir in sorted(figures_dir.iterdir()):
        if (
            not split_dir.is_dir()
            or split_dir.name.casefold() == "golden"
            or split_dir.name.startswith("_")
        ):
            continue
        split_names.append(split_dir.name)
        split_default = split_dir.name.endswith(FAIL_SUFFIX)
        for image_path in sorted(split_dir.rglob("figure_*")):
            if not image_path.is_file() or image_path.suffix.casefold() not in IMAGE_SUFFIXES:
                continue
            sidecar_path = image_path.with_suffix(".json")
            if not sidecar_path.is_file():
                continue
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if not isinstance(sidecar, dict):
                continue
            label_data = sidecar.get("gold")
            label_data = label_data if isinstance(label_data, dict) else {}
            is_corrupted = bool(label_data.get("is_corrupted", split_default))
            sample_id = str(
                sidecar.get("sample_id")
                or f"{split_dir.name}/{image_path.parent.name}/{image_path.stem}"
            )
            caption = str(sidecar.get("caption") or "")
            text_block = str(sidecar.get("text_block") or sidecar.get("premise") or "")
            combined = caption + "\n" + text_block
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split_dir.name,
                    "domain": domain_for(split_dir.name),
                    "label": "mismatch" if is_corrupted else "match",
                    "label_id": int(is_corrupted),
                    "split_default_label_id": int(split_default),
                    "image_path": str(image_path.resolve()),
                    "sidecar_path": str(sidecar_path.resolve()),
                    "image_sha256": sha256_file(image_path),
                    "sidecar_sha256": sha256_file(sidecar_path),
                    "caption": caption,
                    "text_block": text_block,
                    "caption_norm": normalize_text(caption),
                    "text_block_norm": normalize_text(text_block),
                    "combined_norm": normalize_text(combined),
                    "pair_group": pair_group(sample_id),
                }
            )
    return rows, split_names


def entropy(counts: Iterable[int]) -> float:
    values = [count for count in counts if count]
    total = sum(values)
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in values)


def confounding(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    table: dict[str, Counter[int]] = defaultdict(Counter)
    labels = Counter[int]()
    for row in rows:
        table[str(row[key])][row["label_id"]] += 1
        labels[row["label_id"]] += 1
    n = len(rows)
    conditional_entropy = sum(
        (sum(counts.values()) / n) * entropy(counts.values())
        for counts in table.values()
    )
    label_entropy = entropy(labels.values())
    mutual_information = label_entropy - conditional_entropy
    majority_correct = sum(max(counts.values()) for counts in table.values())
    return {
        "field": key,
        "counts": {
            name: {
                "match": counts[0],
                "mismatch": counts[1],
                "mismatch_rate": counts[1] / sum(counts.values()),
            }
            for name, counts in sorted(table.items())
        },
        "label_entropy_bits": label_entropy,
        "conditional_label_entropy_bits": conditional_entropy,
        "mutual_information_bits": mutual_information,
        "normalized_mutual_information": (
            mutual_information / label_entropy if label_entropy else 0.0
        ),
        "majority_lookup_training_accuracy": majority_correct / n,
        "warning": (
            "In-sample lookup accuracy quantifies confounding, not generalization."
        ),
    }


def duplicate_groups(
    rows: list[dict[str, Any]], key: str, *, opposite_only: bool = False
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    result = []
    for value, members in sorted(grouped.items()):
        labels = {row["label_id"] for row in members}
        if len(members) < 2 or (opposite_only and len(labels) < 2):
            continue
        result.append(
            {
                "value": value,
                "n": len(members),
                "labels": sorted("mismatch" if x else "match" for x in labels),
                "sample_ids": sorted(row["sample_id"] for row in members),
            }
        )
    return result


def exact_text_groups(
    rows: list[dict[str, Any]], field: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        normalized = row[field]
        if normalized:
            grouped[normalized].append(row)
    all_duplicates: list[dict[str, Any]] = []
    opposite: list[dict[str, Any]] = []
    for normalized, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        labels = {row["label_id"] for row in members}
        item = {
            "normalized_sha256": sha256_bytes(normalized.encode("utf-8")),
            "characters": len(normalized),
            "n": len(members),
            "sample_ids": sorted(row["sample_id"] for row in members),
            "labels": sorted("mismatch" if x else "match" for x in labels),
        }
        all_duplicates.append(item)
        if len(labels) > 1:
            opposite.append(item)
    return all_duplicates, opposite


def near_text_pairs(
    rows: list[dict[str, Any]], field: str, threshold: float
) -> list[dict[str, Any]]:
    """Find non-exact near duplicates using token and character similarity."""
    values = [row[field] for row in rows]
    token_sets = [tokens(value) for value in values]
    pairs: list[dict[str, Any]] = []
    for left in range(len(rows)):
        if not values[left]:
            continue
        for right in range(left + 1, len(rows)):
            if not values[right] or values[left] == values[right]:
                continue
            union = token_sets[left] | token_sets[right]
            jaccard = (
                len(token_sets[left] & token_sets[right]) / len(union) if union else 1.0
            )
            if jaccard < max(0.75, threshold - 0.15):
                continue
            ratio = SequenceMatcher(
                None, values[left], values[right], autojunk=False
            ).ratio()
            if ratio < threshold:
                continue
            pairs.append(
                {
                    "left": rows[left]["sample_id"],
                    "right": rows[right]["sample_id"],
                    "left_label": rows[left]["label"],
                    "right_label": rows[right]["label"],
                    "opposite_labels": rows[left]["label_id"] != rows[right]["label_id"],
                    "sequence_ratio": ratio,
                    "token_jaccard": jaccard,
                }
            )
    return sorted(
        pairs,
        key=lambda item: (-item["sequence_ratio"], item["left"], item["right"]),
    )


def multimodal_contradictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["image_sha256"], row["combined_norm"])].append(row)
    result = []
    for (image_hash, combined), members in sorted(grouped.items()):
        if len({row["label_id"] for row in members}) < 2:
            continue
        result.append(
            {
                "image_sha256": image_hash,
                "combined_text_sha256": sha256_bytes(combined.encode("utf-8")),
                "sample_ids": sorted(row["sample_id"] for row in members),
                "labels": sorted({row["label"] for row in members}),
            }
        )
    return result


def group_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["pair_group"]].append(row)
    sizes = Counter(len(members) for members in groups.values())
    compositions = Counter(
        (sum(row["label_id"] == 0 for row in members), sum(row["label_id"] == 1 for row in members))
        for members in groups.values()
    )
    supports = {
        label: sum(any(row["label_id"] == label for row in members) for members in groups.values())
        for label in (0, 1)
    }
    return {
        "n_groups": len(groups),
        "size_distribution": {str(k): v for k, v in sorted(sizes.items())},
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
        "mean_size": statistics.mean(map(len, groups.values())) if groups else 0.0,
        "label_compositions": {
            f"match_{clean}_mismatch_{bad}": count
            for (clean, bad), count in sorted(compositions.items())
        },
        "groups_supporting_each_label": {
            "match": supports[0],
            "mismatch": supports[1],
        },
        "necessary_max_stratified_folds": min(supports.values()),
        "groups": [
            {
                "group": group,
                "n": len(members),
                "labels": [row["label"] for row in sorted(members, key=lambda x: x["sample_id"])],
                "sample_ids": sorted(row["sample_id"] for row in members),
            }
            for group, members in sorted(groups.items())
        ],
    }


def cv_feasibility(rows: list[dict[str, Any]], maximum: int = 10) -> dict[str, Any]:
    """Construct deterministic, group-disjoint folds and verify both classes."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["pair_group"]].append(row)
    group_records = [
        {
            "group": group,
            "match": sum(row["label_id"] == 0 for row in members),
            "mismatch": sum(row["label_id"] == 1 for row in members),
            "n": len(members),
        }
        for group, members in grouped.items()
    ]
    totals = [
        sum(record["match"] for record in group_records),
        sum(record["mismatch"] for record in group_records),
    ]
    results: dict[str, Any] = {}
    for n_splits in range(2, maximum + 1):
        fold_counts = [[0, 0] for _ in range(n_splits)]
        fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
        # Harder mixed/larger groups are placed first. The SHA-256 tie-break
        # avoids dependence on source traversal order.
        ordered = sorted(
            group_records,
            key=lambda record: (
                -min(record["match"], record["mismatch"]),
                -record["n"],
                sha256_bytes(record["group"].encode("utf-8")),
            ),
        )
        for record in ordered:
            additions = [record["match"], record["mismatch"]]

            def placement_cost(fold: int) -> tuple[float, float, int]:
                projected = [counts.copy() for counts in fold_counts]
                projected[fold][0] += additions[0]
                projected[fold][1] += additions[1]
                class_imbalance = sum(
                    statistics.pvariance(
                        projected[index][label] / totals[label]
                        for index in range(n_splits)
                    )
                    for label in (0, 1)
                )
                row_imbalance = statistics.pvariance(
                    sum(projected[index]) / len(rows)
                    for index in range(n_splits)
                )
                return class_imbalance, row_imbalance, fold

            selected = min(range(n_splits), key=placement_cost)
            fold_counts[selected][0] += additions[0]
            fold_counts[selected][1] += additions[1]
            fold_groups[selected].append(record["group"])

        folds = []
        valid = True
        for validation_counts, validation_groups in zip(fold_counts, fold_groups):
            train_counts = [
                totals[label] - validation_counts[label] for label in (0, 1)
            ]
            fold_valid = min(train_counts) > 0 and min(validation_counts) > 0
            valid &= fold_valid
            folds.append(
                {
                    "n_train": sum(train_counts),
                    "n_validation": sum(validation_counts),
                    "train_match_mismatch": train_counts,
                    "validation_match_mismatch": validation_counts,
                    "group_overlap": 0,
                    "n_validation_groups": len(validation_groups),
                    "validation_groups_sha256": canonical_sha256(
                        sorted(validation_groups)
                    ),
                    "valid": fold_valid,
                }
            )
        results[str(n_splits)] = {
            "feasible_in_constructed_partition": valid,
            "folds": folds,
        }
    return {
        "available": True,
        "method": (
            "Dependency-free deterministic greedy assignment minimizing per-fold "
            "class-count squared error; group disjointness and both-class presence "
            "are verified after construction."
        ),
        "results": results,
    }


def build_report(
    rows: list[dict[str, Any]],
    figures_dir: Path,
    split_names: list[str],
    near_threshold: float,
) -> dict[str, Any]:
    labels = Counter(row["label"] for row in rows)
    image_duplicates = duplicate_groups(rows, "image_sha256")
    opposite_image = duplicate_groups(rows, "image_sha256", opposite_only=True)
    caption_exact, caption_opposite = exact_text_groups(rows, "caption_norm")
    block_exact, block_opposite = exact_text_groups(rows, "text_block_norm")
    combined_exact, combined_opposite = exact_text_groups(rows, "combined_norm")
    caption_near = near_text_pairs(rows, "caption_norm", near_threshold)
    block_near = near_text_pairs(rows, "text_block_norm", near_threshold)
    combined_near = near_text_pairs(rows, "combined_norm", near_threshold)
    contradictions = multimodal_contradictions(rows)
    id_duplicates = duplicate_groups(rows, "sample_id")
    path_duplicates = duplicate_groups(rows, "image_path")
    sidecar_path_duplicates = duplicate_groups(rows, "sidecar_path")
    split_disagreements = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "split_default": (
                "mismatch" if row["split_default_label_id"] else "match"
            ),
            "sidecar_label": row["label"],
        }
        for row in rows
        if row["label_id"] != row["split_default_label_id"]
    ]
    fingerprints = [
        {
            "sample_id": row["sample_id"],
            "label": row["label"],
            "image_sha256": row["image_sha256"],
            "sidecar_sha256": row["sidecar_sha256"],
            "caption_sha256": sha256_bytes(row["caption_norm"].encode("utf-8")),
            "text_block_sha256": sha256_bytes(row["text_block_norm"].encode("utf-8")),
            "combined_text_sha256": sha256_bytes(row["combined_norm"].encode("utf-8")),
        }
        for row in sorted(rows, key=lambda item: item["sample_id"])
    ]
    return {
        "audit_version": 1,
        "scope": {
            "figures_dir": str(figures_dir.resolve()),
            "policy": (
                "Only immediate non-hidden split directories were traversed; "
                "the golden directory and hidden/backup directories were excluded "
                "without opening their files."
            ),
            "included_splits": split_names,
            "n_rows": len(rows),
        },
        "class_balance": {
            "match": labels["match"],
            "mismatch": labels["mismatch"],
            "mismatch_rate": labels["mismatch"] / len(rows),
        },
        "integrity": {
            "duplicate_sample_id_groups": id_duplicates,
            "duplicate_image_path_groups": path_duplicates,
            "duplicate_sidecar_path_groups": sidecar_path_duplicates,
            "split_default_vs_sidecar_label_disagreements": split_disagreements,
        },
        "pair_groups": group_audit(rows),
        "image_bytes": {
            "duplicate_hash_groups": image_duplicates,
            "opposite_label_duplicate_hash_groups": opposite_image,
            "n_duplicate_hash_groups": len(image_duplicates),
            "n_opposite_label_duplicate_hash_groups": len(opposite_image),
        },
        "text_duplicates": {
            "near_threshold": near_threshold,
            "caption": {
                "exact_groups": caption_exact,
                "opposite_label_exact_groups": caption_opposite,
                "near_pairs_excluding_exact": caption_near,
                "opposite_label_near_pairs_excluding_exact": [
                    pair for pair in caption_near if pair["opposite_labels"]
                ],
            },
            "text_block": {
                "exact_groups": block_exact,
                "opposite_label_exact_groups": block_opposite,
                "near_pairs_excluding_exact": block_near,
                "opposite_label_near_pairs_excluding_exact": [
                    pair for pair in block_near if pair["opposite_labels"]
                ],
            },
            "caption_plus_text_block": {
                "exact_groups": combined_exact,
                "opposite_label_exact_groups": combined_opposite,
                "near_pairs_excluding_exact": combined_near,
                "opposite_label_near_pairs_excluding_exact": [
                    pair for pair in combined_near if pair["opposite_labels"]
                ],
            },
        },
        "label_contradictions": {
            "definition": (
                "Opposite labels assigned to exactly identical observable "
                "image bytes and normalized caption+text input."
            ),
            "irreducible_image_plus_text_groups": contradictions,
            "n_irreducible_groups": len(contradictions),
            "irreducible_rows": sum(len(item["sample_ids"]) for item in contradictions),
        },
        "domain_label_confounding": {
            "split": confounding(rows, "split"),
            "normalized_domain": confounding(rows, "domain"),
        },
        "cross_validation": cv_feasibility(rows),
        "fingerprints": {
            "algorithm": "SHA-256",
            "canonical_dataset_sha256": canonical_sha256(fingerprints),
            "rows": fingerprints,
        },
        "limitations": [
            "Near-duplicate text is defined by normalized character SequenceMatcher "
            "ratio plus token-set Jaccard candidate filtering; it is not semantic entailment.",
            "Exact image duplication compares file bytes, not perceptual similarity.",
            "Confounding lookup accuracy is descriptive and in-sample.",
        ],
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures-dir", type=Path, default=root / "data" / "figures"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / "agent_v5_balanced_dev" / "data_audit.json",
    )
    parser.add_argument("--near-threshold", type=float, default=0.95)
    parser.add_argument("--expect-rows", type=int, default=691)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.near_threshold <= 1.0:
        raise ValueError("--near-threshold must be in (0, 1]")
    rows, split_names = discover_rows(args.figures_dir)
    if len(rows) != args.expect_rows:
        raise RuntimeError(
            f"Expected {args.expect_rows} non-golden rows, discovered {len(rows)}"
        )
    report = build_report(rows, args.figures_dir, split_names, args.near_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "n_rows": len(rows),
                "dataset_sha256": report["fingerprints"][
                    "canonical_dataset_sha256"
                ],
                "irreducible_groups": report["label_contradictions"][
                    "n_irreducible_groups"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
