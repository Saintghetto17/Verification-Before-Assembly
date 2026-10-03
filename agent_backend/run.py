"""CLI modes for training, evaluation, and leakage-safe ablation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from .ablation import run_ablations
from .agent import VerificationAgent
from .arbiter import (
    maybe_shap_summary,
    save_arbiter,
    summarize_predictions,
    train_arbiter,
)
from .ckpt_layout import promote_to_checkpoints
from .config import AgentConfig, pick_device, resolve
from .dataset import load_golden, load_train_figures
from .features import FEATURE_NAMES
from .router import generate_router_data, train_router
from .text_utils import row_texts
from .train_sem import train_semantic_judge


def _setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)


def _train_dir(cfg: AgentConfig, project_root: Path) -> Path:
    """Per-exp weight dir: train_ckpt/<exp>/."""
    d = resolve(project_root, cfg.paths.train_ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _output_dir(cfg: AgentConfig, project_root: Path) -> Path:
    """Per-exp reports dir: outputs/<exp>/ (logs, CSV, metrics)."""
    d = resolve(project_root, cfg.paths.output_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_train_rows(cfg: AgentConfig, project_root: Path) -> List[Dict[str, Any]]:
    """Train pool: data/figures/** excluding golden."""
    figures_dir = resolve(project_root, cfg.paths.figures_dir)
    rows = load_train_figures(
        figures_dir,
        limit=cfg.limit,
        max_clean_per_corrupted=cfg.arbiter.max_clean_per_corrupted,
        seed=cfg.arbiter.seed,
    )
    if not rows:
        raise RuntimeError(f"No train figure rows under {figures_dir} (excl. golden)")
    n_match = sum(1 for r in rows if r.get("label") == "match")
    logging.info(
        "train figures: n=%d match=%d mismatch=%d (excl. golden) dir=%s",
        len(rows),
        n_match,
        len(rows) - n_match,
        figures_dir,
    )
    return rows


def _load_eval_rows(cfg: AgentConfig, project_root: Path) -> List[Dict[str, Any]]:
    """Held-out eval: golden only."""
    figures_dir = resolve(project_root, cfg.paths.figures_dir)
    rows = load_golden(figures_dir, limit=cfg.limit)
    if not rows:
        raise RuntimeError(f"No golden rows under {figures_dir}")
    logging.info("eval golden: n=%d", len(rows))
    return rows


def _should_log_progress(
    done: int,
    total: int,
    *,
    every_n: int,
    every_pct: float,
    last_pct_bucket: int,
) -> Tuple[bool, int]:
    """Return (should_log, new_pct_bucket)."""
    if done >= total:
        return True, 100
    pct = 100.0 * done / max(total, 1)
    bucket = int(pct // max(every_pct, 1e-6))
    by_n = every_n > 0 and done % every_n == 0
    by_pct = bucket > last_pct_bucket
    return by_n or by_pct, bucket if by_pct else last_pct_bucket


def build_feature_matrix(
    cfg: AgentConfig,
    project_root: Path,
    *,
    write_csv: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Runtime feature generation over data/figures (excl. golden).

    Returns X [n, 37], y [n], row metas, info dict.
    Optionally writes features.csv into outputs/<exp>/ for reporting.
    """
    rows = _load_train_rows(cfg, project_root)
    total = len(rows)
    agent = VerificationAgent(cfg, project_root, load_arbiter=False)

    X = np.zeros((total, len(FEATURE_NAMES)), dtype=np.float64)
    y = np.zeros(total, dtype=np.int64)
    metas: List[Dict[str, Any]] = []

    csv_path = resolve(project_root, cfg.paths.features_csv)
    csv_file = None
    writer = None
    if write_csv:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered so NFS/shared FS shows rows while the job runs
        csv_file = csv_path.open("w", encoding="utf-8", newline="", buffering=1)
        fieldnames = ["sample_id", "split", "label", "label_id", *FEATURE_NAMES]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

    every_n = max(int(cfg.arbiter.feature_log_every_n), 1)
    every_pct = float(cfg.arbiter.feature_log_every_pct)
    last_bucket = -1
    t0 = time.time()
    logging.info(
        "feature generation START: %d samples from data/figures (excl. golden) "
        "(feature_batch_size=%d, siglip_batch=%d, ocr_batch=%d, ocr_gpu=%s)",
        total,
        max(int(cfg.arbiter.feature_batch_size), 1),
        cfg.siglip.batch_size,
        cfg.arbiter.ocr_batch_size,
        getattr(agent.judge_struct, "use_gpu", None),
    )

    feat_bs = max(int(cfg.arbiter.feature_batch_size), 1)
    try:
        for start in range(0, total, feat_bs):
            chunk = rows[start : start + feat_bs]
            paths = [r["image_path"] for r in chunk]
            premises: List[str] = []
            captions: List[str] = []
            for r in chunk:
                premise, caption = row_texts(r)
                premises.append(premise)
                captions.append(caption)
            feat_list = agent.compute_features_batch(
                paths, premises, captions=captions
            )
            for j, (r, feats) in enumerate(zip(chunk, feat_list)):
                i = start + j
                label = r.get("label")
                label_id = 1 if label == "match" else 0
                X[i] = [float(feats[n]) for n in FEATURE_NAMES]
                y[i] = label_id
                meta = {
                    "sample_id": r.get("sample_id"),
                    "split": r.get("split"),
                    "label": label,
                    "label_id": label_id,
                }
                metas.append(meta)
                if writer is not None:
                    writer.writerow({**meta, **feats})

                done = i + 1
                should, last_bucket = _should_log_progress(
                    done,
                    total,
                    every_n=every_n,
                    every_pct=every_pct,
                    last_pct_bucket=last_bucket,
                )
                if should:
                    if csv_file is not None:
                        csv_file.flush()
                    pct = 100.0 * done / total
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-6)
                    eta = (total - done) / max(rate, 1e-6)
                    logging.info(
                        "features progress: %d/%d (%.1f%%) | %.2f samples/s | ETA %.0fs",
                        done,
                        total,
                        pct,
                        rate,
                        eta,
                    )
    finally:
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()

    elapsed = time.time() - t0
    info = {
        "n_rows": total,
        "n_features": len(FEATURE_NAMES),
        "source": "data/figures excluding golden",
        "elapsed_sec": round(elapsed, 2),
        "samples_per_sec": round(total / max(elapsed, 1e-6), 3),
        "features_csv": str(csv_path) if write_csv else None,
        "router_uniform": getattr(agent.router, "_uniform", None),
        "sem_mode": getattr(agent.judge_sem, "_mode", None),
        "ocr_backend": getattr(agent.judge_struct, "ocr_backend", "none"),
        "feature_names": FEATURE_NAMES,
    }
    if info["ocr_backend"] == "none":
        logging.warning(
            "OCR backend=none — score_struct is neutral 0.5; "
            "install tesseract+pytesseract or easyocr for Judge B"
        )
    logging.info(
        "feature generation DONE: %d/%d (100%%) in %.1fs → matrix X%s y%s%s",
        total,
        total,
        elapsed,
        list(X.shape),
        list(y.shape),
        f" csv={csv_path}" if write_csv else "",
    )
    if write_csv:
        meta_path = csv_path.parent / "features_meta.json"
        meta_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return X, y, metas, info


def mode_train_router(cfg: AgentConfig, project_root: Path) -> Dict[str, Any]:
    train_dir = _train_dir(cfg, project_root)
    out_ckpt = train_dir / "router.pth"
    data_dir = resolve(project_root, cfg.paths.router_data_dir)
    figures_dir = resolve(project_root, cfg.paths.figures_dir)
    generate_router_data(
        data_dir,
        classes=cfg.router.classes,
        n_train=cfg.router.num_per_class_train,
        n_val=cfg.router.num_per_class_val,
        n_video_train=cfg.router.num_video_train,
        n_video_val=cfg.router.num_video_val,
        seed=cfg.router.seed,
        figures_dir=figures_dir,
    )
    device = pick_device(cfg.device)
    result = train_router(
        data_dir,
        out_ckpt,
        classes=cfg.router.classes,
        epochs=cfg.router.epochs,
        batch_size=cfg.router.batch_size,
        lr=cfg.router.learning_rate,
        image_size=cfg.router.image_size,
        device=device,
        project_root=project_root,
        tb_run_name=f"{cfg.name}_router",
    )
    promoted = promote_to_checkpoints(project_root, train_dir, names=["router.pth"])
    result["train_ckpt"] = str(out_ckpt)
    result["promoted"] = promoted
    return result


def mode_train_sem(cfg: AgentConfig, project_root: Path) -> Dict[str, Any]:
    """Fine-tune SigLIP + match head so mismatch < 0.5 < match."""
    train_dir = _train_dir(cfg, project_root)
    out_ckpt = train_dir / "sem.pt"
    rows = _load_train_rows(cfg, project_root)
    device = pick_device(cfg.device)
    st = cfg.sem_train
    result = train_semantic_judge(
        rows,
        pretrained=cfg.siglip.pretrained,
        out_ckpt=out_ckpt,
        project_root=project_root,
        epochs=st.epochs,
        batch_size=st.batch_size,
        lr_head=st.lr_head,
        lr_backbone=st.lr_backbone,
        weight_decay=st.weight_decay,
        max_text_length=cfg.siglip.max_text_length,
        test_ratio=st.test_ratio,
        seed=st.seed,
        device=device,
        freeze_backbone_epochs=st.freeze_backbone_epochs,
        tb_run_name=f"{cfg.name}_sem",
        loss_type=st.loss_type,
        positive_class_weight=st.positive_class_weight,
        focal_gamma=st.focal_gamma,
        pair_aware_batches=st.pair_aware_batches,
        pairwise_loss_weight=st.pairwise_loss_weight,
        pairwise_margin=st.pairwise_margin,
        refit_all=st.refit_all,
    )
    out_dir = _output_dir(cfg, project_root)
    (out_dir / "sem_train_metrics.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    promoted = promote_to_checkpoints(project_root, train_dir, names=["sem.pt"])
    result["promoted"] = promoted
    result["output_dir"] = str(out_dir)
    return result


def mode_train_graph(cfg: AgentConfig, project_root: Path) -> Dict[str, Any]:
    """SFT SmolLM2 from a pre-built dataset. Does not collect or rewrite JSONL."""
    from .graphs.dataset import load_graph_sft_dataset
    from .graphs.sft import load_jsonl, train_graph_sft

    out_dir = _output_dir(cfg, project_root)
    train_dir = _train_dir(cfg, project_root)
    gcfg = cfg.graph
    data_dir = resolve(project_root, gcfg.dataset_dir)
    info = load_graph_sft_dataset(data_dir)
    train_rows = load_jsonl(Path(info["train_jsonl"]))
    val_rows = load_jsonl(Path(info["val_jsonl"]))
    if cfg.limit is not None:
        train_rows = train_rows[: cfg.limit]
        val_rows = val_rows[: max(cfg.limit // 5, 1)]
    if not train_rows:
        raise RuntimeError(f"empty graph train set: {info['train_jsonl']}")
    ckpt_dir = train_dir / "smollm_graph"
    device = pick_device(cfg.device)
    result = train_graph_sft(
        train_rows,
        val_rows,
        pretrained=gcfg.smol_pretrained,
        out_dir=ckpt_dir,
        project_root=project_root,
        epochs=gcfg.epochs,
        batch_size=gcfg.batch_size,
        grad_accum=gcfg.grad_accum,
        lr=gcfg.lr,
        weight_decay=gcfg.weight_decay,
        max_seq_len=gcfg.max_seq_len,
        warmup_ratio=gcfg.warmup_ratio,
        max_new_tokens=gcfg.max_new_tokens,
        eval_generate_n=gcfg.eval_generate_n,
        lora_r=gcfg.lora_r,
        seed=gcfg.seed,
        device=device,
        tb_run_name=f"{cfg.name}_graph",
    )
    promoted = promote_to_checkpoints(
        project_root, train_dir, names=["smollm_graph"]
    )
    result = {**info, **result, "promoted": promoted, "output_dir": str(out_dir)}
    (out_dir / "graph_train_metrics.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return result


def mode_validate_graph(cfg: AgentConfig, project_root: Path) -> Dict[str, Any]:
    """Run full generative graph validation through run_graph_valid.sh."""
    script = project_root / "run_graph_valid.sh"
    data_path = resolve(project_root, cfg.graph.dataset_dir) / "val.jsonl"
    checkpoint = resolve(project_root, cfg.paths.checkpoints_dir) / "smollm_graph"
    output = _output_dir(cfg, project_root) / "validation_results.json"
    command = [
        str(script),
        "--input",
        str(data_path),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        pick_device(cfg.device),
        "--max-new-tokens",
        str(cfg.graph.max_new_tokens),
    ]
    if cfg.limit is not None:
        command.extend(["--limit", str(cfg.limit)])

    logging.info("running graph validation: %s", " ".join(command))
    subprocess.run(command, cwd=project_root, check=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    return {key: value for key, value in payload.items() if key != "results"}


def mode_prepare_features(cfg: AgentConfig, project_root: Path) -> Dict[str, Any]:
    """Report-only: generate features + write CSV (same path as train_arbiter)."""
    _, _, _, info = build_feature_matrix(cfg, project_root, write_csv=True)
    return info


def mode_train_arbiter(cfg: AgentConfig, project_root: Path) -> Dict[str, Any]:
    """
    1) Generate features → matrix + features.csv in outputs/<exp>/
    2) Train XGBoost; weights → train_ckpt/<exp>/; promote to checkpoints/
    3) Metrics / shap → outputs/<exp>/
    """
    train_dir = _train_dir(cfg, project_root)
    out_dir = _output_dir(cfg, project_root)
    X, y, _metas, feat_info = build_feature_matrix(cfg, project_root, write_csv=True)
    model, scaler, thr, metrics, y_test, proba = train_arbiter(
        X,
        y,
        test_ratio=cfg.arbiter.test_ratio,
        seed=cfg.arbiter.seed,
        n_estimators=cfg.arbiter.n_estimators,
        max_depth=cfg.arbiter.max_depth,
        learning_rate=cfg.arbiter.learning_rate,
        project_root=project_root,
        tb_run_name=f"{cfg.name}_arbiter",
    )
    save_arbiter(
        model,
        scaler,
        thr,
        train_dir / "arbiter.pkl",
        train_dir / "scaler.pkl",
        threshold_meta={
            "f1": metrics.get("f1"),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "n_test": metrics.get("n_test"),
        },
    )
    shap_path = maybe_shap_summary(
        model, scaler, X, out_dir / "shap_summary.json"
    )
    metrics["shap_summary"] = shap_path
    metrics["feature_generation"] = feat_info
    (out_dir / "arbiter_train_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    promoted = promote_to_checkpoints(
        project_root,
        train_dir,
        names=["arbiter.pkl", "scaler.pkl", "decision_threshold.json"],
    )
    metrics["promoted"] = promoted
    metrics["train_ckpt"] = str(train_dir)
    metrics["output_dir"] = str(out_dir)
    return metrics


def mode_eval(cfg: AgentConfig, project_root: Path) -> Dict[str, Any]:
    """Eval on golden only (never used in feature gen / train_arbiter)."""
    out_dir = _output_dir(cfg, project_root)
    arb = resolve(project_root, cfg.paths.arbiter_ckpt)
    if not arb.is_file():
        logging.info("arbiter missing in checkpoints/ — training first")
        mode_train_arbiter(cfg, project_root)

    rows = _load_eval_rows(cfg, project_root)
    agent = VerificationAgent(cfg, project_root, load_arbiter=True)
    preds = agent.predict_rows(rows)

    slim = []
    for p in preds:
        slim.append(
            {
                "sample_id": p.get("sample_id"),
                "image_path": p.get("image_path"),
                "label": p.get("label"),
                "pred_label": p.get("pred_label"),
                "verdict": p.get("verdict"),
                "prob_keep": p.get("prob_keep"),
                "decision_threshold": p.get("decision_threshold"),
                "probs": {
                    "match": p.get("prob_keep"),
                    "mismatch": 1.0 - float(p.get("prob_keep") or 0.0),
                },
                "judge_confidences": p.get("judge_confidences"),
                "features": p.get("features"),
                "gold_is_corrupted": p.get("gold_is_corrupted"),
                "arbiter": p.get("arbiter"),
            }
        )

    (out_dir / "predictions.json").write_text(
        json.dumps(slim, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in slim:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = summarize_predictions(slim)
    metrics["split"] = "golden"
    metrics["n_rows"] = len(slim)
    metrics["train_source"] = "data/figures excluding golden"
    if agent.arbiter is not None:
        metrics["decision_threshold"] = agent.arbiter.decision_threshold
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    logging.info("eval golden metrics: %s", metrics)
    return metrics


def mode_ablate(cfg: AgentConfig, project_root: Path) -> Dict[str, Any]:
    """Run leakage-safe variants; golden is final evaluation only."""
    train_rows = _load_train_rows(cfg, project_root)
    golden_rows = _load_eval_rows(cfg, project_root)
    return run_ablations(
        cfg,
        project_root,
        train_rows,
        golden_rows,
        _output_dir(cfg, project_root),
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verification Agent")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "train_router",
            "train_sem",
            "prepare_features",
            "train_arbiter",
            "train_graph",
            "validate_graph",
            "eval",
            "ablate",
            "all",
        ],
        default="eval",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name → outputs/<name>/ + train_ckpt/<name>/",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="Deprecated alias for --exp-name (basename used if path given)",
    )
    args = parser.parse_args(argv)

    project_root = args.project_root or Path(__file__).resolve().parents[1]
    cfg = AgentConfig.from_yaml(args.config)
    if args.limit is not None:
        cfg.limit = args.limit
    exp = args.exp_name or args.dump_dir
    if exp is not None:
        # Accept "agent_v3" or legacy "outputs/agent_v3" / absolute path → basename
        exp_name = Path(str(exp)).name
        cfg.apply_exp_name(exp_name)

    out_dir = _output_dir(cfg, project_root)
    _train_dir(cfg, project_root)
    resolve(project_root, cfg.paths.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir)
    (out_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    logging.info(
        "exp=%s outputs=%s train_ckpt=%s checkpoints=%s",
        cfg.name,
        cfg.paths.output_dir,
        cfg.paths.train_ckpt_dir,
        cfg.paths.checkpoints_dir,
    )

    mode = args.mode
    result: Dict[str, Any]
    if mode == "train_router":
        result = mode_train_router(cfg, project_root)
    elif mode == "train_sem":
        result = mode_train_sem(cfg, project_root)
    elif mode == "prepare_features":
        result = mode_prepare_features(cfg, project_root)
    elif mode == "train_arbiter":
        result = mode_train_arbiter(cfg, project_root)
    elif mode == "train_graph":
        result = mode_train_graph(cfg, project_root)
    elif mode == "validate_graph":
        result = mode_validate_graph(cfg, project_root)
    elif mode == "eval":
        result = mode_eval(cfg, project_root)
    elif mode == "ablate":
        result = mode_ablate(cfg, project_root)
    else:  # all
        r1 = mode_train_router(cfg, project_root)
        r2 = mode_train_sem(cfg, project_root)
        r3 = mode_train_arbiter(cfg, project_root)
        r4 = mode_eval(cfg, project_root)
        result = {
            "train_router": r1,
            "train_sem": r2,
            "train_arbiter": r3,
            "eval": r4,
        }

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
