"""SFT SmolLM2-Instruct to emit scene-graph JSON from a visual claim."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .schema import (
    FEWSHOT_CLAIM,
    FEWSHOT_GRAPH,
    NODE_KINDS,
    SYSTEM_PROMPT,
    canonicalize,
    dump_graph,
    graph_sets,
    parse_graph_json,
    prf_from_counts,
)

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_messages(claim: str, graph: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """System vocab + one few-shot, then the target claim."""
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Claim:\n{FEWSHOT_CLAIM}"},
        {"role": "assistant", "content": dump_graph(FEWSHOT_GRAPH)},
        {"role": "user", "content": f"Claim:\n{claim.strip()}"},
    ]
    if graph is not None:
        msgs.append({"role": "assistant", "content": dump_graph(graph)})
    return msgs


class GraphSFTDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        tokenizer,
        max_seq_len: int,
    ):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.rows[idx]
        tok = self.tokenizer
        prompt_ids = tok.apply_chat_template(
            build_messages(r["claim"]),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        full_ids = tok.apply_chat_template(
            build_messages(r["claim"], r["graph"]),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[: self.max_seq_len]
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100
        if all(x == -100 for x in labels):
            labels[-1] = full_ids[-1]
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _collate(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].size(0) for x in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, x in enumerate(batch):
        n = x["input_ids"].size(0)
        input_ids[i, :n] = x["input_ids"]
        labels[i, :n] = x["labels"]
        attn[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


@torch.inference_mode()
def _eval_loss(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        bs = int(batch["input_ids"].size(0))
        total += float(out.loss.item()) * bs
        n += bs
    return total / max(n, 1)


def _add_set_counts(gold: set, pred: set) -> Tuple[int, int, int]:
    return len(gold & pred), len(pred - gold), len(gold - pred)


def _add_bag_counts(gold: Counter, pred: Counter) -> Tuple[int, int, int]:
    keys = set(gold) | set(pred)
    tp = fp = fn = 0
    for k in keys:
        g, p = int(gold[k]), int(pred[k])
        tp += min(g, p)
        fp += max(0, p - g)
        fn += max(0, g - p)
    return tp, fp, fn


@torch.inference_mode()
def eval_generation(
    model,
    tokenizer,
    rows: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    max_new_tokens: int = 512,
    limit: int = 64,
    seed: int = 42,
) -> Dict[str, Any]:
    """Micro TP/FP/FN over the eval sample.

    query: set of node.query per claim.
    node:  exact (query, kind) pairs — strict node match on val gold.
    kind:  bag of node.kind per claim (text can appear many times).
    by_kind[k]: query sets restricted to nodes with that kind.
    kind_agree: among query-TP, fraction with the same kind.
    """
    rng = random.Random(seed)
    sample = list(rows)
    if len(sample) > limit:
        sample = rng.sample(sample, limit)
    model.eval()

    q_tp = q_fp = q_fn = 0
    k_tp = k_fp = k_fn = 0
    n_tp = n_fp = n_fn = 0
    e_tp = e_fp = e_fn = 0
    by_kind = {k: [0, 0, 0] for k in NODE_KINDS}
    kind_agree_ok = 0
    kind_agree_n = 0
    json_ok = 0

    for r in sample:
        gold = canonicalize(r["graph"])
        gold_nodes = gold.get("nodes") or []
        gold_q = {n["query"] for n in gold_nodes}
        gold_n = {(n["query"], n["kind"]) for n in gold_nodes}
        gold_k = Counter(n["kind"] for n in gold_nodes)
        gold_e = graph_sets(gold)[1]
        gold_q_by_k = {
            k: {n["query"] for n in gold_nodes if n["kind"] == k} for k in NODE_KINDS
        }
        gold_kind_of = {n["query"]: n["kind"] for n in gold_nodes}

        prompt = tokenizer.apply_chat_template(
            build_messages(r["claim"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = tokenizer.decode(out[0, enc["input_ids"].size(1) :], skip_special_tokens=True)
        pred = parse_graph_json(gen)
        if pred is None:
            pred_nodes: List[Dict[str, Any]] = []
            pred_q: set = set()
            pred_n: set = set()
            pred_k: Counter = Counter()
            pred_e: set = set()
            pred_q_by_k = {k: set() for k in NODE_KINDS}
            pred_kind_of: Dict[str, str] = {}
        else:
            json_ok += 1
            pred_nodes = pred.get("nodes") or []
            pred_q = {n["query"] for n in pred_nodes}
            pred_n = {(n["query"], n["kind"]) for n in pred_nodes}
            pred_k = Counter(n["kind"] for n in pred_nodes)
            pred_e = graph_sets(pred)[1]
            pred_q_by_k = {
                k: {n["query"] for n in pred_nodes if n["kind"] == k} for k in NODE_KINDS
            }
            pred_kind_of = {n["query"]: n["kind"] for n in pred_nodes}

        # f1 считаем по query -- есть ли все query что и в графе
        tp, fp, fn = _add_set_counts(gold_q, pred_q)
        q_tp += tp
        q_fp += fp
        q_fn += fn
        # f1 считаем по node -- есть ли все node что и в графе
        tp, fp, fn = _add_set_counts(gold_n, pred_n)
        n_tp += tp
        n_fp += fp
        n_fn += fn
        # f1 считаем по kind -- есть ли все kind что и в графе
        tp, fp, fn = _add_bag_counts(gold_k, pred_k)
        k_tp += tp
        k_fp += fp
        k_fn += fn
        # f1 считаем по edge -- есть ли все edge что и в графе
        tp, fp, fn = _add_set_counts(gold_e, pred_e)
        e_tp += tp
        e_fp += fp
        e_fn += fn
        for k in NODE_KINDS:
            # f1 считаем по всем query, что есть в конкретном kind-е (покрытие query внутри kind-а)
            tp, fp, fn = _add_set_counts(gold_q_by_k[k], pred_q_by_k[k])
            by_kind[k][0] += tp
            by_kind[k][1] += fp
            by_kind[k][2] += fn
        
        for q in gold_q & pred_q:
            kind_agree_n += 1
            if gold_kind_of.get(q) == pred_kind_of.get(q):
                kind_agree_ok += 1

    n = max(len(sample), 1)
    out_m: Dict[str, Any] = {
        "n": float(n),
        "json_ok": json_ok / n,
        "query": prf_from_counts(q_tp, q_fp, q_fn),
        "node": prf_from_counts(n_tp, n_fp, n_fn),
        "kind": prf_from_counts(k_tp, k_fp, k_fn),
        "edge": prf_from_counts(e_tp, e_fp, e_fn),
        
        # доля query, у которых kind совпадает в gold и pred
        "kind_agree": (kind_agree_ok / kind_agree_n) if kind_agree_n else 1.0,
        "by_kind": {
            k: prf_from_counts(tp, fp, fn) for k, (tp, fp, fn) in by_kind.items()
        },
    }
    return out_m


def flatten_gen_metrics(m: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {
        "n": float(m["n"]),
        "json_ok": float(m["json_ok"]),
        "kind_agree": float(m["kind_agree"]),
    }
    for group in ("query", "node", "kind", "edge"):
        for k, v in m[group].items():
            flat[f"{group}_{k}"] = float(v)
    for kind, d in m["by_kind"].items():
        for k, v in d.items():
            flat[f"kind_{kind}_{k}"] = float(v)
    return flat


def train_graph_sft(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    *,
    pretrained: str | Path,
    out_dir: Path,
    project_root: Path,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    max_seq_len: int = 1536,
    warmup_ratio: float = 0.03,
    max_new_tokens: int = 512,
    eval_generate_n: int = 64,
    lora_r: int = 16,
    seed: int = 42,
    device: str = "cuda",
    tb_run_name: str = "graph_sft",
) -> Dict[str, Any]:
    from peft import LoraConfig, TaskType, get_peft_model

    from ..tb_logging import make_summary_writer

    device_t = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    tok = AutoTokenizer.from_pretrained(str(pretrained), padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device_t.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(pretrained),
        torch_dtype=dtype,
        device_map=None,
    )
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.to(device_t)
    model.print_trainable_parameters()

    train_ds = GraphSFTDataset(train_rows, tok, max_seq_len)
    val_ds = GraphSFTDataset(val_rows, tok, max_seq_len)
    pad_id = int(tok.pad_token_id)

    def loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lambda b: _collate(b, pad_id),
        )

    train_loader = loader(train_ds, True)
    val_loader = loader(val_ds, False)

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max(int(epochs * steps_per_epoch / max(grad_accum, 1)), 1)
    warmup = max(int(total_steps * warmup_ratio), 1)

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * step / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return lr * 0.5 * (1.0 + torch.cos(torch.tensor(prog * 3.14159265)).item())

    writer, tb_dir = make_summary_writer(project_root, tb_run_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "graph SFT start device=%s n_train=%d n_val=%d epochs=%d bs=%d accum=%d seq=%d lora_r=%d",
        device_t,
        len(train_rows),
        len(val_rows),
        epochs,
        batch_size,
        grad_accum,
        max_seq_len,
        lora_r,
    )

    global_step = 0
    opt_step = 0
    best_val = float("inf")
    history: List[Dict[str, Any]] = []
    model.train()
    opt.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        model.train()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device_t, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / max(grad_accum, 1)
            loss.backward()
            bs = int(batch["input_ids"].size(0))
            total_loss += float(out.loss.item()) * bs
            n += bs
            global_step += 1
            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for g in opt.param_groups:
                    g["lr"] = lr_at(opt_step)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if writer is not None:
                    writer.add_scalar("train/loss_step", float(out.loss.item()), opt_step)
                    writer.add_scalar("train/lr", opt.param_groups[0]["lr"], opt_step)
                opt_step += 1
            if opt_step == 1 and i < grad_accum:
                logger.info("graph SFT first step ok loss=%.4f", float(out.loss.item()))

        train_loss = total_loss / max(n, 1)
        val_loss = _eval_loss(model, val_loader, device_t)
        gen_metrics = eval_generation(
            model,
            tok,
            val_rows,
            device=device_t,
            max_new_tokens=max_new_tokens,
            limit=eval_generate_n,
            seed=seed,
        )
        rec = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **flatten_gen_metrics(gen_metrics),
        }
        history.append(rec)
        logger.info(
            "graph SFT epoch %d/%d train_loss=%.4f val_loss=%.4f json_ok=%.3f "
            "query_f1=%.3f node_f1=%.3f node_tp=%.0f node_fp=%.0f node_fn=%.0f "
            "kind_f1=%.3f kind_agree=%.3f edge_f1=%.3f",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            rec["json_ok"],
            rec["query_f1"],
            rec["node_f1"],
            rec["node_tp"],
            rec["node_fp"],
            rec["node_fn"],
            rec["kind_f1"],
            rec["kind_agree"],
            rec["edge_f1"],
        )
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/json_ok", rec["json_ok"], epoch)
            writer.add_scalar("val/kind_agree", rec["kind_agree"], epoch)
            for group in ("query", "node", "kind", "edge"):
                for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
                    writer.add_scalar(f"val/{group}/{key}", rec[f"{group}_{key}"], epoch)
            for kind in NODE_KINDS:
                for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
                    writer.add_scalar(
                        f"val/kind/{kind}/{key}", rec[f"kind_{kind}_{key}"], epoch
                    )
            writer.flush()

        if val_loss <= best_val:
            best_val = val_loss
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            (out_dir / "metrics.json").write_text(
                json.dumps(rec, indent=2) + "\n", encoding="utf-8"
            )

    if writer is not None:
        writer.close()

    # merge adapter into a small inference folder pointer (adapter only)
    meta = {
        "kind": "smol_graph_sft_v1",
        "base": str(pretrained),
        "best_val_loss": best_val,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "history": history,
        "tfevents": str(tb_dir),
        "ckpt": str(out_dir),
    }
    (out_dir / "train_metrics.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return meta
