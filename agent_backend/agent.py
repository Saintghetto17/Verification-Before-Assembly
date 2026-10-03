"""VerificationAgent: router + judges + features + arbiter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .arbiter import Arbiter
from .config import AgentConfig, pick_device, resolve
from .features import FEATURE_NAMES, build_feature_vector
from .judges import JudgeObject, JudgeSemantic, JudgeStructural
from .router import ImageRouter
from .text_utils import row_texts, short_caption

logger = logging.getLogger(__name__)


class VerificationAgent:
    def __init__(self, cfg: AgentConfig, project_root: Path, *, load_arbiter: bool = True):
        self.cfg = cfg
        self.project_root = Path(project_root)
        self.device = pick_device(cfg.device)

        sem_ckpt = resolve(self.project_root, cfg.paths.sem_ckpt)
        self.judge_sem = JudgeSemantic(
            cfg.siglip.pretrained,
            device=self.device,
            max_text_length=cfg.siglip.max_text_length,
            batch_size=cfg.siglip.batch_size,
            sem_ckpt=sem_ckpt,
        )
        ocr_gpu = cfg.arbiter.ocr_use_gpu
        if ocr_gpu is None:
            ocr_gpu = self.device == "cuda"
        self.judge_struct = JudgeStructural(
            prefer_ocr=True,
            use_gpu=ocr_gpu,
            ocr_batch_size=cfg.arbiter.ocr_batch_size,
        )
        self.judge_obj = JudgeObject(
            device=self.device,
            max_text_length=cfg.siglip.max_text_length,
            top_k=3,
        )
        # Share SigLIP encoder with object judge
        self.judge_obj.attach_siglip(self.judge_sem.model, self.judge_sem.processor)

        router_ckpt = resolve(self.project_root, cfg.paths.router_ckpt)
        self.router = ImageRouter(
            router_ckpt if router_ckpt.is_file() else None,
            classes=cfg.router.classes,
            device=self.device,
            image_size=cfg.router.image_size,
        )

        self.arbiter: Optional[Arbiter] = None
        if load_arbiter:
            ap = resolve(self.project_root, cfg.paths.arbiter_ckpt)
            sp = resolve(self.project_root, cfg.paths.scaler_ckpt)
            if ap.is_file() and sp.is_file():
                self.arbiter = Arbiter(
                    ap,
                    sp,
                    decision_threshold=cfg.arbiter.decision_threshold,
                )
            else:
                logger.warning("Arbiter checkpoints missing; predict() will stop at features")

    def compute_features(
        self,
        image_path: str | Path,
        text: str,
        *,
        caption: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        text → Judge A (full premise / page context).
        caption → Judges B/C (short); if omitted, derived from text.
        """
        text = (text or "").strip()
        short = (caption or "").strip() or short_caption(premise=text, text_block=text)
        score_sem = self.judge_sem.predict(image_path, text)
        score_struct = self.judge_struct.predict(image_path, short)
        score_obj = self.judge_obj.predict(image_path, short)
        probs = self.router.predict_proba(image_path)
        return build_feature_vector(
            score_sem=score_sem,
            score_struct=score_struct,
            score_obj=score_obj,
            router_probs=probs,
            image_path=image_path,
            text=text,
        )

    def compute_features_batch(
        self,
        image_paths: Sequence[str | Path],
        texts: Sequence[str],
        *,
        captions: Optional[Sequence[Optional[str]]] = None,
    ) -> List[Dict[str, float]]:
        """Batched feature vectors: SigLIP sem + OCR struct batched; obj/router per item."""
        n = len(image_paths)
        if n == 0:
            return []
        if len(texts) != n:
            raise ValueError("texts length must match image_paths")
        caps = list(captions) if captions is not None else [None] * n
        if len(caps) != n:
            raise ValueError("captions length must match image_paths")

        premises = [(t or "").strip() for t in texts]
        shorts = [
            ((c or "").strip() or short_caption(premise=p, text_block=p))
            for p, c in zip(premises, caps)
        ]
        score_sems = self.judge_sem.score_batch(
            [(str(p), t) for p, t in zip(image_paths, premises)]
        )
        score_structs = self.judge_struct.predict_batch(image_paths, shorts)
        out: List[Dict[str, float]] = []
        for i in range(n):
            score_obj = self.judge_obj.predict(image_paths[i], shorts[i])
            probs = self.router.predict_proba(image_paths[i])
            out.append(
                build_feature_vector(
                    score_sem=score_sems[i],
                    score_struct=score_structs[i],
                    score_obj=score_obj,
                    router_probs=probs,
                    image_path=image_paths[i],
                    text=premises[i],
                )
            )
        return out

    def judge_confidences(self) -> List[float]:
        return [
            self.judge_sem.get_confidence(),
            self.judge_struct.get_confidence(),
            self.judge_obj.get_confidence(),
        ]

    def predict(
        self,
        image_path: str | Path,
        text: str,
        *,
        caption: Optional[str] = None,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        feats = self.compute_features(image_path, text, caption=caption)
        confs = self.judge_confidences()
        out: Dict[str, Any] = {
            "image_path": str(image_path),
            "premise": text,
            "caption_bc": (caption or "").strip()
            or short_caption(premise=text, text_block=text),
            "features": feats,
            "judge_confidences": {
                "sem": confs[0],
                "struct": confs[1],
                "obj": confs[2],
            },
            "ocr_backend": getattr(self.judge_struct, "ocr_backend", "none"),
            "feature_names": FEATURE_NAMES,
        }
        if label is not None:
            out["label"] = label
        if self.arbiter is None:
            # Fallback: use score_sem only
            sem = feats["score_sem"]
            out["prob_keep"] = sem
            out["verdict"] = "PASS" if sem >= 0.5 else "REGENERATE"
            out["pred_label"] = "match" if sem >= 0.5 else "mismatch"
            out["arbiter"] = False
            return out

        decision = self.arbiter.predict(feats, confs)
        out.update(decision)
        out["arbiter"] = True
        return out

    def predict_rows(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for i, r in enumerate(rows):
            premise, caption = row_texts(r)
            pred = self.predict(
                r["image_path"], premise, caption=caption, label=r.get("label")
            )
            pred["sample_id"] = r.get("sample_id")
            pred["split"] = r.get("split")
            pred["gold_is_corrupted"] = r.get("gold_is_corrupted")
            results.append(pred)
            if (i + 1) % 25 == 0:
                logger.info("agent progress %d/%d", i + 1, len(rows))
        return results
