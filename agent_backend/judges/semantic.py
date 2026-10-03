"""Judge A: SigLIP (+ optional fine-tuned match head) → score_sem ∈ [0, 1]."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .match_head import (
    MatchHead,
    as_embed as _as_embed,
    resize_text_position_embeddings,
)

logger = logging.getLogger(__name__)


class JudgeSemantic:
    def __init__(
        self,
        pretrained: str | Path,
        *,
        device: str = "cuda",
        max_text_length: int = 64,
        batch_size: int = 8,
        sem_ckpt: str | Path | None = None,
    ):
        self.device = torch.device(device)
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(str(pretrained))
        self.model = AutoModel.from_pretrained(str(pretrained))
        resize_text_position_embeddings(self.model, self.max_text_length)
        self.head: Optional[MatchHead] = None
        self._mode = "native_cosine"

        ckpt_path = Path(sem_ckpt) if sem_ckpt else None
        if ckpt_path is not None and ckpt_path.is_file():
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if blob.get("kind") == "sem_match_v1":
                checkpoint_length = int(
                    blob.get("max_text_length") or self.max_text_length
                )
                resize_text_position_embeddings(self.model, checkpoint_length)
                self.model.load_state_dict(blob["backbone_state_dict"], strict=True)
                dim = int(blob.get("embed_dim") or 768)
                self.head = MatchHead(dim)
                self.head.load_state_dict(blob["head_state_dict"])
                self.head.to(self.device)
                self.head.eval()
                self.max_text_length = checkpoint_length
                self._mode = "finetuned_head"
                logger.info("Loaded fine-tuned sem checkpoint: %s", ckpt_path)
            else:
                logger.warning("Unknown sem ckpt kind in %s — using native cosine", ckpt_path)
        else:
            if sem_ckpt:
                logger.warning(
                    "sem_ckpt missing (%s) — Judge A uses native SigLIP cosine (weak on L2)",
                    sem_ckpt,
                )

        self.model.to(self.device)
        self.model.eval()
        self._last_confidence = 0.0

    def get_confidence(self) -> float:
        return float(self._last_confidence)

    @torch.inference_mode()
    def score_pair(self, image_path: str | Path, text: str) -> float:
        return self.score_batch([(str(image_path), text)])[0]

    @torch.inference_mode()
    def score_batch(self, pairs: Sequence[tuple[str, str]]) -> List[float]:
        """Return list of match scores in [0, 1] (higher = better match)."""
        out: List[float] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = list(pairs[start : start + self.batch_size])
            images, texts = [], []
            for path, text in chunk:
                images.append(Image.open(path).convert("RGB"))
                t = (text or "").strip() or "scientific figure"
                texts.append(t)
            enc = self.processor(
                text=texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            outputs = self.model(**enc)
            if getattr(outputs, "image_embeds", None) is not None:
                img = outputs.image_embeds
                txt = outputs.text_embeds
            else:
                img = _as_embed(
                    self.model.get_image_features(pixel_values=enc["pixel_values"])
                )
                txt = _as_embed(
                    self.model.get_text_features(
                        input_ids=enc["input_ids"],
                        attention_mask=enc.get("attention_mask"),
                    )
                )

            if self.head is not None:
                logits = self.head(img, txt)
                probs = torch.sigmoid(logits).detach().cpu()
            else:
                img_n = F.normalize(img, dim=-1)
                txt_n = F.normalize(txt, dim=-1)
                cosine = (img_n * txt_n).sum(dim=-1)
                scale = getattr(self.model, "logit_scale", None)
                if scale is not None:
                    logits = cosine * scale.exp().clamp(max=100)
                else:
                    logits = cosine * 20.0
                probs = torch.sigmoid(logits).detach().cpu()

            for p in probs.tolist():
                score = float(p)
                out.append(score)
                self._last_confidence = abs(score - 0.5) * 2.0
        if out:
            self._last_confidence = abs(out[-1] - 0.5) * 2.0
        return out

    def predict(self, image_path: str | Path, text: str) -> float:
        return self.score_pair(image_path, text)
