"""Judge C: object/noun overlap via SigLIP zero-shot over noun candidates."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "been", "have", "has", "had", "not", "but", "all", "can", "our", "their",
    "into", "over", "under", "than", "then", "when", "which", "while", "also",
    "using", "used", "use", "based", "show", "shown", "shows", "figure", "fig",
    "table", "results", "method", "methods", "model", "models", "data", "each",
    "both", "between", "after", "before", "through", "during", "where", "what",
    "how", "thus", "further", "assess", "compared", "performed", "presented",
    "demonstrates", "demonstrate", "investigate", "investigated", "study",
    "studies", "analysis", "approach", "approaches", "paper", "section",
    "following", "corresponding", "respectively", "including", "via", "per",
    "left", "right", "top", "bottom", "panel", "panels", "image", "images",
    "more", "here", "text", "other", "same", "such", "these", "those", "very",
    "high", "low", "new", "old", "first", "second", "third", "total", "number",
}

_VERB_SUFFIX = ("ing", "ized", "ised", "ated", "ated")
_VERB_EXACT = {
    "did", "does", "done", "make", "made", "take", "took", "give", "gave",
    "get", "got", "see", "saw", "find", "found", "reveal", "reveals",
    "increase", "decrease", "train", "trained", "generate", "predict",
    "evaluate", "measure", "compute", "learn", "optimize", "apply", "propose",
    "combine", "extend", "provide", "provides", "differs", "differ",
}


def _is_bad_noun(w: str) -> bool:
    if w in _STOP or w in _VERB_EXACT:
        return True
    if len(w) < 3:
        return True
    if w.endswith(_VERB_SUFFIX) and len(w) > 5:
        return True
    return False


def extract_nouns(text: str, max_n: int = 12) -> Tuple[List[str], bool]:
    """
    Extract content-like tokens from caption-like text.

    Returns (nouns, used_fallback).
    Prefers longer / capitalized tokens; skips discourse verbs.
    """
    text = text or ""
    scored: List[Tuple[float, str]] = []
    seen = set()
    for m in _WORD_RE.finditer(text):
        raw = m.group(0)
        w = raw.lower()
        if w in seen or _is_bad_noun(w):
            continue
        seen.add(w)
        # Prefer capitalized / long scientific tokens
        score = float(len(w))
        if raw[0].isupper():
            score += 3.0
        if "-" in w or any(ch.isdigit() for ch in w):
            score += 2.0
        scored.append((score, w))

    scored.sort(key=lambda x: (-x[0], x[1]))
    nouns = [w for _, w in scored[:max_n]]
    if nouns:
        return nouns, False
    return ["diagram", "scene"], True


class JudgeObject:
    """
    Zero-shot: score whether image matches noun candidates from caption.
    score_obj = mean of top-k SigLIP match probs (default k=3).
    """

    def __init__(
        self,
        *,
        siglip_model=None,
        siglip_processor=None,
        device: str = "cuda",
        max_text_length: int = 64,
        top_k: int = 3,
    ):
        self.device = torch.device(device)
        self.max_text_length = max_text_length
        self.top_k = max(int(top_k), 1)
        self.model = siglip_model
        self.processor = siglip_processor
        self._last_confidence = 0.0
        self._pretrained: Optional[str] = None
        self._used_fallback = False

    def attach_siglip(self, model, processor) -> None:
        self.model = model
        self.processor = processor

    def ensure_model(self, pretrained: str) -> None:
        if self.model is not None and self.processor is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.model = AutoModel.from_pretrained(pretrained).to(self.device).eval()
        self._pretrained = pretrained

    def get_confidence(self) -> float:
        return float(self._last_confidence)

    @torch.inference_mode()
    def predict(self, image_path: str | Path, text: str, pretrained: Optional[str] = None) -> float:
        if self.model is None or self.processor is None:
            if not pretrained:
                raise RuntimeError("JudgeObject needs SigLIP model or pretrained path")
            self.ensure_model(pretrained)

        nouns, used_fallback = extract_nouns(text)
        self._used_fallback = used_fallback
        prompts = [f"a figure showing {n}" for n in nouns[:8]]
        image = Image.open(image_path).convert("RGB")

        img_enc = self.processor(images=[image], return_tensors="pt")
        txt_enc = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        img_enc = {k: v.to(self.device) for k, v in img_enc.items()}
        txt_enc = {k: v.to(self.device) for k, v in txt_enc.items()}

        def _as_embed(x):
            if torch.is_tensor(x):
                return x
            if hasattr(x, "pooler_output") and x.pooler_output is not None:
                return x.pooler_output
            if hasattr(x, "last_hidden_state"):
                return x.last_hidden_state[:, 0]
            raise TypeError(f"Unexpected feature type: {type(x)}")

        if hasattr(self.model, "get_image_features") and hasattr(self.model, "get_text_features"):
            img = F.normalize(_as_embed(self.model.get_image_features(**img_enc)), dim=-1)
            txt = F.normalize(_as_embed(self.model.get_text_features(**txt_enc)), dim=-1)
        else:
            enc = self.processor(
                text=prompts,
                images=[image] * len(prompts),
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.model(**enc)
            img = F.normalize(outputs.image_embeds[:1], dim=-1)
            txt = F.normalize(outputs.text_embeds, dim=-1)

        sims = (img * txt).sum(dim=-1)
        scale = getattr(self.model, "logit_scale", None)
        logits = sims * (scale.exp().clamp(max=100) if scale is not None else 20.0)
        probs = torch.sigmoid(logits).flatten()

        k = min(self.top_k, int(probs.numel()))
        top = torch.topk(probs, k=k).values
        score = float(top.mean().item())

        if used_fallback:
            self._last_confidence = 0.15
        else:
            # Higher conf when several nouns agree (top-k not dominated by one spike)
            spread = float(top.min().item()) if k > 1 else score
            self._last_confidence = float(
                min(1.0, abs(score - 0.5) * 2.0 * (0.5 + 0.5 * spread))
            )
        return score
