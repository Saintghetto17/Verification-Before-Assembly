"""Judge B: OCR caption alignment → score_struct ∈ [0, 1]."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{1,}")
_DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")

# Tokens that don't help caption↔OCR alignment
_LABEL_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "been", "have", "has", "had", "not", "but", "all", "can", "our", "their",
    "into", "over", "under", "than", "then", "when", "which", "while", "also",
    "using", "used", "use", "based", "show", "shown", "shows", "figure", "fig",
    "table", "results", "method", "methods", "model", "models", "data", "each",
    "both", "between", "after", "before", "through", "during", "where", "what",
    "how", "et", "al", "vs", "via", "per", "see", "left", "right", "top",
    "bottom", "panel", "panels", "image", "images", "plot", "plots", "chart",
    "graph", "graphs", "schematic", "diagram", "illustration",
}


def _label_tokens(s: str) -> Set[str]:
    """Alphanumeric label-like tokens from caption / OCR (no stopwords)."""
    out: Set[str] = set()
    for t in _TOKEN_RE.findall(s or ""):
        tl = t.lower()
        if tl in _LABEL_STOP or len(tl) < 2:
            continue
        out.add(tl)
    return out


def _digits(s: str) -> Set[str]:
    return set(_DIGIT_RE.findall(s or ""))


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class JudgeStructural:
    def __init__(
        self,
        *,
        prefer_ocr: bool = True,
        use_gpu: Optional[bool] = None,
        ocr_batch_size: int = 64,
    ):
        self._last_confidence = 0.0
        self._ocr = None
        self.ocr_backend: str = "none"
        self.use_gpu = bool(_cuda_available() if use_gpu is None else use_gpu)
        self.ocr_batch_size = max(int(ocr_batch_size), 1)
        if prefer_ocr:
            self._ocr = self._init_ocr()
            if self._ocr is not None:
                self.ocr_backend = self._ocr[0]
                logger.info(
                    "JudgeStructural OCR backend: %s (gpu=%s, ocr_batch_size=%d)",
                    self.ocr_backend,
                    self.use_gpu and self.ocr_backend == "easyocr",
                    self.ocr_batch_size,
                )
            else:
                logger.warning(
                    "JudgeStructural: no OCR (install tesseract+pytesseract or easyocr) "
                    "— score_struct will be neutral 0.5 with conf=0"
                )

    def _init_ocr(self) -> Optional[Tuple[str, object]]:
        try:
            import pytesseract  # type: ignore

            # Fail fast if binary missing
            ver = pytesseract.get_tesseract_version()
            logger.info("JudgeStructural: tesseract binary ok version=%s", ver)
            return ("tesseract", pytesseract)
        except Exception as e:
            logger.warning(
                "JudgeStructural: tesseract unavailable (%s: %s) — trying easyocr",
                type(e).__name__,
                e,
            )
        try:
            import easyocr  # type: ignore

            logger.info(
                "JudgeStructural: easyocr import ok, building Reader(en, gpu=%s)…",
                self.use_gpu,
            )
            reader = easyocr.Reader(["en"], gpu=self.use_gpu, verbose=False)
            return ("easyocr", reader)
        except Exception as e:
            logger.warning(
                "JudgeStructural: easyocr init failed (%s: %s)",
                type(e).__name__,
                e,
                exc_info=True,
            )
            return None

    def get_confidence(self) -> float:
        return float(self._last_confidence)

    def _ocr_text(self, image_path: str | Path) -> str:
        if self._ocr is None:
            return ""
        kind, eng = self._ocr
        try:
            if kind == "tesseract":
                import cv2

                img = cv2.imread(str(image_path))
                if img is None:
                    return ""
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return eng.image_to_string(gray) or ""
            results = eng.readtext(
                str(image_path),
                detail=0,
                paragraph=True,
                batch_size=self.ocr_batch_size,
            )
            return " ".join(results) if results else ""
        except Exception as e:
            logger.warning("OCR failed on %s: %s", image_path, e)
            return ""

    def _score_alignment(self, text: str, ocr_text: str) -> float:
        text = (text or "").strip()
        ocr_text = (ocr_text or "").strip()
        if not ocr_text:
            self._last_confidence = 0.0
            return 0.5

        a = _label_tokens(text)
        b = _label_tokens(ocr_text)
        text_digits = _digits(text)
        ocr_digits = _digits(ocr_text)

        if a and b:
            jacc = len(a & b) / len(a | b)
        else:
            jacc = 0.0

        if text_digits:
            digit_hit = len(text_digits & ocr_digits) / len(text_digits)
            score = float(np.clip(0.5 * jacc + 0.5 * digit_hit, 0.0, 1.0))
        else:
            score = float(np.clip(jacc, 0.0, 1.0))

        evidence = min(1.0, (len(a) + len(text_digits)) / 8.0)
        ocr_rich = min(1.0, (len(b) + len(ocr_digits)) / 12.0)
        self._last_confidence = float(0.35 + 0.65 * evidence * ocr_rich)
        return score

    def predict(self, image_path: str | Path, text: str) -> float:
        """
        Align OCR on image with *caption-like* text (not page context).

        No OCR / empty OCR → 0.5 with confidence 0 (arbiter should ignore).
        """
        return self._score_alignment(text, self._ocr_text(image_path))

    def predict_batch(
        self,
        image_paths: Sequence[str | Path],
        texts: Sequence[str],
    ) -> List[float]:
        """Score many pairs; uses easyocr readtext_batched when available."""
        if len(image_paths) != len(texts):
            raise ValueError("image_paths and texts must have the same length")
        if not image_paths:
            return []
        if self._ocr is None:
            self._last_confidence = 0.0
            return [0.5] * len(image_paths)

        kind, eng = self._ocr
        ocr_texts: List[str] = [""] * len(image_paths)
        if kind == "easyocr" and hasattr(eng, "readtext_batched") and len(image_paths) > 1:
            try:
                paths = [str(p) for p in image_paths]
                # Same-size resize required by readtext_batched
                batched = eng.readtext_batched(
                    paths,
                    n_width=1280,
                    n_height=1280,
                    detail=0,
                    paragraph=True,
                    batch_size=self.ocr_batch_size,
                )
                for i, res in enumerate(batched):
                    if isinstance(res, str):
                        ocr_texts[i] = res
                    elif isinstance(res, (list, tuple)):
                        ocr_texts[i] = " ".join(str(x) for x in res)
                    else:
                        ocr_texts[i] = str(res) if res else ""
            except Exception as e:
                logger.warning(
                    "easyocr readtext_batched failed (%s) — falling back to per-image",
                    e,
                )
                ocr_texts = [self._ocr_text(p) for p in image_paths]
        else:
            ocr_texts = [self._ocr_text(p) for p in image_paths]

        return [self._score_alignment(t, o) for t, o in zip(texts, ocr_texts)]
