"""37-feature vector for the XGBoost arbiter."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

FEATURE_NAMES: List[str] = [
    # 3 judge scores
    "score_sem",
    "score_struct",
    "score_obj",
    # 4 router probs
    "p_byt",
    "p_video",
    "p_graph",
    "p_scheme",
    # 3 weighted
    "sem_w",
    "struct_w",
    "obj_w",
    # 1 router conf
    "router_conf",
    # 15 image meta
    "ratio",
    "mean_brightness",
    "std_brightness",
    "entropy",
    "edge_density",
    "color_variance",
    "texture_contrast",
    "num_blobs",
    "avg_blob_area",
    "horizontal_lines",
    "vertical_lines",
    "symmetry_score",
    "saturation_mean",
    "saturation_std",
    "sharpness",
    # 11 text meta
    "len_words",
    "len_chars",
    "num_digits",
    "num_verbs",
    "num_nouns",
    "num_entities",
    "has_numbers",
    "has_actions",
    "sentence_complexity",
    "stopword_ratio",
    "unique_word_ratio",
]

assert len(FEATURE_NAMES) == 37

BALANCED_DERIVED_FEATURE_NAMES: List[str] = [
    "sem_margin",
    "struct_margin",
    "obj_margin",
    "judge_mean",
    "judge_std",
    "judge_range",
    "judge_disagreement_mean",
    "router_entropy",
    "router_margin",
    "ocr_text_consistency",
    "ocr_numeric_consistency",
]
BALANCED_FEATURE_NAMES: List[str] = [
    *FEATURE_NAMES,
    *BALANCED_DERIVED_FEATURE_NAMES,
]

_STOP = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "this",
    "that", "these", "those", "it", "its", "we", "our", "they", "their", "figure",
}

_ACTION = {
    "show", "shows", "shown", "compare", "compares", "increase", "decrease",
    "train", "trained", "generate", "predict", "evaluate", "measure", "compute",
    "learn", "optimize", "converge", "improve", "reduce", "apply", "propose",
}

_VERB_SUFFIX = ("ed", "ing", "es", "ize", "ise", "ate")


def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def extract_image_meta(image_path: str | Path) -> List[float]:
    img = cv2.imread(str(image_path))
    if img is None:
        return [0.0] * 15
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(edges.mean() / 255.0)

    # blobs via simple threshold contours
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [float(cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > 20]
    num_blobs = float(len(areas))
    avg_blob = float(np.mean(areas)) if areas else 0.0

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=25, maxLineGap=6)
    h_lines = v_lines = 0.0
    if lines is not None:
        # OpenCV 4: (N, 1, 4); OpenCV 5: (N, 4)
        pts = lines.reshape(-1, 4)
        for x1, y1, x2, y2 in pts:
            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
            ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if ang < 20 or ang > 160:
                h_lines += 1
            elif 70 < ang < 110:
                v_lines += 1

    # symmetry: left-right correlation
    mid = w // 2
    if mid > 1:
        left = gray[:, :mid]
        right = cv2.flip(gray[:, w - mid :], 1)
        m = min(left.shape[1], right.shape[1])
        a = left[:, :m].astype(np.float32).ravel()
        b = right[:, :m].astype(np.float32).ravel()
        if a.std() > 1e-6 and b.std() > 1e-6:
            symmetry = float(np.corrcoef(a, b)[0, 1])
            if math.isnan(symmetry):
                symmetry = 0.0
        else:
            symmetry = 0.0
    else:
        symmetry = 0.0

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var() / 10000.0)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0

    # texture contrast via GLCM-lite: std of local mean subtract
    blur = cv2.blur(gray, (5, 5)).astype(np.float32)
    texture = float(np.std(gray.astype(np.float32) - blur) / 255.0)

    color_var = float(np.var(img.reshape(-1, 3).astype(np.float32), axis=0).mean() / (255.0**2))

    return [
        float(w / max(h, 1)),
        float(gray.mean() / 255.0),
        float(gray.std() / 255.0),
        _entropy(gray),
        edge_density,
        color_var,
        texture,
        num_blobs,
        avg_blob / max(w * h, 1),
        h_lines,
        v_lines,
        symmetry,
        float(sat.mean()),
        float(sat.std()),
        min(sharpness, 10.0),
    ]


def extract_text_meta(text: str) -> List[float]:
    text = text or ""
    words = re.findall(r"[A-Za-z0-9']+", text)
    n_words = len(words)
    lower = [w.lower() for w in words]
    num_digits = float(sum(c.isdigit() for c in text))
    num_verbs = float(sum(1 for w in lower if w in _ACTION or w.endswith(_VERB_SUFFIX)))
    num_nouns = float(sum(1 for w in lower if w not in _STOP and len(w) > 3 and w not in _ACTION))
    # crude "entities": Capitalized tokens
    entities = re.findall(r"\b[A-Z][a-zA-Z0-9\-]{2,}\b", text)
    num_entities = float(len(set(entities)))
    has_numbers = 1.0 if any(c.isdigit() for c in text) else 0.0
    has_actions = 1.0 if any(w in _ACTION for w in lower) else 0.0
    complexity = float(np.mean([len(w) for w in words])) if words else 0.0
    stop_ratio = float(sum(w in _STOP for w in lower) / n_words) if n_words else 0.0
    unique_ratio = float(len(set(lower)) / n_words) if n_words else 0.0
    return [
        float(n_words),
        float(len(text)),
        num_digits,
        num_verbs,
        num_nouns,
        num_entities,
        has_numbers,
        has_actions,
        complexity,
        stop_ratio,
        unique_ratio,
    ]


def build_feature_vector(
    *,
    score_sem: float,
    score_struct: float,
    score_obj: float,
    router_probs: Sequence[float],
    image_path: str | Path,
    text: str,
) -> Dict[str, float]:
    if len(router_probs) != 4:
        raise ValueError(f"router_probs must have 4 values, got {len(router_probs)}")
    p_byt, p_video, p_graph, p_scheme = map(float, router_probs)
    sem_w = score_sem * (p_byt + p_video)
    struct_w = score_struct * (p_graph + p_scheme)
    obj_w = score_obj * (p_scheme + p_byt)
    router_conf = max(p_byt, p_video, p_graph, p_scheme)
    img_meta = extract_image_meta(image_path)
    txt_meta = extract_text_meta(text)
    values = [
        float(score_sem),
        float(score_struct),
        float(score_obj),
        p_byt,
        p_video,
        p_graph,
        p_scheme,
        float(sem_w),
        float(struct_w),
        float(obj_w),
        float(router_conf),
        *img_meta,
        *txt_meta,
    ]
    if len(values) != 37:
        raise RuntimeError(f"expected 37 features, got {len(values)}")
    return dict(zip(FEATURE_NAMES, values))


def add_balanced_derived_features(
    feature_row: Dict[str, float],
) -> Dict[str, float]:
    """Add deterministic margins, disagreements, and OCR-consistency signals."""
    out = {name: float(value) for name, value in feature_row.items()}
    scores = np.asarray(
        [out["score_sem"], out["score_struct"], out["score_obj"]],
        dtype=np.float64,
    )
    router = np.asarray(
        [out["p_byt"], out["p_video"], out["p_graph"], out["p_scheme"]],
        dtype=np.float64,
    )
    router = np.clip(router, 0.0, 1.0)
    router = router / max(float(router.sum()), 1e-12)
    sorted_router = np.sort(router)
    pairwise = [
        abs(float(scores[i] - scores[j]))
        for i in range(len(scores))
        for j in range(i + 1, len(scores))
    ]
    has_numbers = bool(out.get("has_numbers", 0.0))
    out.update(
        {
            "sem_margin": abs(float(scores[0]) - 0.5),
            "struct_margin": abs(float(scores[1]) - 0.5),
            "obj_margin": abs(float(scores[2]) - 0.5),
            "judge_mean": float(np.mean(scores)),
            "judge_std": float(np.std(scores)),
            "judge_range": float(np.ptp(scores)),
            "judge_disagreement_mean": float(np.mean(pairwise)),
            "router_entropy": float(
                -np.sum(router * np.log(np.clip(router, 1e-12, 1.0)))
                / math.log(len(router))
            ),
            "router_margin": float(sorted_router[-1] - sorted_router[-2]),
            # Judge B is computed from OCR/caption token and numeric overlap.
            # Expose the score separately for all captions and for captions
            # containing numbers so models need not infer this interaction.
            "ocr_text_consistency": float(scores[1]),
            "ocr_numeric_consistency": float(scores[1]) if has_numbers else 0.5,
        }
    )
    return out
