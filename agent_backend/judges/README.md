# Judges

Three independent scorers for a pair `(image_path, text)` → **score ∈ [0, 1]** (higher = better agreement).

| Judge | File | Text input | Question | Output |
|-------|------|------------|----------|--------|
| A semantic | `semantic.py` | full premise / page context | Does the text match this image? | `score_sem` |
| B structural | `structural.py` | short caption | Do OCR tokens / numbers agree? | `score_struct` |
| C object | `object.py` | short caption | Are mentioned entities visible? | `score_obj` |

`VerificationAgent` feeds Judge A the premise / text block and Judges B/C a short caption.

## Judge A — semantic

With `checkpoints/sem.pt`: fine-tuned SigLIP + `MatchHead`.  
Without a checkpoint: native cosine similarity (weak on local L2 edits).

## Judge B — structural

OCR (EasyOCR / Tesseract) vs caption:

1. informative word-set Jaccard (stopwords like `figure` / `plot` removed)
2. number recall when the caption contains digits
3. `score = 0.5·Jaccard + 0.5·number_recall` (Jaccard only if no digits)

If OCR is unavailable or empty: `score_struct = 0.5`.

## Judge C — object

1. Extract noun-like terms from the caption.
2. Build prompts `"a figure showing {noun}"`.
3. `score_obj = mean(top-3 SigLIP probabilities)`.

Shares the SigLIP backbone with Judge A.

## Downstream

```text
score_sem, score_struct, score_obj + router probs
  → features.py (sem_w / struct_w / obj_w)
  → arbiter → PASS / REGENERATE
```
