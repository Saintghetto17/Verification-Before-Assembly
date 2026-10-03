"""Load fine-tuned SmolLM graph extractor (LoRA adapter on SmolLM2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .schema import parse_graph_json
from .sft import build_messages


class GraphGenerator:
    def __init__(
        self,
        adapter_dir: str | Path,
        *,
        base: Optional[str | Path] = None,
        device: str = "cuda",
        max_new_tokens: int = 512,
    ):
        adapter_dir = Path(adapter_dir)
        import json

        base_name = str(base) if base else None
        for meta_path in (
            adapter_dir / "adapter_config.json",
            adapter_dir / "train_metrics.json",
        ):
            if base_name or not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            base_name = meta.get("base_model_name_or_path") or meta.get("base")
        if not base_name:
            raise ValueError(f"cannot resolve base model for adapter {adapter_dir}")

        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(str(base_name), torch_dtype=dtype)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
        self.model = model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate_text(self, claim: str) -> str:
        prompt = self.tokenizer.apply_chat_template(
            build_messages(claim),
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(
            out[0, enc["input_ids"].size(1) :],
            skip_special_tokens=True,
        )

    def generate(self, claim: str, claim_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        text = self.generate_text(claim)
        graph = parse_graph_json(text)
        if graph is not None and claim_id is not None:
            graph["claim_id"] = int(claim_id)
        return graph
