"""Shared match head for fine-tuned Judge A."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def as_embed(x):
    if torch.is_tensor(x):
        return x
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state[:, 0]
    raise TypeError(f"Unexpected feature type: {type(x)}")


def resize_text_position_embeddings(model: nn.Module, max_length: int) -> None:
    """Interpolate SigLIP text positions when testing longer scientific context."""
    text_model = getattr(model, "text_model", None)
    embeddings = getattr(text_model, "embeddings", None)
    position_embedding = getattr(embeddings, "position_embedding", None)
    if position_embedding is None:
        if max_length > 64:
            raise ValueError("Model does not expose resizable text position embeddings")
        return
    old_length = int(position_embedding.num_embeddings)
    if max_length <= old_length:
        return
    weight = position_embedding.weight.detach().T.unsqueeze(0)
    resized_weight = F.interpolate(
        weight,
        size=max_length,
        mode="linear",
        align_corners=True,
    ).squeeze(0).T
    resized = nn.Embedding(
        max_length,
        position_embedding.embedding_dim,
        device=position_embedding.weight.device,
        dtype=position_embedding.weight.dtype,
    )
    with torch.no_grad():
        resized.weight.copy_(resized_weight)
    embeddings.position_embedding = resized
    embeddings.position_ids = torch.arange(max_length).expand((1, -1))
    if getattr(text_model, "config", None) is not None:
        text_model.config.max_position_embeddings = max_length
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    if text_config is not None:
        text_config.max_position_embeddings = max_length


class MatchHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)
        x = torch.cat([img, txt, img * txt, (img - txt).abs()], dim=-1)
        return self.net(x).squeeze(-1)
