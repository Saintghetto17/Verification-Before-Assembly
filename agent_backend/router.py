"""ConvNeXt-Tiny image-type router: byt / video / graph / scheme."""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

logger = logging.getLogger(__name__)

CLASS_TO_ID = {"byt": 0, "video": 1, "graph": 2, "scheme": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


def get_router_transform(image_size: int = 224, train: bool = False):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class FolderImageDataset(Dataset):
    def __init__(self, root: Path, classes: Sequence[str], transform):
        self.samples: List[Tuple[Path, int]] = []
        self.transform = transform
        for c in classes:
            d = root / c
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*")):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                    self.samples.append((p, CLASS_TO_ID[c]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_router_model(num_classes: int = 4) -> nn.Module:
    try:
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    except Exception:
        model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def _write_synth_photo(dst: Path, np_rng) -> None:
    import numpy as np

    arr = (np_rng.random((224, 224, 3)) * 255).astype("uint8")
    for _ in range(5):
        cy, cx = np_rng.integers(20, 200, size=2)
        rr = np_rng.integers(10, 40)
        yy, xx = np.ogrid[:224, :224]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= rr**2
        arr[mask] = np_rng.integers(0, 255, size=3)
    Image.fromarray(arr).save(dst)


def _copy_or_synth(src: Optional[Path], dst: Path, np_rng) -> None:
    if src is not None and src.is_file():
        shutil.copy2(src, dst)
    else:
        _write_synth_photo(dst, np_rng)


def generate_router_data(
    out_dir: Path,
    *,
    classes: Sequence[str],
    n_train: int,
    n_val: int,
    n_video_train: Optional[int] = None,
    n_video_val: Optional[int] = None,
    seed: int = 42,
    figures_dir: Path | None = None,
) -> Dict[str, Any]:
    """Synthesize graph/scheme; byt/video from real figures (video gets more)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    n_video_train = n_video_train if n_video_train is not None else max(n_train * 4, n_train)
    n_video_val = n_video_val if n_video_val is not None else max(n_val * 4, n_val)

    real_pool: List[Path] = []
    if figures_dir and figures_dir.is_dir():
        for p in figures_dir.rglob("*.png"):
            if "golden" in p.parts:
                continue
            real_pool.append(p)
    rng.shuffle(real_pool)
    logger.info(
        "Router real figure pool: %d (figures_dir=%s)",
        len(real_pool),
        figures_dir,
    )

    counts: Dict[str, Any] = {"real_pool": len(real_pool)}
    cursor = 0

    for split, n_base, n_video in (
        ("train", n_train, n_video_train),
        ("val", n_val, n_video_val),
    ):
        for cls in classes:
            d = out_dir / split / cls
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        for i in range(n_base):
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            kind = rng.choice(["bar", "line", "scatter", "hist"])
            x = np.arange(6)
            y = np_rng.uniform(0.5, 5.0, size=6)
            if kind == "bar":
                ax.bar(x, y, color=plt.cm.viridis(np_rng.random()))
            elif kind == "line":
                ax.plot(x, y, marker="o")
            elif kind == "scatter":
                ax.scatter(np_rng.random(40), np_rng.random(40), c=np_rng.random(40))
            else:
                ax.hist(np_rng.normal(0, 1, 200), bins=12)
            ax.set_title(f"{kind} {i}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / split / "graph" / f"graph_{i:04d}.png")
            plt.close(fig)

        for i in range(n_base):
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            g = nx.barabasi_albert_graph(rng.randint(6, 14), 2, seed=seed + i)
            pos = nx.spring_layout(g, seed=seed + i)
            nx.draw(
                g,
                pos,
                ax=ax,
                node_size=200,
                node_color="#4C72B0",
                edge_color="#555555",
                with_labels=False,
            )
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(out_dir / split / "scheme" / f"scheme_{i:04d}.png")
            plt.close(fig)

        for i in range(n_base):
            src = real_pool[cursor] if cursor < len(real_pool) else None
            cursor += 1
            _copy_or_synth(src, out_dir / split / "byt" / f"byt_{i:04d}.png", np_rng)

        for i in range(n_video):
            src = real_pool[cursor] if cursor < len(real_pool) else None
            cursor += 1
            _copy_or_synth(src, out_dir / split / "video" / f"video_{i:04d}.png", np_rng)

        counts[split] = {
            "byt": n_base,
            "video": n_video,
            "graph": n_base,
            "scheme": n_base,
        }

    counts["real_used"] = min(cursor, len(real_pool))
    logger.info("Generated router data under %s: %s", out_dir, counts)
    return counts


def train_router(
    data_dir: Path,
    ckpt_path: Path,
    *,
    classes: Sequence[str],
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 1e-4,
    image_size: int = 224,
    device: str = "cpu",
    project_root: Path | None = None,
    tb_run_name: str = "agent_router",
) -> Dict[str, float]:
    device_t = torch.device(device)
    train_ds = FolderImageDataset(
        data_dir / "train", classes, get_router_transform(image_size, train=True)
    )
    val_ds = FolderImageDataset(
        data_dir / "val", classes, get_router_transform(image_size, train=False)
    )
    if len(train_ds) == 0:
        raise RuntimeError(f"No router training images in {data_dir / 'train'}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # inverse-frequency weights (video is oversampled from real figures)
    counts = torch.zeros(len(classes), dtype=torch.float)
    for _, y in train_ds.samples:
        counts[y] += 1.0
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean()
    logger.info("router class counts=%s weights=%s", counts.tolist(), weights.tolist())

    model = build_router_model(len(classes)).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=weights.to(device_t))

    writer = None
    tb_dir: Path | None = None
    if project_root is not None:
        from .tb_logging import make_summary_writer

        writer, tb_dir = make_summary_writer(project_root, tb_run_name)
        if writer is not None:
            writer.add_text(
                "hparams",
                f"epochs={epochs} batch={batch_size} lr={lr} device={device} "
                f"classes={list(classes)} counts={counts.tolist()}",
            )

    best_acc = -1.0
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_correct = 0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device_t), yb.to(device_t)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(yb)
            train_correct += int((logits.argmax(dim=-1) == yb).sum().item())
            n += len(yb)

        train_loss = total_loss / max(n, 1)
        train_acc = train_correct / max(n, 1)

        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device_t), yb.to(device_t)
                logits = model(xb)
                val_loss_sum += float(crit(logits, yb).item()) * len(yb)
                correct += int((logits.argmax(dim=-1) == yb).sum().item())
                total += len(yb)
        val_loss = val_loss_sum / max(total, 1)
        acc = correct / max(total, 1)
        logger.info(
            "router epoch %d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            acc,
        )
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/acc", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc", acc, epoch)
            writer.add_scalar("train/lr", lr, epoch)
            writer.flush()

        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {"state_dict": model.state_dict(), "classes": list(classes), "acc": acc},
                ckpt_path,
            )

    if writer is not None:
        writer.add_scalar("val/best_acc", best_acc, epochs - 1)
        writer.close()

    out: Dict[str, float] = {"best_val_acc": float(best_acc), "ckpt": str(ckpt_path)}
    if tb_dir is not None:
        out["tfevents"] = str(tb_dir)
    return out


class ImageRouter:
    def __init__(
        self,
        ckpt_path: Path | None,
        *,
        classes: Sequence[str],
        device: str = "cpu",
        image_size: int = 224,
    ):
        self.classes = list(classes)
        self.device = torch.device(device)
        self.transform = get_router_transform(image_size, train=False)
        self.model = build_router_model(len(self.classes))
        self._uniform = True
        if ckpt_path and Path(ckpt_path).is_file():
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(blob["state_dict"])
            self._uniform = False
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_proba(self, image_path: str | Path) -> List[float]:
        if self._uniform:
            n = len(self.classes)
            return [1.0 / n] * n
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        return [float(p) for p in probs]
