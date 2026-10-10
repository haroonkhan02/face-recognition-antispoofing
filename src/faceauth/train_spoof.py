"""Train the face anti-spoofing (presentation-attack detection) classifier on CelebA-Spoof crops.

Binary task: 0 = live (bona fide), 1 = spoof (print, replay, mask, paper cut-out ...). An
ImageNet ResNet-18 is fine-tuned on 128 px face crops with context. The decision threshold is
*not* 0.5: it is set on the validation split at the equal-error point (APCER = BPCER) and stored
in the checkpoint, so the test split is never used to tune anything.

Examples:
    faceauth-train-spoof                       # crop scale 1.6
    faceauth-train-spoof --crop-scale 1.0      # face only, for the context ablation
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models

from .config import CONFIGS_DIR, IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR, REPORTS_DIR, SpoofConfig
from .metrics import eer
from .spoof_data import load_split

DEFAULT_CONFIG = CONFIGS_DIR / "spoof_resnet18.yaml"


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(arch: str = "resnet18", pretrained: bool = True) -> nn.Module:
    if arch == "resnet18":
        net = models.resnet18(weights="DEFAULT" if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, 2)
    elif arch == "mobilenet_v3_small":
        net = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 2)
    else:
        raise ValueError(f"Unsupported arch {arch!r}")
    return net


class Normalizer:
    """uint8 NHWC batch -> ImageNet-normalised float NCHW on the device."""

    def __init__(self, device):
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1) * 255

    def __call__(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return (x_uint8.permute(0, 3, 1, 2).float() - self.mean) / self.std


def augment(x: torch.Tensor, gen: torch.Generator | None = None, max_shift: int = 8
            ) -> torch.Tensor:
    """Per-sample flips and brightness/contrast jitter, plus one small translation per batch.

    Deliberately gentle: colour casts, blur and moire are themselves spoof cues, so strong colour
    or blur augmentation would teach the model to ignore exactly what it needs.
    """
    n = x.shape[0]
    flip = (torch.rand(n, generator=gen) < 0.5).to(x.device)
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
    dx, dy = (int(v) for v in torch.randint(-max_shift, max_shift + 1, (2,), generator=gen))
    x = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
    a = (1 + 0.15 * (torch.rand(n, 1, 1, 1, generator=gen) * 2 - 1)).to(x.device)
    b = (0.15 * (torch.rand(n, 1, 1, 1, generator=gen) * 2 - 1)).to(x.device)
    return x * a + b


@torch.no_grad()
def predict_scores(model: nn.Module, x_uint8: np.ndarray, device, batch: int = 512) -> np.ndarray:
    """P(spoof) for every crop."""
    model.eval()
    norm = Normalizer(device)
    out = []
    for s in range(0, len(x_uint8), batch):
        xb = norm(torch.from_numpy(x_uint8[s:s + batch]).to(device))
        out.append(torch.softmax(model(xb).float(), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def run_name(cfg: SpoofConfig) -> str:
    return f"spoof_{cfg.arch}_s{cfg.crop_scale:g}"


def train(cfg: SpoofConfig, device: str | None = None) -> dict:
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = pick_device(device)
    x_tr, y_tr = load_split("train", cfg.crop_scale, cfg.image_size)
    x_va, y_va = load_split("valid", cfg.crop_scale, cfg.image_size)
    model = build_model(cfg.arch, cfg.pretrained).to(dev)
    norm = Normalizer(dev)
    steps = len(x_tr) // cfg.batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total,
                                                pct_start=min(0.5, max(0.15, 2 / total)))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    name = run_name(cfg)
    ckpt_path = MODELS_DIR / f"{name}.pt"
    hist_path = REPORTS_DIR / "spoof" / f"history_{name}.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    history = {"config": cfg.to_dict(), "device": str(dev), "train_images": int(len(y_tr)),
               "train_spoof_share": float(y_tr.mean()), "val_images": int(len(y_va)),
               "epochs": []}
    best, t_start = float("inf"), time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        order = rng.permutation(len(y_tr))
        for k in range(steps):
            idx = np.sort(order[k * cfg.batch_size:(k + 1) * cfg.batch_size])
            xb = augment(norm(torch.from_numpy(x_tr[idx]).to(dev)), gen)
            loss = criterion(model(xb), torch.from_numpy(y_tr[idx].astype(np.int64)).to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        scores = predict_scores(model, x_va, dev)
        val_eer, val_thr = eer(scores[y_va == 1], scores[y_va == 0])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_eer": val_eer,
               "val_threshold": val_thr, "seconds": round(time.time() - t0, 1)}
        history["epochs"].append(row)
        hist_path.write_text(json.dumps(history, indent=1))
        print(f"[{name}] epoch {epoch}/{cfg.epochs}  loss {row['train_loss']:.4f}  "
              f"val EER {val_eer:.4f}  ({row['seconds']} s)", flush=True)
        if val_eer < best:
            best = val_eer
            torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                        "epoch": epoch, "val_eer": val_eer, "threshold": val_thr}, ckpt_path)
    history.update(best_val_eer=best, train_minutes=round((time.time() - t_start) / 60, 2),
                   checkpoint=ckpt_path.name)
    hist_path.write_text(json.dumps(history, indent=1))
    return history


def load_spoof_model(path: Path, device="cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--crop-scale", type=float)
    ap.add_argument("--arch")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    cfg = SpoofConfig.load(args.config).override(crop_scale=args.crop_scale, arch=args.arch,
                                                 epochs=args.epochs)
    hist = train(cfg, args.device)
    print(f"best val EER {hist['best_val_eer']:.4f} in {hist['train_minutes']} min "
          f"-> models/{hist['checkpoint']}")


if __name__ == "__main__":
    main()
