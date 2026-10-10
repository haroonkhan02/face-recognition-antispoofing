"""CelebA-Spoof subset -> cached face crops at several context scales.

Each mirror row is a full scene photo (e.g. 450x600) with the face box [x1, y1, x2, y2] and a
'live' / 'spoof' class. A square crop around the box, enlarged by ``scale``, is resized to
``size`` px. Larger scales keep more surrounding context (paper edges, screen bezels, moire),
which is where many attack cues live; the ablation trains one model per scale.

Example:
    faceauth-prepare-spoof            # scales 1.0 / 1.6 / 2.4 at 128 px -> data/celeba_spoof/cache/
"""

from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import CELEBA_SPOOF_DIR

SCALES = (1.0, 1.6, 2.4)
CACHE_DIR = CELEBA_SPOOF_DIR / "cache"


def square_crop(img: Image.Image, box: list[int] | np.ndarray, scale: float, size: int
                ) -> np.ndarray:
    """Square crop centred on the box, side = scale * max(w, h), kept *inside* the photo.

    If the window would cross the image border it is shifted back in (and shrunk if it is larger
    than the image). Padding with black instead would create a shortcut: spoof photos in this
    data are more often framed so the window overflows, and a CNN happily learns
    "black border = attack", which means nothing for a real camera.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    width, height = img.size
    side = min(max(x2 - x1, y2 - y1) * scale, width, height)
    cx = min(max((x1 + x2) / 2, side / 2), width - side / 2)
    cy = min(max((y1 + y2) / 2, side / 2), height - side / 2)
    crop = img.crop((round(cx - side / 2), round(cy - side / 2),
                     round(cx + side / 2), round(cy + side / 2)))
    return np.asarray(crop.resize((size, size), Image.BILINEAR))


def _process_shard(args: tuple[str, tuple[float, ...], int]) -> dict:
    import pyarrow.parquet as pq

    path, scales, size = args
    table = pq.read_table(path)
    imgs = table.column("Filepath").to_pylist()
    boxes = table.column("Bbox").to_pylist()
    classes = table.column("Class").to_pylist()
    out = {s: [] for s in scales}
    labels, names, skipped = [], [], 0
    for im, box, cls in zip(imgs, boxes, classes, strict=True):
        if box is None or len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            skipped += 1
            continue
        pil = Image.open(io.BytesIO(im["bytes"])).convert("RGB")
        for s in scales:
            out[s].append(square_crop(pil, box, s, size))
        labels.append(1 if cls == "spoof" else 0)
        names.append(Path(im.get("path") or "").stem)
    return {"crops": {s: np.stack(v) for s, v in out.items()}, "labels": np.array(labels, np.uint8),
            "names": names, "skipped": skipped}


def cache_path(split: str, scale: float, size: int) -> Path:
    return CACHE_DIR / f"{split}_s{scale:g}_{size}px.npz"


def prepare(split: str, scales=SCALES, size: int = 128, workers: int = 6) -> dict:
    shards = sorted((CELEBA_SPOOF_DIR / split).glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No {split} shards; run `faceauth-download celeba-spoof`.")
    jobs = [(str(p), tuple(scales), size) for p in shards]
    parts = []
    with ProcessPoolExecutor(workers) as pool:
        for part in tqdm(pool.map(_process_shard, jobs), total=len(jobs), desc=split):
            parts.append(part)
    labels = np.concatenate([p["labels"] for p in parts])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for s in scales:
        np.savez(cache_path(split, s, size), x=np.concatenate([p["crops"][s] for p in parts]),
                 y=labels)
    stats = {"split": split, "shards": [p.name for p in shards], "images": int(len(labels)),
             "spoof": int(labels.sum()), "live": int((labels == 0).sum()),
             "skipped_bad_boxes": int(sum(p["skipped"] for p in parts))}
    return stats


def load_split(split: str, scale: float, size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    path = cache_path(split, scale, size)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `faceauth-prepare-spoof`.")
    d = np.load(path)
    return d["x"], d["y"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--scales", type=float, nargs="+", default=list(SCALES))
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = ap.parse_args(argv)
    path = CACHE_DIR / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for split in args.splits:
        summary[split] = prepare(split, args.scales, args.size)
        print(summary[split])
    path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
