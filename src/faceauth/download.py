"""Download the pretrained face models, LFW and a CelebA-Spoof subset. All open, no login.

Examples:
    faceauth-download models        # YuNet, SFace, ArcFace (buffalo_sc) -> models/  (~54 MB)
    faceauth-download lfw           # 173 MB tgz -> data/lfw/  + pairs.txt
    faceauth-download celeba-spoof  # 11 parquet shards (~4.8 GB) -> data/celeba_spoof/
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from .config import (
    ARCFACE_FILE,
    BUFFALO_SC,
    CELEBA_SPOOF_DIR,
    CELEBA_SPOOF_HF,
    CELEBA_SPOOF_SHARDS,
    LFW_DIR,
    LFW_PAIRS,
    LFW_TGZ,
    MODELS_DIR,
    SFACE,
    YUNET,
)

_UA = {"User-Agent": "Mozilla/5.0 (faceauth downloader)"}  # figshare rejects the urllib default


def fetch(url: str, dest: Path, expected_bytes: int | None = None, chunk: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dest``, resuming a partial file with an HTTP Range request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if expected_bytes and have == expected_bytes:
        return dest
    headers = dict(_UA, **({"Range": f"bytes={have}-"} if have else {}))
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp:
        if have and resp.status != 206:
            have = 0
        total = expected_bytes or (have + int(resp.headers.get("Content-Length", 0)))
        with open(dest, "ab" if have else "wb") as f, tqdm(
                total=total, initial=have, unit="B", unit_scale=True, desc=dest.name) as bar:
            while block := resp.read(chunk):
                f.write(block)
                bar.update(len(block))
    if expected_bytes and dest.stat().st_size != expected_bytes:
        raise OSError(f"{dest.name}: incomplete download, re-run to resume")
    return dest


def download_models() -> None:
    for url, name, size in (YUNET, SFACE):
        fetch(url, MODELS_DIR / name, size)
    if not (MODELS_DIR / ARCFACE_FILE).exists():
        zpath = fetch(BUFFALO_SC[0], MODELS_DIR / BUFFALO_SC[1], BUFFALO_SC[2])
        with zipfile.ZipFile(zpath) as zf:
            member = next(m for m in zf.namelist() if m.endswith(ARCFACE_FILE))
            (MODELS_DIR / ARCFACE_FILE).write_bytes(zf.read(member))
        zpath.unlink()
    print(f"models ready in {MODELS_DIR}")


def download_lfw() -> None:
    if not (LFW_DIR / "images").exists():
        tgz = fetch(LFW_TGZ[0], LFW_DIR / LFW_TGZ[1])
        with tarfile.open(tgz) as tf:
            tf.extractall(LFW_DIR, filter="data")
        (LFW_DIR / "lfw").rename(LFW_DIR / "images")
        tgz.unlink()
    fetch(LFW_PAIRS[0], LFW_DIR / LFW_PAIRS[1])
    people = sum(1 for p in (LFW_DIR / "images").iterdir() if p.is_dir())
    print(f"LFW ready: {people} people in {LFW_DIR / 'images'}")


def shard_names(split: str) -> list[tuple[str, int]]:
    """Sorted (file name, size) of the mirror's parquet shards for one split."""
    url = "https://huggingface.co/api/datasets/Ar4ikov/celebA_spoof/tree/main/data"
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=60) as r:
        files = json.load(r)
    out = [(Path(f["path"]).name, f["size"]) for f in files
           if Path(f["path"]).name.startswith(f"{split}-")]
    return sorted(out)


def download_celeba_spoof(shards: dict | None = None) -> None:
    for split, (count, indices) in (shards or CELEBA_SPOOF_SHARDS).items():
        names = shard_names(split)
        if len(names) != count:
            raise RuntimeError(f"expected {count} {split} shards, found {len(names)}")
        for i in indices:
            name, size = names[i]
            fetch(f"{CELEBA_SPOOF_HF}/{name}", CELEBA_SPOOF_DIR / split / name, size)
    print(f"CelebA-Spoof shards ready in {CELEBA_SPOOF_DIR}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["models", "lfw", "celeba-spoof", "all"])
    args = ap.parse_args(argv)
    if args.what in ("models", "all"):
        download_models()
    if args.what in ("lfw", "all"):
        download_lfw()
    if args.what in ("celeba-spoof", "all"):
        download_celeba_spoof()


if __name__ == "__main__":
    main()
