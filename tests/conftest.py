"""Synthetic fixtures: no dataset, pretrained model, camera or GPU needed."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


def jpeg_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def spoof_shard(tmp_path):
    """A parquet shard in the CelebA-Spoof mirror's schema (image bytes, box, class)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(0)
    rows = {"Filepath": [], "Bbox": [], "Class": []}
    for i in range(6):
        img = rng.integers(0, 255, (120, 90, 3), dtype=np.uint8)
        rows["Filepath"].append({"bytes": jpeg_bytes(img), "path": f"{i:06d}.jpg"})
        rows["Bbox"].append([20, 30, 60, 80] if i != 5 else [50, 50, 40, 40])  # last one invalid
        rows["Class"].append("spoof" if i % 2 else "live")
    path = tmp_path / "train" / "train-00000-of-00001.parquet"
    path.parent.mkdir()
    pq.write_table(pa.table(rows), path)
    return path
