"""Paths, model / dataset sources and the YAML-backed anti-spoofing training config."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CONFIGS_DIR = ROOT / "configs"
OUTPUTS_DIR = ROOT / "outputs"
GALLERY_DIR = DATA_DIR / "gallery"  # enrolled identities (embeddings only, never committed)

# Pretrained models: (url, file name, bytes).
# YuNet face detector and SFace recogniser from OpenCV Zoo (MIT / Apache-2.0).
# ArcFace MobileFaceNet (w600k_mbf) from InsightFace's buffalo_sc pack: non-commercial research
# use only, which is why it is downloaded at run time and never redistributed.
YUNET = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
         "face_detection_yunet_2023mar.onnx", "face_detection_yunet_2023mar.onnx", 232_589)
SFACE = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
         "face_recognition_sface_2021dec.onnx", "face_recognition_sface_2021dec.onnx", 38_696_353)
BUFFALO_SC = ("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
              "buffalo_sc.zip", 14_969_382)
ARCFACE_FILE = "w600k_mbf.onnx"  # inside buffalo_sc.zip

# LFW (Huang et al., 2007): 13,233 web photos of 5,749 people; 6,000 verification pairs in 10
# folds. Mirror used by scikit-learn (the original UMass server is often offline).
LFW_TGZ = ("https://ndownloader.figshare.com/files/5976018", "lfw.tgz", None)
LFW_PAIRS = ("https://ndownloader.figshare.com/files/5976006", "pairs.txt", None)
LFW_DIR = DATA_DIR / "lfw"

# CelebA-Spoof (Zhang et al., ECCV 2020): live / spoof photos of 10,177 CelebA identities with
# print, replay, mask and paper-cut attacks. Hugging Face mirror of the authors' release as
# parquet shards (image bytes, face box [x1, y1, x2, y2], class). Non-commercial research use.
CELEBA_SPOOF_HF = "https://huggingface.co/datasets/Ar4ikov/celebA_spoof/resolve/main/data"
CELEBA_SPOOF_DIR = DATA_DIR / "celeba_spoof"
CELEBA_SPOOF_SHARDS = {  # subset that fits a laptop: shard indices per split
    "train": (93, (0, 1, 2, 3)),
    "valid": (9, (0,)),
    "test": (65, (0, 1, 2, 3, 4, 5)),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ArcFace's canonical 5-point template for a 112x112 crop (eyes, nose tip, mouth corners).
ARCFACE_TEMPLATE = (
    (38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366),
    (41.5493, 92.3655), (70.7299, 92.2041),
)


@dataclass
class SpoofConfig:
    """Anti-spoofing classifier run; saved into the checkpoint and the history."""

    arch: str = "resnet18"
    pretrained: bool = True
    crop_scale: float = 1.6  # face box enlarged by this factor: spoof cues sit around the face
    image_size: int = 128
    epochs: int = 8
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    seed: int = 0

    @classmethod
    def load(cls, path: str | Path | None = None) -> SpoofConfig:
        if path is None or not Path(path).exists():
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def override(self, **kwargs) -> SpoofConfig:
        data = self.to_dict()
        data.update({k: v for k, v in kwargs.items() if v is not None})
        return SpoofConfig(**data)
