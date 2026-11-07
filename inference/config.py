
# inference/config.py

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_METADATA_PATH = (
    ROOT / "datasets" / "dataset_v1" / "preprocessing_metadata.json"
)

MODEL_PATH = (
    ROOT / "training" / "models" / "v3_resnet18.keras"
)

TARGET_NORMALIZATION_PATH = (
    ROOT / "training" / "results" / "v3_resnet18"
    / "target_normalization.json"
)

UNCERTAINTY_CALIBRATION_PATH = (
    ROOT / "training" / "results" / "v3_resnet18"
    / "uncertainty" / "calibration_factors.json"
)

IMAGE_SIZE = (160, 160)
IMAGE_CHANNELS = 1
MC_PASSES = 30
