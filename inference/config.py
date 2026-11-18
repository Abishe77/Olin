
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_VERSION = "v3_resnet18"

# Private Hugging Face model repository
HF_MODEL_REPO = "nagaabishe/olin-v3-resnet18"
HF_MODEL_REVISION = "6c9ce075d956ee9fe1ef6ce4092d09f992463c2b"

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
