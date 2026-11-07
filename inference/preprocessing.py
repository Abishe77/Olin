
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from inference.config import (
    DATASET_METADATA_PATH,
    IMAGE_SIZE
)


SUPPORTED_FORMATS = {".npy", ".png", ".jpg", ".jpeg", ".webp"}


def load_image(image_path):
    """Load an image and return a 160x160 grayscale array."""

    image_path = Path(image_path)
    suffix = image_path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {suffix}. "
            "Use PNG, JPG, JPEG, WebP, or NPY."
        )

    if suffix == ".npy":
        image = np.load(image_path).astype(np.float32)

        if image.shape != IMAGE_SIZE:
            raise ValueError(
                f"Expected image shape {IMAGE_SIZE}, got {image.shape}"
            )

        return image, "synthetic"

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        image = image.convert("L")
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )

        image = np.asarray(image, dtype=np.float32) / 255.0

    return image, "exploratory"


def preprocess_image(image_path):
    """Prepare an image for the ResNet-18 model."""

    image, input_type = load_image(image_path)

    if input_type == "synthetic":
        import json

        with open(DATASET_METADATA_PATH, "r") as file:
            metadata = json.load(file)

        mean = metadata["normalization"]["mean"]
        std = metadata["normalization"]["std"]

        image = (image - mean) / std

    else:
        # Exploratory path for ordinary display images.
        # This is not equivalent to calibrated scientific preprocessing.
        mean = float(np.mean(image))
        std = float(np.std(image))

        if std < 1e-8:
            raise ValueError("Image has insufficient intensity variation.")

        image = (image - mean) / std

    image = image[np.newaxis, ..., np.newaxis]

    return image.astype(np.float32)


