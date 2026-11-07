import numpy as np

from inference.config import MC_PASSES
from inference.model_loader import load_inference_assets


# Load model and supporting assets once.
ASSETS = load_inference_assets()

MODEL = ASSETS["model"]
TARGET_MEAN = ASSETS["target_mean"]
TARGET_STD = ASSETS["target_std"]
CALIBRATION_FACTORS = ASSETS["calibration_factors"]
TARGET_COLUMNS = ASSETS["target_columns"]


def predict(image):
    """
    Predict gravitational lens parameters using MC Dropout.

    Parameters
    ----------
    image : numpy.ndarray
        Preprocessed image with shape (1, 160, 160, 1).

    Returns
    -------
    dict
        Physical parameter predictions and calibrated uncertainties.
    """

    image = np.asarray(image, dtype=np.float32)

    if image.shape != (1, 160, 160, 1):
        raise ValueError(
            f"Expected image shape (1, 160, 160, 1), "
            f"received {image.shape}"
        )

    if not np.all(np.isfinite(image)):
        raise ValueError("Input image contains invalid values.")

    predictions = []

    # Run repeated stochastic forward passes.
    for _ in range(MC_PASSES):
        prediction = MODEL(image, training=False).numpy()
        predictions.append(prediction[0])

    predictions = np.asarray(predictions)

    # Calculate mean and standard deviation in normalized space.
    mean_scaled = np.mean(predictions, axis=0)
    std_scaled = np.std(predictions, axis=0, ddof=1)

    # Convert predictions and uncertainties to physical units.
    mean_physical = (
        mean_scaled * TARGET_STD + TARGET_MEAN
    )

    std_physical = std_scaled * TARGET_STD

    # Apply calibration factors.
    calibrated_uncertainty = (
        std_physical * CALIBRATION_FACTORS
    )

    # Organize results.
    results = {}

    for index, parameter in enumerate(TARGET_COLUMNS):

        results[parameter] = {
            "prediction": float(mean_physical[index]),
            "raw_uncertainty": float(std_physical[index]),
            "calibrated_uncertainty": float(
                calibrated_uncertainty[index]
            )
        }

    return {
        "model": "Olin ResNet-18",
        "mc_passes": MC_PASSES,
        "parameters": results
    }