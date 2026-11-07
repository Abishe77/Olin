
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from FM1.dataset_generation_forward_model import (
    build_image_model,
    make_lens_kwargs,
    make_source_kwargs
)




ROOT = Path(r"D:\Olin")

REAL_DATA_DIR = ROOT / "real_data" / "HE0435-1223"

PREDICTION_PATH = REAL_DATA_DIR / "exploratory_inference.json"

OBSERVED_IMAGE_PATH = REAL_DATA_DIR / "original_HE0435H.gif"

OUTPUT_IMAGE_PATH = REAL_DATA_DIR / "reconstructed_HE0435.png"

COMPARISON_PATH = REAL_DATA_DIR / "reconstruction_comparison.png"




def reconstruct_image(theta_E, e1, e2, gamma1, gamma2):

    image_model = build_image_model()

    kwargs_lens = make_lens_kwargs(
        theta_E=theta_E,
        e1=e1,
        e2=e2,
        gamma1=gamma1,
        gamma2=gamma2
    )

    # Source parameters remain at FM1 defaults.
    kwargs_source = make_source_kwargs()

    reconstructed_image = image_model.image(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source
    )

    return reconstructed_image




if __name__ == "__main__":

    # Load CNN predictions
    with open(PREDICTION_PATH, "r") as file:
       results = json.load(file)

    predictions = results["parameters"]

    theta_E = predictions["theta_E"]["prediction"]
    e1 = predictions["e1"]["prediction"]
    e2 = predictions["e2"]["prediction"]
    gamma1 = predictions["gamma1"]["prediction"]
    gamma2 = predictions["gamma2"]["prediction"]

    # Generate reconstruction
    reconstructed_image = reconstruct_image(
        theta_E,
        e1,
        e2,
        gamma1,
        gamma2
    )

    # Basic validation
    print("\nReconstruction checks:")
    print("Shape:", reconstructed_image.shape)
    print("Minimum:", reconstructed_image.min())
    print("Maximum:", reconstructed_image.max())
    print("Contains NaN:", np.isnan(reconstructed_image).any())
    print("Contains Inf:", np.isinf(reconstructed_image).any())

    # Save reconstructed image as an array
    np.save(
        REAL_DATA_DIR / "reconstructed_HE0435.npy",
        reconstructed_image
    )

    # Save a display version
    plt.imsave(
        OUTPUT_IMAGE_PATH,
        reconstructed_image,
        origin="lower",
        cmap="magma"
    )

    # Load original observation for visual comparison
    observed_image = np.asarray(
        Image.open(OBSERVED_IMAGE_PATH).convert("L"),
        dtype=np.float32
    )

    # Resize only for side-by-side display
    observed_display = Image.fromarray(
        observed_image.astype(np.uint8)
    ).resize(
        reconstructed_image.shape[::-1],
        Image.Resampling.LANCZOS
    )

    observed_display = np.asarray(observed_display)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(
        observed_display,
        origin="lower",
        cmap="gray"
    )
    axes[0].set_title("Observed HE 0435-1223")

    axes[1].imshow(
        reconstructed_image,
        origin="lower",
        cmap="magma"
    )
    axes[1].set_title("FM2 Reconstruction")

    for ax in axes:
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

    plt.tight_layout()
    plt.savefig(COMPARISON_PATH, dpi=200)
    plt.show()

    print("\nFM2 reconstruction completed.")
    print("Saved:", OUTPUT_IMAGE_PATH)
    print("Saved:", COMPARISON_PATH)
