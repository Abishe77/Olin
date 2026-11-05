import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from FM2.reverse_reconstruction_model import reconstruct_image


ROOT = Path(__file__).resolve().parent.parent

# Load validation data
data = np.load(ROOT / "FM2" / "validation_data.npz")

image_ids = data["image_ids"]
observed_images = data["observed_images"]
true_theta = data["true_theta_E"]
predicted_theta = data["predicted_theta_E"]

# Load validation metrics
results = pd.read_csv(
    ROOT / "FM2" / "results" / "validation_30_results.csv"
)

# Select representative samples

# Best prediction: smallest parameter error
best_sample = results["parameter_error"].idxmin()

# Worst prediction: largest parameter error
worst_sample = results["parameter_error"].idxmax()

# Typical prediction: error closest to the median
median_error = results["parameter_error"].median()

typical_sample = (
    results["parameter_error"] - median_error
).abs().idxmin()


selected_indices = [
    best_sample,
    typical_sample,
    worst_sample
]

print("\nSelected validation samples:")

for index in selected_indices:
    row = results.iloc[index]

    print(
        f"Image ID: {row['image_id']} | "
        f"True theta_E: {row['true_theta_E']:.4f} | "
        f"Predicted theta_E: {row['predicted_theta_E']:.4f} | "
        f"Parameter error: {row['parameter_error']:.5f}"
    )


# Generate visual comparisons

fig, axes = plt.subplots(
    3, 5,
    figsize=(20, 12)
)

column_titles = [
    "Original Observation",
    "GT Reconstruction",
    "CNN Reconstruction",
    "GT Residual",
    "CNN Residual"
]

intensity_limit = max(
    np.max(observed_images),
    np.max([
        np.max(reconstruct_image(theta))
        for theta in np.concatenate([true_theta, predicted_theta])
    ])
)

for col, title in enumerate(column_titles):
    axes[0, col].set_title(title, fontsize=11)


for row_index, sample_index in enumerate(selected_indices):

    # Get original data index
    data_index = sample_index
    image_id = str(image_ids[data_index])

    observed = observed_images[data_index]
    true_parameter = true_theta[data_index]
    predicted_parameter = predicted_theta[data_index]

    # Reconstruct images
    gt_reconstruction = reconstruct_image(true_parameter)
    cnn_reconstruction = reconstruct_image(predicted_parameter)

    # Calculate residuals
    gt_residual = observed - gt_reconstruction
    cnn_residual = observed - cnn_reconstruction

    images = [
        observed,
        gt_reconstruction,
        cnn_reconstruction,
        gt_residual,
        cnn_residual
    ]

    for col, image in enumerate(images):

        ax = axes[row_index, col]

        if col < 3:
            im = ax.imshow(
                image,
                origin="lower",
                cmap="magma",
                vmin =0,
                vmax = intensity_limit
            )
        else:
            limit = max(
                np.max(np.abs(gt_residual)),
                np.max(np.abs(cnn_residual))
            )

            im = ax.imshow(
                image,
                origin="lower",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit
            )

        ax.set_xticks([])
        ax.set_yticks([])

    axes[row_index, 0].set_ylabel(
        f"Sample {row_index + 1}\nID: {image_id}",
        fontsize=10
    )


fig.suptitle(
    "FM2 Reconstruction Validation: Best, Typical and Worst CNN Predictions",
    fontsize=14, y =0.98
)

plt.tight_layout(rect=[0,0,1,0.93])

# Save figure
output_path = ROOT / "FM2" / "results" / "validation_examples.png"

plt.savefig(output_path, dpi=200, bbox_inches="tight")

plt.show()

print("\nFigure saved to:", output_path)
