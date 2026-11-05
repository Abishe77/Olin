import numpy as np

import csv

from pathlib import Path
from FM2.reverse_reconstruction_model import reconstruct_image


ROOT = Path(__file__).resolve().parent.parent

# Load validation dataset
data = np.load(ROOT / "FM2" / "validation_data.npz")

image_ids = data["image_ids"]
observed_images = data["observed_images"]
true_theta = data["true_theta_E"]
predicted_theta = data["predicted_theta_E"]

# Calculate pixel residual metrics
def calculate_metrics(residual):

    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual ** 2))

    return mae, rmse


# Validate all 30 samples
results = []

for index in range(len(image_ids)):

    observed = observed_images[index]
    true_parameter = true_theta[index]
    predicted_parameter = predicted_theta[index]

    # Generate both reconstructions
    true_reconstruction = reconstruct_image(true_parameter)
    predicted_reconstruction = reconstruct_image(predicted_parameter)

    # Calculate residuals
    true_residual = observed - true_reconstruction
    predicted_residual = observed - predicted_reconstruction

    # Calculate metrics
    true_mae, true_rmse = calculate_metrics(true_residual)
    pred_mae, pred_rmse = calculate_metrics(predicted_residual)

    parameter_error = abs(predicted_parameter - true_parameter)

    results.append([
        image_ids[index],
        true_parameter,
        predicted_parameter,
        parameter_error,
        true_mae,
        pred_mae,
        true_rmse,
        pred_rmse
    ])



    print(
        f"{image_ids[index]} | "
        f"Parameter error: {parameter_error:.5f} | "
        f"GT MAE: {true_mae:.8f} | "
        f"CNN MAE: {pred_mae:.8f}"
    )
# Save individual validation results
results_dir = ROOT / "FM2" / "results"
results_dir.mkdir(parents=True, exist_ok=True)

output_file = results_dir / "validation_30_results.csv"

with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "image_id",
        "true_theta_E",
        "predicted_theta_E",
        "parameter_error",
        "gt_mae",
        "cnn_mae",
        "gt_rmse",
        "cnn_rmse"
    ])

    writer.writerows(results)


# Calculate aggregate metrics
parameter_errors = np.array([row[3] for row in results])
gt_maes = np.array([row[4] for row in results])
cnn_maes = np.array([row[5] for row in results])
gt_rmses = np.array([row[6] for row in results])
cnn_rmses = np.array([row[7] for row in results])

print("\nAggregrate validation")

print(f"Mean parameter error: {np.mean(parameter_errors):.6f} arcsec")
print(f"Median parameter error: {np.median(parameter_errors):.6f} arcsec")

print(f"\nMean GT MAE:  {np.mean(gt_maes):.8f}")
print(f"Mean CNN MAE: {np.mean(cnn_maes):.8f}")

print(f"\nMean GT RMSE:  {np.mean(gt_rmses):.8f}")
print(f"Mean CNN RMSE: {np.mean(cnn_rmses):.8f}")

print("\nResults saved to:", output_file)

print("\nCompleted:", len(results), "reconstructions")
