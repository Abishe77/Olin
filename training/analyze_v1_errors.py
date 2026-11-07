
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
datasets_dir = Path("../datasets")
labels_file = datasets_dir / "dataset_v1" / "labels.csv"
results_dir = Path("results/v1_cnn_1")
output_dir = results_dir / "diagnostics" / "parameter_errors"
output_dir.mkdir(parents=True, exist_ok=True)

# Load original labels
with open(labels_file, "r", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Reconstruct the exact V1 train/validation/test split
samples = [
    (int(row["image_id"]), float(row["theta_E"]))
    for row in rows
]

rng = np.random.default_rng(42)
rng.shuffle(samples)

test_samples = samples[9000:]

test_ids = np.array([sample[0] for sample in test_samples])
test_theta = np.array([sample[1] for sample in test_samples])

# Match test image IDs to their complete physical labels
labels_by_id = {
    int(row["image_id"]): row
    for row in rows
}

test_labels = [labels_by_id[int(image_id)] for image_id in test_ids]

# Load prepared test data
x_test = np.load(datasets_dir / "x_test_1.npy")
y_test = np.load(datasets_dir / "y_test_1.npy")

# Verify that reconstructed labels match preprocessing
if not np.allclose(y_test, test_theta, rtol=1e-6, atol=1e-7):
    raise ValueError("Test labels do not match reconstructed split!")

# Load V1 model and generate predictions
model = tf.keras.models.load_model("models/v1_cnn_1.keras")

x_test = x_test[..., np.newaxis]
predictions = model.predict(x_test, verbose=0).flatten()

# Calculate errors
residuals = predictions - y_test
absolute_errors = np.abs(residuals)

print("Test samples:", len(test_ids))
print("Test MAE:", np.mean(absolute_errors))
print("Test MSE:", np.mean(residuals ** 2))

# Extract physical parameters
def get_parameter(name):
    return np.array(
        [float(row[name]) for row in test_labels]
    )

ellipticity = np.sqrt(
    get_parameter("e1")**2 + get_parameter("e2")**2
)

shear = np.sqrt(
    get_parameter("gamma1")**2 + get_parameter("gamma2")**2
)

parameters = {
    "theta_E": y_test,
    "ellipticity_magnitude": ellipticity,
    "shear_magnitude": shear,
    "center_x": get_parameter("center_x"),
    "center_y": get_parameter("center_y"),
    "R_sersic": get_parameter("R_sersic"),
    "n_sersic": get_parameter("n_sersic"),
    "edge_flux_fraction": get_parameter("edge_flux_fraction")
}

# Save per-image prediction results
output_csv = output_dir / "v1_error_analysis.csv"

with open(output_csv, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "image_id", "theta_E", "prediction",
        "residual", "absolute_error"
    ])

    for i in range(len(test_ids)):
        writer.writerow([
            test_ids[i],
            y_test[i],
            predictions[i],
            residuals[i],
            absolute_errors[i]
        ])

# Analyse errors against each physical parameter
print("\nPHYSICAL PARAMETER ERROR ANALYSIS")

for name, values in parameters.items():

    correlation = np.corrcoef(values, absolute_errors)[0, 1]

    print(f"{name:25s} Pearson correlation: {correlation:.4f}")

    plt.figure(figsize=(8, 5))

    plt.scatter(
        values,
        absolute_errors,
        alpha=0.5
    )

    plt.xlabel(name)
    plt.ylabel("Absolute prediction error (arcsec)")
    plt.title(f"V1 CNN - Error vs {name}")
    plt.grid(True)
    plt.tight_layout()

    filename = name.replace("/", "_") + ".png"
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()

print("\nAnalysis saved to:", output_dir)
