import csv
import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

#load original labels
labels_file = ROOT / "datasets" / "dataset_v0" / "labels.csv"

samples =[]

with open(labels_file, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        samples.append((
            row["image_id"],
            float(row["theta_E"])
        ))


# Reproduce the original test split
rng = np.random.default_rng(42)
rng.shuffle(samples)

test_samples = samples[9000:]

image_ids = np.array([sample[0] for sample in test_samples])
true_theta = np.array([sample[1] for sample in test_samples])


# Define five equal-width bins
bin_edges = np.linspace(
    true_theta.min(),
    true_theta.max(),
    6
)


# Select six samples from each bin
selection_rng = np.random.default_rng(42)

selected_indices = []

for i in range(5):

    if i == 4:
        indices = np.where(
            (true_theta >= bin_edges[i]) &
            (true_theta <= bin_edges[i + 1])
        )[0]

    else:
        indices = np.where(
            (true_theta >= bin_edges[i]) &
            (true_theta < bin_edges[i + 1])
        )[0]

    selected = selection_rng.choice(
        indices,
        size=6,
        replace=False
    )

    selected_indices.extend(selected)

# Save selected validation samples
output_file = ROOT / "FM2" / "validation_samples.csv"

with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(["image_id", "true_theta_E"])

    for index in selected_indices:
        writer.writerow([
            image_ids[index],
            true_theta[index]
        ])

print(f"\nSaved validation samples to: {output_file}")


# Display selected samples
print("Selected validation samples:\n")

for index in selected_indices:

    print(
        f"Image ID: {image_ids[index]} | "
        f"True theta_E: {true_theta[index]:.4f}"
    )

print("\nTotal selected:", len(selected_indices))
