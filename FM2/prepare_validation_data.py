import csv
import numpy as np

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# Load selected validation samples
samples = []

with open(ROOT / "FM2" / "validation_samples.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        samples.append((
            row["image_id"],
            float(row["true_theta_E"])
        ))


# Load the complete test predictions
predictions = np.load(
    ROOT / "training" / "results" / "cnn_1" / "predictions.npy"
)

# Load the complete test labels
test_labels = np.load(ROOT / "datasets" / "y_test_0.npy")

# Reproduce the original test split ordering
labels_file = ROOT / "datasets" / "dataset_v0" / "labels.csv"

all_samples = []

with open(labels_file, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        all_samples.append((
            row["image_id"],
            float(row["theta_E"])
        ))

rng = np.random.default_rng(42)
rng.shuffle(all_samples)

test_samples = all_samples[9000:]

# Map image IDs to their test-array indices
test_id_to_index = {
    image_id: index
    for index, (image_id, _) in enumerate(test_samples)
}


# Verify selected samples and retrieve predictions
print("\nSelected sample mappings:\n")

for image_id, true_theta in samples:

    index = test_id_to_index[image_id]

    predicted_theta = predictions[index]
    saved_test_theta = test_labels[index]

    print(
        f"ID: {image_id} | "
        f"Index: {index} | "
        f"True: {true_theta:.4f} | "
        f"Test label: {saved_test_theta:.4f} | "
        f"Predicted: {predicted_theta:.4f}"
    )


# Load original, unnormalized observations
image_directory = ROOT / "datasets" / "dataset_v0" / "images"

observed_images = []

for image_id, _ in samples:

    image_path = image_directory / f"{image_id}.npy"
    image = np.load(image_path)

    observed_images.append(image)

observed_images = np.array(observed_images)

print("\nObserved images shape:", observed_images.shape)
print("Image dtype:", observed_images.dtype)

print("\nPredictions shape:", predictions.shape)
print("Test labels shape:", test_labels.shape)
print("Selected samples:", len(samples))


# Extract predictions and labels for selected samples
selected_predictions = []
selected_true_theta = []

for image_id, true_theta in samples:

    index = test_id_to_index[image_id]

    selected_predictions.append(predictions[index])
    selected_true_theta.append(true_theta)


selected_predictions = np.array(selected_predictions)
selected_true_theta = np.array(selected_true_theta)


# Save validation dataset
output_file = ROOT / "FM2" / "validation_data.npz"

np.savez(
    output_file,
    image_ids=np.array([sample[0] for sample in samples]),
    observed_images=observed_images,
    true_theta_E=selected_true_theta,
    predicted_theta_E=selected_predictions
)

print("\nValidation dataset saved:", output_file)
print("Saved image shape:", observed_images.shape)
print("Saved prediction shape:", selected_predictions.shape)