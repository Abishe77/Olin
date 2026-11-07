
import csv
from pathlib import Path
import numpy as np
import json


# Dataset paths
dataset_dir = Path("../datasets/dataset_v1")
labels_file = dataset_dir / "labels.csv"
image_dir = dataset_dir / "images"


# Load image IDs and target values
samples = []

with open(labels_file, "r", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:
        image_id = int(row["image_id"])
        theta_E = float(row["theta_E"])

        samples.append((image_id, theta_E))


# Shuffle and split into training, validation and testing sets
rng = np.random.default_rng(42)
rng.shuffle(samples)

train_samples = samples[:8000]
validation_samples = samples[8000:9000]
test_samples = samples[9000:]


# Load training images
x_train = []
y_train = []

for image_id, theta_E in train_samples:

    image_path = image_dir / f"{image_id:06d}.npy"
    image = np.load(image_path)

    x_train.append(image)
    y_train.append(theta_E)


x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)


# Calculate normalization statistics using training data only
mean = x_train.mean()
std = x_train.std()

if std == 0:
    raise ValueError("Training image standard deviation is zero.")


# Save preprocessing metadata separately
preprocessing_metadata = {
    "dataset_version": "dataset_v1",
    "model_version": "olin_v1",
    "split": {
        "training": 8000,
        "validation": 1000,
        "testing": 1000,
        "random_seed": 42
    },
    "normalization": {
        "method": "standardization",
        "mean": float(mean),
        "std": float(std)
    }
}

metadata_file = dataset_dir / "preprocessing_metadata.json"

with open(metadata_file, "w") as file:
    json.dump(preprocessing_metadata, file, indent=4)

print(f"Preprocessing metadata saved: {metadata_file}")


# Standardize training images
x_train = (x_train - mean) / std


# Load validation images
x_validation = []
y_validation = []

for image_id, theta_E in validation_samples:

    image_path = image_dir / f"{image_id:06d}.npy"
    image = np.load(image_path)

    x_validation.append(image)
    y_validation.append(theta_E)


x_validation = np.array(x_validation, dtype=np.float32)
y_validation = np.array(y_validation, dtype=np.float32)


# Load testing images
x_test = []
y_test = []

for image_id, theta_E in test_samples:

    image_path = image_dir / f"{image_id:06d}.npy"
    image = np.load(image_path)

    x_test.append(image)
    y_test.append(theta_E)


x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)


# Apply training-set normalization to validation and testing images
x_validation = (x_validation - mean) / std
x_test = (x_test - mean) / std


# Save prepared datasets
prepared_dir = Path("../datasets")

np.save(prepared_dir / "x_train_1.npy", x_train)
np.save(prepared_dir / "y_train_1.npy", y_train)

np.save(prepared_dir / "x_validation_1.npy", x_validation)
np.save(prepared_dir / "y_validation_1.npy", y_validation)

np.save(prepared_dir / "x_test_1.npy", x_test)
np.save(prepared_dir / "y_test_1.npy", y_test)


print("\nV1 preprocessing completed!")
print("Training images:", x_train.shape)
print("Validation images:", x_validation.shape)
print("Testing images:", x_test.shape)
print("Training normalization mean:", mean)
print("Training normalization std:", std)
print("Prepared arrays saved.")
