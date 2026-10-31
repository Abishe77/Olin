import numpy as np
from pathlib import Path
import csv

from dataset_generation_forward_model import make_lens_kwargs
from dataset_generation_forward_model import make_source_kwargs
from dataset_generation_forward_model import build_image_model
from dataset_generation_forward_model import simulate


theta_E_min = 1.6
theta_E_max = 2.8

num_samples = 10000

rng = np.random.default_rng(42)

dataset_dir = Path("../datasets/dataset_v0")
image_dir = dataset_dir / "images"

image_dir.mkdir(parents=True, exist_ok=True)


image_model = build_image_model()

labels = []


for i in range(num_samples):

    theta_E = rng.uniform(theta_E_min, theta_E_max)

    kwargs_lens = make_lens_kwargs(theta_E=theta_E)
    kwargs_source = make_source_kwargs()

    clean_image, noisy_image = simulate(
        image_model,
        kwargs_lens,
        kwargs_source,
        seed=i
    )

    image_id = f"{i + 1:06d}"

    np.save(
        image_dir / f"{image_id}.npy",
        noisy_image
    )

    labels.append([image_id, theta_E])

    print(f"generated {image_id} | theta_E = {theta_E:.4f}")


with open(dataset_dir / "labels.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["image_id", "theta_E"])

    writer.writerows(labels)


print("\nDataset generation complete.")
print(f"images: {num_samples}")
print(f"location: {dataset_dir}")
