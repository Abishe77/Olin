import numpy as np
import pandas as pd
from pathlib import Path
import json

from v1_parameter_sampling import sample_parameters

from dataset_generation_forward_model import (
    make_lens_kwargs,
    make_source_kwargs,
    build_image_model,
    simulate,
    border_flux_fraction
)

N_SAMPLES = 10_000
IMAGE_SIZE = 160
RANDOM_SEED = 42

OUTPUT_DIR = Path("../datasets/dataset_v1")
IMAGE_DIR = OUTPUT_DIR / "images"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)


rng = np.random.default_rng(RANDOM_SEED)

image_model = build_image_model()



metadata=[]

for i in range(N_SAMPLES):

    #  Sample physical parameters
    params = sample_parameters(rng)

    #  Construct lens model
    kwargs_lens = make_lens_kwargs(
        theta_E=params["theta_E"],
        e1=params["e1"],
        e2=params["e2"],
        gamma1=params["gamma1"],
        gamma2=params["gamma2"],
        center_x=0.0,
        center_y=0.0
    )

    #  Construct source model
    kwargs_source = make_source_kwargs(
        amp=1.0,
        R_sersic=params["R_sersic"],
        n_sersic=params["n_sersic"],
        e1=0.0,
        e2=0.0,
        center_x=params["center_x"],
        center_y=params["center_y"]
    )

    #  Generate clean and noisy images
    clean_image, noisy_image = simulate(
        image_model,
        kwargs_lens,
        kwargs_source,
        seed=RANDOM_SEED + i
    )

    # Save noisy image
    image_filename = f"{i + 1:06d}.npy"

    np.save(
    IMAGE_DIR / image_filename,
    noisy_image.astype(np.float32)
    )
   

    # Calculate boundary flux diagnostic
    edge_fraction = border_flux_fraction(clean_image, width=3)

    #  Store metadata
    metadata.append({
        "image_id": i+1,
        **params,
        "edge_flux_fraction": edge_fraction
    })

    #  Display progress
    if (i + 1) % 100 == 0:
        print(f"Generated {i + 1}/{N_SAMPLES} images")

# Save metadata
labels_df = pd.DataFrame(metadata)
labels_df.to_csv(OUTPUT_DIR / "labels.csv", index=False)

dataset_info = {
    "dataset_version": "V1",
    "num_samples": N_SAMPLES,
    "image_size": IMAGE_SIZE,
    "target": "theta_E",
    "random_seed": RANDOM_SEED,
    "image_format": "npy",
    "image_dtype": "float32"
}

with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(dataset_info, f, indent=4)



print("\nDataset generation completed!")
print(f"Images saved in: {IMAGE_DIR}")
print(f"Labels saved: {OUTPUT_DIR / 'labels.csv'}")
print(f"Total images: {len(labels_df)}")        