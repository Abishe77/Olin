
import copy
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dataset_generation_forward_model import (
    build_image_model,
    make_lens_kwargs,
    make_source_kwargs,
    add_noise,
    photon_scale,
    read_noise,
)




OUTPUT_DIR = Path(__file__).resolve().parent / "sensitivity_results"
OUTPUT_DIR.mkdir(exist_ok=True)

image_model = build_image_model()

# Fixed baseline: all other parameters remain unchanged.
baseline_lens = make_lens_kwargs(
    theta_E=2.2,
    e1=0.15,
    e2=0.0,
    gamma1=0.05,
    gamma2=0.02,
)

baseline_source = make_source_kwargs(
    amp=1.0,
    R_sersic=0.3,
    n_sersic=2.0,
    e1=0.0,
    e2=0.0,
    center_x=0.6,
    center_y=0.3,
)


# Values selected within the approximate V1 component ranges.
experiments = {
    "e1": [-0.25, -0.125, 0.0, 0.15, 0.25],
    "e2": [-0.25, -0.125, 0.0, 0.125, 0.25],
    "gamma1": [-0.08, -0.04, 0.0, 0.05, 0.08],
    "gamma2": [-0.08, -0.04, 0.02, 0.04, 0.08],
}


def generate_clean_image(lens_kwargs):

    return image_model.image(
        kwargs_lens=lens_kwargs,
        kwargs_source=baseline_source,
    )


print("Generating baseline image...")

baseline_clean = generate_clean_image(baseline_lens)

baseline_norm = np.linalg.norm(baseline_clean)

print("Baseline image generated.")
print("Maximum intensity:", baseline_clean.max())




results = []
response_patterns = {}

for parameter, values in experiments.items():

    print(f"\nTesting {parameter}...")

    parameter_results = []

    for value in values:

        modified_lens = copy.deepcopy(baseline_lens)

        # Change exactly one component.
        modified_lens[0 if parameter in ["e1", "e2"] else 1][parameter] = value

        modified_clean = generate_clean_image(modified_lens)

        # Difference between the two ideal images.
        difference = modified_clean - baseline_clean

        relative_change = (
            np.linalg.norm(difference) / baseline_norm
        )

        pixel_rms_difference = np.sqrt(np.mean(difference ** 2))

        # Approximate independent Poisson + read-noise variance.
        # Both images contribute noise to their difference.
        variance = (
            np.maximum(baseline_clean, 0)
            + np.maximum(modified_clean, 0)
        ) / photon_scale

        variance += 2 * (read_noise / photon_scale) ** 2

        # Idealized detectability under the assumed noise model.
        detection_snr = np.sqrt(
            np.sum(difference ** 2 / variance)
        )

        parameter_results.append({
            "parameter": parameter,
            "value": value,
            "relative_image_change": relative_change,
            "pixel_rms_difference": pixel_rms_difference,
            "idealized_detection_snr": detection_snr,
        })

        results.append(parameter_results[-1])

        print(
            f"  {value: .4f} | "
            f"change={relative_change:.6f} | "
            f"RMS={pixel_rms_difference:.8f} | "
            f"SNR={detection_snr:.3f}"
        )



    # Use a symmetric perturbation around the baseline.
    step = 0.05 if parameter in ["e1", "e2"] else 0.02

    plus_lens = copy.deepcopy(baseline_lens)
    minus_lens = copy.deepcopy(baseline_lens)

    lens_index = 0 if parameter in ["e1", "e2"] else 1

    plus_lens[lens_index][parameter] += step
    minus_lens[lens_index][parameter] -= step

    plus_image = generate_clean_image(plus_lens)
    minus_image = generate_clean_image(minus_lens)

    response = (
        plus_image - minus_image
    ) / (2 * step)

    response_patterns[parameter] = response.flatten()


print("\n" + "=" * 65)
print("RESPONSE PATTERN SIMILARITY")
print("=" * 65)

parameters = list(response_patterns.keys())

similarities = []

for i in range(len(parameters)):

    for j in range(i + 1, len(parameters)):

        p1 = parameters[i]
        p2 = parameters[j]

        r1 = response_patterns[p1]
        r2 = response_patterns[p2]

        cosine = np.dot(r1, r2) / (
            np.linalg.norm(r1) * np.linalg.norm(r2)
        )

        similarities.append({
            "parameter_1": p1,
            "parameter_2": p2,
            "cosine_similarity": float(cosine),
            "absolute_similarity": float(abs(cosine)),
        })

        print(
            f"{p1:8s} vs {p2:8s}: "
            f"{cosine:.5f}"
        )




csv_path = OUTPUT_DIR / "parameter_sensitivity.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "parameter",
            "value",
            "relative_image_change",
            "pixel_rms_difference",
            "idealized_detection_snr",
        ],
    )

    writer.writeheader()
    writer.writerows(results)


similarity_path = OUTPUT_DIR / "response_similarities.csv"

with open(similarity_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "parameter_1",
            "parameter_2",
            "cosine_similarity",
            "absolute_similarity",
        ],
    )

    writer.writeheader()
    writer.writerows(similarities)




fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, parameter in zip(axes.flat, experiments):

    selected = [
        row for row in results
        if row["parameter"] == parameter
    ]

    values = [row["value"] for row in selected]

    changes = [
        row["relative_image_change"] for row in selected
    ]

    ax.plot(values, changes, marker="o")

    ax.set_title(parameter)
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Relative image change")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "relative_image_changes.png",
    dpi=150,
)
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, parameter in zip(axes.flat, experiments):

    selected = [
        row for row in results
        if row["parameter"] == parameter
    ]

    values = [row["value"] for row in selected]

    snrs = [
        row["idealized_detection_snr"] for row in selected
    ]

    ax.plot(values, snrs, marker="o")

    ax.set_title(parameter)
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Idealized detection SNR")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "detection_snr.png",
    dpi=150,
)
plt.close()


print("\n" + "=" * 65)
print("SENSITIVITY EXPERIMENT COMPLETE")
print("=" * 65)

print("Results:", OUTPUT_DIR)
