
import copy
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from dataset_generation_forward_model import (
    build_image_model,
    make_lens_kwargs,
    make_source_kwargs,
    photon_scale,
    read_noise,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "compensation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

image_model = build_image_model()

baseline_lens = make_lens_kwargs(
    theta_E=2.2,
    e1=0.15,
    e2=0.0,
    gamma1=0.05,
    gamma2=0.02,
)

source = make_source_kwargs(
    amp=1.0,
    R_sersic=0.3,
    n_sersic=2.0,
    e1=0.0,
    e2=0.0,
    center_x=0.6,
    center_y=0.3,
)


def generate_image(lens_kwargs):
    return image_model.image(
        kwargs_lens=lens_kwargs,
        kwargs_source=source,
    )


print("Generating baseline image...")
baseline_image = generate_image(baseline_lens)
baseline_norm = np.linalg.norm(baseline_image)


def calculate_metrics(candidate_image):
    difference = candidate_image - baseline_image

    relative_difference = (
        np.linalg.norm(difference) / baseline_norm
    )

    pixel_rms = np.sqrt(np.mean(difference ** 2))

    variance = (
        np.maximum(baseline_image, 0)
        + np.maximum(candidate_image, 0)
    ) / photon_scale

    variance += 2 * (read_noise / photon_scale) ** 2

    detection_snr = np.sqrt(
        np.sum(difference ** 2 / variance)
    )

    return relative_difference, pixel_rms, detection_snr


# Each pair is varied independently.
# The other ellipticity/shear components remain fixed.

experiments = [
    {
        "name": "e1_gamma1",
        "e_parameter": "e1",
        "gamma_parameter": "gamma1",
        "e_values": np.linspace(0.07, 0.23, 9),
        "gamma_bounds": (
            -np.sqrt(0.1**2 - 0.02**2),
            np.sqrt(0.1**2 - 0.02**2),
        ),
    },
    {
        "name": "e2_gamma2",
        "e_parameter": "e2",
        "gamma_parameter": "gamma2",
        "e_values": np.linspace(-0.20, 0.20, 9),
        "gamma_bounds": (
            -np.sqrt(0.1**2 - 0.05**2),
            np.sqrt(0.1**2 - 0.05**2),
        ),
    },
]


all_results = []


for experiment in experiments:

    name = experiment["name"]
    e_parameter = experiment["e_parameter"]
    gamma_parameter = experiment["gamma_parameter"]

    e_values = experiment["e_values"]
    gamma_bounds = experiment["gamma_bounds"]

    print("\n" + "=" * 65)
    print("EXPERIMENT:", name)
    print("=" * 65)

    for e_value in e_values:

        # First measure the image change without compensation.
        unadjusted_lens = copy.deepcopy(baseline_lens)

        unadjusted_lens[0][e_parameter] = float(e_value)

        unadjusted_image = generate_image(unadjusted_lens)

        _, _, unadjusted_snr = calculate_metrics(unadjusted_image)

        # Then search for the best compensating shear.
        def objective(gamma_value):

            candidate_lens = copy.deepcopy(baseline_lens)

            candidate_lens[0][e_parameter] = float(e_value)
            candidate_lens[1][gamma_parameter] = float(gamma_value)

            candidate_image = generate_image(candidate_lens)

            _, _, snr = calculate_metrics(candidate_image)

            return snr ** 2

        result = minimize_scalar(
            objective,
            bounds=gamma_bounds,
            method="bounded",
            options={"xatol": 1e-5},
        )

        best_gamma = float(result.x)

        best_lens = copy.deepcopy(baseline_lens)

        best_lens[0][e_parameter] = float(e_value)
        best_lens[1][gamma_parameter] = best_gamma

        best_image = generate_image(best_lens)

        relative_difference, pixel_rms, best_snr = (
            calculate_metrics(best_image)
        )

        row = {
            "experiment": name,
            "ellipticity_value": float(e_value),
            "best_gamma": best_gamma,
            "unadjusted_snr": float(unadjusted_snr),
            "compensated_snr": float(best_snr),
            "relative_image_difference": float(relative_difference),
            "pixel_rms_difference": float(pixel_rms),
        }

        all_results.append(row)

        print(
            f"{e_parameter}={e_value:+.4f} | "
            f"best {gamma_parameter}={best_gamma:+.5f} | "
            f"SNR before={unadjusted_snr:.3f} | "
            f"SNR after={best_snr:.3f}"
        )


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

csv_path = OUTPUT_DIR / "compensation_results.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=all_results[0].keys(),
    )

    writer.writeheader()
    writer.writerows(all_results)


# --------------------------------------------------
# PLOTS
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, experiment in zip(axes, experiments):

    selected = [
        row for row in all_results
        if row["experiment"] == experiment["name"]
    ]

    e_values = [
        row["ellipticity_value"] for row in selected
    ]

    gamma_values = [
        row["best_gamma"] for row in selected
    ]

    ax.plot(e_values, gamma_values, marker="o")

    ax.set_title(experiment["name"])
    ax.set_xlabel(experiment["e_parameter"])
    ax.set_ylabel(
        "Best compensating " + experiment["gamma_parameter"]
    )

    ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "compensation_relationships.png",
    dpi=150,
)

plt.close()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, experiment in zip(axes, experiments):

    selected = [
        row for row in all_results
        if row["experiment"] == experiment["name"]
    ]

    e_values = [
        row["ellipticity_value"] for row in selected
    ]

    snr_values = [
        row["compensated_snr"] for row in selected
    ]

    ax.plot(e_values, snr_values, marker="o")

    ax.set_title(experiment["name"])
    ax.set_xlabel(experiment["e_parameter"])
    ax.set_ylabel("SNR after compensation")

    ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "compensated_detection_snr.png",
    dpi=150,
)

plt.close()


print("\n" + "=" * 65)
print("COMPENSATION EXPERIMENT COMPLETE")
print("=" * 65)
print("Results:", OUTPUT_DIR)

