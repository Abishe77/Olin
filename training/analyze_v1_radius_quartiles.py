
import csv
import numpy as np
from pathlib import Path

dataset_dir = Path("../datasets/dataset_v1")
results_dir = Path("results/v1_cnn_1/diagnostics")

# Load existing prediction errors
with open(results_dir / "parameter_errors/v1_error_analysis.csv") as f:
    predictions = list(csv.DictReader(f))

# Load original physical labels
with open(dataset_dir / "labels.csv") as f:
    labels = {
        int(row["image_id"]): row
        for row in csv.DictReader(f)
    }

# Match source radius with prediction error
data = []

for row in predictions:
    image_id = int(row["image_id"])

    data.append((
        float(labels[image_id]["R_sersic"]),
        float(row["absolute_error"])
    ))

data = np.array(data)

radii = data[:, 0]
errors = data[:, 1]

# Divide into four equal-sized groups
quartiles = np.quantile(radii, [0, 0.25, 0.50, 0.75, 1])

print("\nSOURCE RADIUS QUARTILE ANALYSIS")
print("-" * 65)

summary = []

for i in range(4):
    if i == 3:
        mask = (radii >= quartiles[i]) & (radii <= quartiles[i+1])
    else:
        mask = (radii >= quartiles[i]) & (radii < quartiles[i+1])

    group_errors = errors[mask]

    mae = np.mean(group_errors)
    median = np.median(group_errors)
    p90 = np.percentile(group_errors, 90)

    print(f"\nQ{i+1}")
    print(f"Radius range: {quartiles[i]:.4f} - {quartiles[i+1]:.4f}")
    print(f"Samples:      {len(group_errors)}")
    print(f"MAE:          {mae:.5f} arcsec")
    print(f"Median error: {median:.5f} arcsec")
    print(f"90th pct:     {p90:.5f} arcsec")

    summary.append((f"Q{i+1}", quartiles[i], quartiles[i+1],
                    len(group_errors), mae, median, p90))

# Save summary
with open(results_dir / "radius_quartile_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "quartile", "radius_min", "radius_max",
        "samples", "MAE", "median_error", "p90_error"
    ])
    writer.writerows(summary)

print("\nSummary saved to radius_quartile_summary.csv")
