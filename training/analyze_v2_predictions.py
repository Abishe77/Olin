
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt





BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "v2_multoutput"

PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"

PLOT_DIR = RESULTS_DIR / "diagnostics"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PARAMETERS = ["theta_E", "e1", "e2", "gamma1", "gamma2"]




print("\nLoading V2 test predictions...")

with open(PREDICTIONS_PATH, "r", newline="") as file:
    rows = list(csv.DictReader(file))

print(f"Test samples: {len(rows)}")




summary = {}

for parameter in PARAMETERS:

    true_values = np.array(
        [float(row[f"true_{parameter}"]) for row in rows]
    )

    predictions = np.array(
        [float(row[f"predicted_{parameter}"]) for row in rows]
    )

    errors = predictions - true_values

    # Basic statistics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    true_std = np.std(true_values)
    pred_std = np.std(predictions)

    # Correlation
    if true_std > 0 and pred_std > 0:
        correlation = np.corrcoef(
            true_values, predictions
        )[0, 1]
    else:
        correlation = float("nan")

    # R-squared
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Standard deviation ratio
    std_ratio = pred_std / true_std if true_std > 0 else float("nan")

    # Baseline: predict the test-set mean
    baseline_predictions = np.full_like(
        true_values, np.mean(true_values)
    )
    baseline_mae = np.mean(
        np.abs(baseline_predictions - true_values)
    )

    summary[parameter] = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "baseline_MAE": float(baseline_mae),
        "correlation": float(correlation),
        "R2": float(r2),
        "true_std": float(true_std),
        "prediction_std": float(pred_std),
        "prediction_to_true_std_ratio": float(std_ratio),
    }

    print("\n" + "=" * 55)
    print(parameter)
    print("=" * 55)

    print(f"MAE:                 {mae:.6f}")
    print(f"Baseline MAE:        {baseline_mae:.6f}")
    print(f"Correlation:         {correlation:.6f}")
    print(f"R2:                  {r2:.6f}")
    print(f"True std:            {true_std:.6f}")
    print(f"Prediction std:      {pred_std:.6f}")
    print(f"Prediction/True std: {std_ratio:.4f}")



    lower = min(true_values.min(), predictions.min())
    upper = max(true_values.max(), predictions.max())

    plt.figure(figsize=(7, 7))

    plt.scatter(
        true_values,
        predictions,
        alpha=0.35,
        s=15,
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=2,
        label="Perfect prediction",
    )

    plt.xlabel(f"True {parameter}")
    plt.ylabel(f"Predicted {parameter}")
    plt.title(f"V2: True vs Predicted — {parameter}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"{parameter}_true_vs_predicted.png",
        dpi=150,
    )

    plt.close()




with open(RESULTS_DIR / "prediction_diagnostics.json", "w") as file:
    json.dump(summary, file, indent=4)

print("\n" + "=" * 55)
print("V2 PREDICTION ANALYSIS COMPLETE")
print("=" * 55)

print(f"Summary: {RESULTS_DIR / 'prediction_diagnostics.json'}")
print(f"Plots:   {PLOT_DIR}")
