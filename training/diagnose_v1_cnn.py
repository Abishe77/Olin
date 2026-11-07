
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
datasets_dir = Path("../datasets")
results_dir = Path("results/v1_cnn_1")
plots_dir = results_dir / "diagnostics"
plots_dir.mkdir(parents=True, exist_ok=True)

# Load test data
x_test = np.load(datasets_dir / "x_test_1.npy")
y_test = np.load(datasets_dir / "y_test_1.npy")
x_test = x_test[..., np.newaxis]

# Load best trained model
model = tf.keras.models.load_model(
    "models/v1_cnn_1.keras"
)

# Generate predictions
predictions = model.predict(x_test, verbose=0).flatten()

errors = predictions - y_test
mae = np.mean(np.abs(errors))
mse = np.mean(errors ** 2)

print("Test MAE:", mae)
print("Test MSE:", mse)
print("Maximum absolute error:", np.max(np.abs(errors)))

# Load training history
with open(results_dir / "history.json", "r") as f:
    history = json.load(f)


# 1. Training and validation curves
plt.figure(figsize=(9, 5))
plt.plot(history["mae"], label="Training MAE")
plt.plot(history["val_mae"], label="Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE (arcsec)")
plt.title("V1 CNN - Training and Validation MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "training_curves.png", dpi=150)
plt.show()


# 2. True vs predicted Einstein radius
plt.figure(figsize=(7, 7))
plt.scatter(y_test, predictions, alpha=0.5)

lower = min(y_test.min(), predictions.min())
upper = max(y_test.max(), predictions.max())

plt.plot([lower, upper], [lower, upper], "r--", label="Perfect prediction")
plt.xlabel("True Einstein radius (arcsec)")
plt.ylabel("Predicted Einstein radius (arcsec)")
plt.title("V1 CNN - True vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "true_vs_predicted.png", dpi=150)
plt.show()


# 3. Residuals vs true Einstein radius
plt.figure(figsize=(9, 5))
plt.scatter(y_test, errors, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("True Einstein radius (arcsec)")
plt.ylabel("Prediction error (arcsec)")
plt.title("V1 CNN - Residual Analysis")
plt.grid(True)
plt.tight_layout()
plt.savefig(plots_dir / "residuals.png", dpi=150)
plt.show()


print("\nDiagnostic plots saved in:", plots_dir)
