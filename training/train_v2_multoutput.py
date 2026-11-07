import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks



BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATASET_DIR = PROJECT_DIR / "datasets"
LABELS_PATH = DATASET_DIR / "dataset_v1" / "labels.csv"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / "v2_multoutput"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "v2_multoutput.keras"

TARGET_NAMES = [
    "theta_E",
    "e1",
    "e2",
    "gamma1",
    "gamma2",
]

RANDOM_SEED = 42
EPOCHS = 30
BATCH_SIZE = 32

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)




print("\nLoading existing V1 image splits...")

x_train = np.load(DATASET_DIR / "x_train_1.npy")
x_val = np.load(DATASET_DIR / "x_validation_1.npy")
x_test = np.load(DATASET_DIR / "x_test_1.npy")

# Add grayscale channel dimension for Conv2D
x_train = x_train[..., np.newaxis]
x_val = x_val[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("Training images:  ", x_train.shape)
print("Validation images:", x_val.shape)
print("Test images:      ", x_test.shape)




print("\nLoading labels and recreating V1 split...")

with open(LABELS_PATH, "r", newline="") as file:
    rows = list(csv.DictReader(file))

rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(rows)

train_rows = rows[:8000]
val_rows = rows[8000:9000]
test_rows = rows[9000:]


def extract_targets(selected_rows):
    return np.array(
        [
            [float(row[name]) for name in TARGET_NAMES]
            for row in selected_rows
        ],
        dtype=np.float32,
    )


y_train = extract_targets(train_rows)
y_val = extract_targets(val_rows)
y_test = extract_targets(test_rows)

print("Training labels:  ", y_train.shape)
print("Validation labels:", y_val.shape)
print("Test labels:      ", y_test.shape)




print("\nVerifying split against existing V1 labels...")

y_train_v1 = np.load(DATASET_DIR / "y_train_1.npy")
y_val_v1 = np.load(DATASET_DIR / "y_validation_1.npy")
y_test_v1 = np.load(DATASET_DIR / "y_test_1.npy")

assert np.allclose(
    y_train[:, 0], y_train_v1.reshape(-1),
    rtol=1e-6, atol=1e-7
)

assert np.allclose(
    y_val[:, 0], y_val_v1.reshape(-1),
    rtol=1e-6, atol=1e-7
)

assert np.allclose(
    y_test[:, 0], y_test_v1.reshape(-1),
    rtol=1e-6, atol=1e-7
)

print("Verified! V2 uses the exact same V1 split.")




print("\nCalculating target normalization statistics...")

target_mean = np.mean(y_train, axis=0)
target_std = np.std(y_train, axis=0)

if np.any(target_std == 0):
    raise ValueError("At least one target has zero standard deviation.")

y_train_scaled = (y_train - target_mean) / target_std
y_val_scaled = (y_val - target_mean) / target_std

normalization_metadata = {
    "target_names": TARGET_NAMES,
    "mean": target_mean.tolist(),
    "std": target_std.tolist(),
    "note": "Statistics calculated from training labels only.",
}

with open(RESULTS_DIR / "target_normalization.json", "w") as file:
    json.dump(normalization_metadata, file, indent=4)

print("\nTarget normalization:")
for name, mean, std in zip(TARGET_NAMES, target_mean, target_std):
    print(f"{name:10s} mean={mean: .6f}, std={std: .6f}")




print("\nBuilding V2 multi-output CNN...")

model = models.Sequential([
    layers.Input(shape=x_train.shape[1:]),

    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(64, activation="relu"),

    # Five regression outputs
    layers.Dense(5),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="mse",
)

model.summary()




training_callbacks = [
    callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),

    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),

    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]




print("\nStarting V2 training...\n")

history = model.fit(
    x_train,
    y_train_scaled,
    validation_data=(x_val, y_val_scaled),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=training_callbacks,
    verbose=1,
)

with open(RESULTS_DIR / "history.json", "w") as file:
    json.dump(history.history, file, indent=4)


print("\nLoading best checkpoint...")

best_model = tf.keras.models.load_model(MODEL_PATH)

print("Generating test predictions...")

predictions_scaled = best_model.predict(
    x_test,
    batch_size=BATCH_SIZE,
    verbose=1,
)

# Convert predictions back to physical units
predictions = predictions_scaled * target_std + target_mean



print("\n" + "=" * 65)
print("V2 TEST RESULTS")
print("=" * 65)

metrics = {}

for i, name in enumerate(TARGET_NAMES):

    errors = predictions[:, i] - y_test[:, i]
    absolute_errors = np.abs(errors)

    mae = float(np.mean(absolute_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    median_error = float(np.median(absolute_errors))
    p90_error = float(np.percentile(absolute_errors, 90))

    metrics[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "Median_absolute_error": median_error,
        "P90_absolute_error": p90_error,
    }

    print(f"\n{name}")
    print("-" * 35)
    print(f"MAE:             {mae:.6f}")
    print(f"RMSE:            {rmse:.6f}")
    print(f"Median error:    {median_error:.6f}")
    print(f"90th percentile: {p90_error:.6f}")




print("\n" + "=" * 65)
print("CONSTANT-MEAN BASELINE")
print("=" * 65)

baseline_metrics = {}

for i, name in enumerate(TARGET_NAMES):

    # Predict the training-set mean for every test sample
    baseline_predictions = np.full(
        len(y_test),
        target_mean[i],
        dtype=np.float32,
    )

    baseline_mae = float(
        np.mean(np.abs(baseline_predictions - y_test[:, i]))
    )

    baseline_metrics[name] = baseline_mae

    print(f"{name:10s} baseline MAE: {baseline_mae:.6f}")



final_metrics = {
    "test_samples": int(len(y_test)),
    "target_names": TARGET_NAMES,
    "model_metrics": metrics,
    "constant_mean_baseline_MAE": baseline_metrics,
}

with open(RESULTS_DIR / "metrics.json", "w") as file:
    json.dump(final_metrics, file, indent=4)


# Save predictions and ground truth
prediction_path = RESULTS_DIR / "test_predictions.csv"

with open(prediction_path, "w", newline="") as file:
    writer = csv.writer(file)

    headers = []

    for name in TARGET_NAMES:
        headers.extend([
            f"true_{name}",
            f"predicted_{name}",
            f"absolute_error_{name}",
        ])

    writer.writerow(headers)

    for true_values, predicted_values in zip(y_test, predictions):

        output_row = []

        for true_value, predicted_value in zip(
            true_values,
            predicted_values,
        ):
            output_row.extend([
                float(true_value),
                float(predicted_value),
                float(abs(true_value - predicted_value)),
            ])

        writer.writerow(output_row)


np.save(RESULTS_DIR / "y_test_true.npy", y_test)
np.save(RESULTS_DIR / "y_test_predictions.npy", predictions)




print("\n" + "=" * 65)
print("V2 TRAINING COMPLETE")
print("=" * 65)

print(f"Model:       {MODEL_PATH}")
print(f"Results:     {RESULTS_DIR}")
print(f"Predictions: {prediction_path}")

print("\nAll five parameters evaluated in their physical units.")