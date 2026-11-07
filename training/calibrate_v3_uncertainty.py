
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import layers, Model


# Setting project paths and parameters
ROOT = Path("/mnt/d/Olin")
DATA_DIR = ROOT / "datasets"
MODEL_PATH = ROOT / "training" / "models" / "v3_resnet18.keras"
RESULT_DIR = ROOT / "training" / "results" / "v3_resnet18"

UNCERTAINTY_DIR = RESULT_DIR / "uncertainty"
UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)

N_PASSES = 30
BATCH_SIZE = 32
SEED = 42

TARGET_COVERAGE = 0.6827
TWO_SIGMA_MULTIPLIER = 1.96

tf.keras.utils.set_random_seed(SEED)


# Loading the validation and test images
x_validation = np.load(DATA_DIR / "x_validation_1.npy")
x_test = np.load(DATA_DIR / "x_test_1.npy")

x_validation = x_validation[..., np.newaxis].astype(np.float32)
x_test = x_test[..., np.newaxis].astype(np.float32)


# Reconstructing the original dataset split
labels = pd.read_csv(DATA_DIR / "dataset_v1" / "labels.csv")

indices = np.arange(len(labels))
rng = np.random.default_rng(SEED)
rng.shuffle(indices)

validation_idx = indices[8000:9000]
test_idx = indices[9000:10000]

target_columns = ["theta_E", "e1", "e2", "gamma1", "gamma2"]

y_validation = labels.iloc[validation_idx][target_columns].to_numpy(
    dtype=np.float32
)

y_test = labels.iloc[test_idx][target_columns].to_numpy(
    dtype=np.float32
)


# Loading the target normalization values used during training
with open(RESULT_DIR / "target_normalization.json") as f:
    normalization = json.load(f)

target_mean = np.array(normalization["mean"])
target_std = np.array(normalization["std"])


# Creating a dropout layer that remains active during prediction
class MCDropout(layers.Layer):

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs):
        return tf.nn.dropout(inputs, rate=self.rate)

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config


# Creating the residual block
def residual_block(x, filters, stride=1):

    shortcut = x

    x = layers.Conv2D(
        filters, 3, strides=stride,
        padding="same", use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters, 3, padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:

        shortcut = layers.Conv2D(
            filters, 1, strides=stride,
            padding="same", use_bias=False
        )(shortcut)

        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


# Rebuilding ResNet-18 with Monte Carlo dropout
def build_mc_resnet18():

    inputs = layers.Input(shape=(160, 160, 1))

    x = layers.Conv2D(
        64, 7, strides=2,
        padding="same", use_bias=False
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    for filters, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:

        x = residual_block(x, filters, stride)
        x = residual_block(x, filters)

    x = layers.GlobalAveragePooling2D()(x)

    shared = layers.Dense(256, activation="relu")(x)
    shared = MCDropout(0.3)(shared)

    theta_head = layers.Dense(1, name="theta_E")(shared)
    ellipticity_head = layers.Dense(2, name="ellipticity")(shared)
    shear_head = layers.Dense(2, name="shear")(shared)

    outputs = layers.Concatenate()([
        theta_head,
        ellipticity_head,
        shear_head
    ])

    return Model(inputs, outputs)


# Loading the trained model and transferring its weights
trained_model = tf.keras.models.load_model(MODEL_PATH)

mc_model = build_mc_resnet18()
mc_model.set_weights(trained_model.get_weights())

print("Trained ResNet-18 weights loaded successfully.")


# Running multiple stochastic prediction passes
def estimate_uncertainty(images, dataset_name):

    all_predictions = []

    for i in range(N_PASSES):

        batch_predictions = []

        for start in range(0, len(images), BATCH_SIZE):

            batch = images[start:start + BATCH_SIZE]

            predictions = mc_model(
                batch,
                training=False
            ).numpy()

            batch_predictions.append(predictions)

        all_predictions.append(
            np.concatenate(batch_predictions, axis=0)
        )

        print(
            f"{dataset_name}: completed pass "
            f"{i + 1}/{N_PASSES}"
        )

    all_predictions = np.array(all_predictions)

    mean_scaled = np.mean(all_predictions, axis=0)
    std_scaled = np.std(all_predictions, axis=0, ddof=1)

    mean_predictions = mean_scaled * target_std + target_mean
    uncertainty = std_scaled * target_std

    return mean_predictions, uncertainty


# Estimating validation uncertainty
validation_predictions, validation_uncertainty = estimate_uncertainty(
    x_validation,
    "Validation"
)


# Calculating one calibration factor per parameter
calibration_factors = {}

for i, name in enumerate(target_columns):

    absolute_errors = np.abs(
        validation_predictions[:, i] - y_validation[:, i]
    )

    parameter_uncertainty = validation_uncertainty[:, i]

    ratios = absolute_errors / np.maximum(
        parameter_uncertainty,
        1e-8
    )

    factor = np.quantile(ratios, TARGET_COVERAGE)

    calibration_factors[name] = float(factor)

    print(f"\n{name}")
    print(f"Calibration factor: {factor:.6f}")


# Saving the calibration factors
with open(UNCERTAINTY_DIR / "calibration_factors.json", "w") as f:
    json.dump(calibration_factors, f, indent=4)


# Estimating test uncertainty without using test labels for calibration
test_predictions, test_uncertainty = estimate_uncertainty(
    x_test,
    "Test"
)


# Applying the validation-derived calibration factors
factor_array = np.array([
    calibration_factors[name]
    for name in target_columns
])

calibrated_uncertainty = test_uncertainty * factor_array


# Evaluating raw and calibrated uncertainty
metrics = {}

for i, name in enumerate(target_columns):

    errors = test_predictions[:, i] - y_test[:, i]
    absolute_errors = np.abs(errors)

    raw_sigma = test_uncertainty[:, i]
    calibrated_sigma = calibrated_uncertainty[:, i]

    correlation = np.corrcoef(
        absolute_errors,
        raw_sigma
    )[0, 1]

    raw_coverage_1sigma = np.mean(
        absolute_errors <= raw_sigma
    )

    raw_coverage_2sigma = np.mean(
        absolute_errors <= TWO_SIGMA_MULTIPLIER * raw_sigma
    )

    calibrated_coverage_1sigma = np.mean(
        absolute_errors <= calibrated_sigma
    )

    calibrated_coverage_2sigma = np.mean(
        absolute_errors <= TWO_SIGMA_MULTIPLIER * calibrated_sigma
    )

    metrics[name] = {
        "MAE": float(np.mean(absolute_errors)),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "calibration_factor": float(factor_array[i]),
        "mean_raw_uncertainty": float(np.mean(raw_sigma)),
        "mean_calibrated_uncertainty": float(np.mean(calibrated_sigma)),
        "error_uncertainty_correlation": float(correlation),
        "raw_1sigma_coverage": float(raw_coverage_1sigma),
        "raw_2sigma_coverage": float(raw_coverage_2sigma),
        "calibrated_1sigma_coverage": float(calibrated_coverage_1sigma),
        "calibrated_2sigma_coverage": float(calibrated_coverage_2sigma)
    }

    print(f"\n{name}")
    print(f"MAE: {metrics[name]['MAE']:.6f}")
    print(f"RMSE: {metrics[name]['RMSE']:.6f}")
    print(f"Calibration factor: {factor_array[i]:.6f}")
    print(f"Error-uncertainty correlation: {correlation:.6f}")

    print(
        f"Raw coverage: "
        f"1σ = {raw_coverage_1sigma:.4f}, "
        f"1.96σ = {raw_coverage_2sigma:.4f}"
    )

    print(
        f"Calibrated coverage: "
        f"1σ = {calibrated_coverage_1sigma:.4f}, "
        f"1.96σ = {calibrated_coverage_2sigma:.4f}"
    )


# Saving test predictions and uncertainty estimates
prediction_df = pd.DataFrame()

for i, name in enumerate(target_columns):

    prediction_df[f"true_{name}"] = y_test[:, i]
    prediction_df[f"pred_{name}"] = test_predictions[:, i]
    prediction_df[f"raw_uncertainty_{name}"] = test_uncertainty[:, i]
    prediction_df[f"calibrated_uncertainty_{name}"] = calibrated_uncertainty[:, i]

prediction_df.to_csv(
    UNCERTAINTY_DIR / "calibrated_test_uncertainty.csv",
    index=False
)


# Saving evaluation metrics
with open(UNCERTAINTY_DIR / "calibrated_uncertainty_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


print("\nUncertainty calibration and test evaluation completed.")
