
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

tf.keras.utils.set_random_seed(SEED)


# Loading the test images
x_test = np.load(DATA_DIR / "x_test_1.npy")
x_test = x_test[..., np.newaxis].astype(np.float32)


# Loading the test labels using the same split as the training script
labels = pd.read_csv(DATA_DIR / "dataset_v1" / "labels.csv")

indices = np.arange(len(labels))
rng = np.random.default_rng(SEED)
rng.shuffle(indices)

test_idx = indices[9000:10000]

target_columns = ["theta_E", "e1", "e2", "gamma1", "gamma2"]

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

    # Keeping dropout active during inference
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


# Running multiple predictions with dropout active
all_predictions = []

for i in range(N_PASSES):

    batch_predictions = []

    for start in range(0, len(x_test), BATCH_SIZE):

        batch = x_test[start:start + BATCH_SIZE]

        # Batch normalization stays in inference mode.
        # Only the custom dropout layer remains stochastic.
        predictions = mc_model(batch, training=False).numpy()

        batch_predictions.append(predictions)

    all_predictions.append(
        np.concatenate(batch_predictions, axis=0)
    )

    print(f"Completed prediction pass {i + 1}/{N_PASSES}")


# Calculating mean and standard deviation across predictions
all_predictions = np.array(all_predictions)

mean_scaled = np.mean(all_predictions, axis=0)
std_scaled = np.std(all_predictions, axis=0, ddof=1)

# Converting predictions and uncertainty to original parameter units
mean_predictions = mean_scaled * target_std + target_mean
uncertainty = std_scaled * target_std


# Evaluating prediction accuracy and uncertainty
errors = mean_predictions - y_test

metrics = {}

for i, name in enumerate(target_columns):

    absolute_errors = np.abs(errors[:, i])
    parameter_uncertainty = uncertainty[:, i]

    correlation = np.corrcoef(
        absolute_errors,
        parameter_uncertainty
    )[0, 1]

    coverage_1sigma = np.mean(
        absolute_errors <= parameter_uncertainty
    )

    coverage_2sigma = np.mean(
        absolute_errors <= 2 * parameter_uncertainty
    )

    metrics[name] = {
        "MAE": float(np.mean(absolute_errors)),
        "RMSE": float(np.sqrt(np.mean(errors[:, i] ** 2))),
        "mean_uncertainty": float(np.mean(parameter_uncertainty)),
        "error_uncertainty_correlation": float(correlation),
        "coverage_1sigma": float(coverage_1sigma),
        "coverage_2sigma": float(coverage_2sigma)
    }

    print(f"\n{name}")
    print(f"MAE: {metrics[name]['MAE']:.6f}")
    print(f"Mean uncertainty: {metrics[name]['mean_uncertainty']:.6f}")
    print(f"Error-uncertainty correlation: {correlation:.6f}")
    print(f"1-sigma coverage: {coverage_1sigma:.4f}")
    print(f"2-sigma coverage: {coverage_2sigma:.4f}")


# Saving predictions and uncertainty estimates
prediction_df = pd.DataFrame()

for i, name in enumerate(target_columns):

    prediction_df[f"true_{name}"] = y_test[:, i]
    prediction_df[f"pred_{name}"] = mean_predictions[:, i]
    prediction_df[f"uncertainty_{name}"] = uncertainty[:, i]

prediction_df.to_csv(
    UNCERTAINTY_DIR / "test_uncertainty.csv",
    index=False
)

with open(UNCERTAINTY_DIR / "uncertainty_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

np.save(
    UNCERTAINTY_DIR / "mc_predictions.npy",
    all_predictions
)

print("\nMonte Carlo uncertainty estimation completed.")
