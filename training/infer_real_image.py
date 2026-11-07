
import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import layers, Model




ROOT = Path("/mnt/d/Olin")

IMAGE_PATH = ROOT / "real_data/HE0435-1223/exploratory_model_input.npy"
OUTPUT_DIR = ROOT / "real_data/HE0435-1223"

MODEL_PATH = ROOT / "training/models/v3_resnet18.keras"
RESULT_DIR = ROOT / "training/results/v3_resnet18"
UNCERTAINTY_DIR = RESULT_DIR / "uncertainty"

NORM_PATH = RESULT_DIR / "target_normalization.json"
CALIBRATION_PATH = UNCERTAINTY_DIR / "calibration_factors.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PASSES = 30
SEED = 42

TARGET_NAMES = ["theta_E", "e1", "e2", "gamma1", "gamma2"]

tf.keras.utils.set_random_seed(SEED)




image = np.load(IMAGE_PATH).astype(np.float32)

if image.shape != (160, 160):
    raise ValueError(f"Unexpected image shape: {image.shape}")

image = image[np.newaxis, ..., np.newaxis]

with open(NORM_PATH) as f:
    normalization = json.load(f)

target_mean = np.array(normalization["mean"], dtype=np.float32)
target_std = np.array(normalization["std"], dtype=np.float32)

with open(CALIBRATION_PATH) as f:
    calibration_factors = json.load(f)

factor_array = np.array(
    [calibration_factors[name] for name in TARGET_NAMES],
    dtype=np.float32
)




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



trained_model = tf.keras.models.load_model(MODEL_PATH)

mc_model = build_mc_resnet18()
mc_model.set_weights(trained_model.get_weights())

print("Frozen ResNet-18 weights loaded successfully.")




all_predictions = []

for i in range(N_PASSES):

    prediction = mc_model(image, training=False).numpy()[0]

    all_predictions.append(prediction)

    print(f"Completed MC pass {i + 1}/{N_PASSES}")

all_predictions = np.asarray(all_predictions)

mean_scaled = np.mean(all_predictions, axis=0)
std_scaled = np.std(all_predictions, axis=0, ddof=1)




mean_predictions = mean_scaled * target_std + target_mean
raw_uncertainty = std_scaled * target_std




calibrated_uncertainty = raw_uncertainty * factor_array




results = {
    "case": "HE 0435-1223",
    "evaluation_type": "exploratory_real_image_inference",
    "model": "Olin ResNet-18",
    "mc_passes": N_PASSES,
    "parameters": {}
}

print("\nEXPLORATORY PREDICTIONS")

for i, name in enumerate(TARGET_NAMES):

    prediction = float(mean_predictions[i])
    raw_sigma = float(raw_uncertainty[i])
    calibrated_sigma = float(calibrated_uncertainty[i])

    results["parameters"][name] = {
        "prediction": prediction,
        "raw_uncertainty": raw_sigma,
        "calibration_factor": float(factor_array[i]),
        "calibrated_uncertainty": calibrated_sigma
    }

    print(f"\n{name}")
    print(f"Prediction: {prediction:.6f}")
    print(f"Raw uncertainty: {raw_sigma:.6f}")
    print(f"Calibration factor: {factor_array[i]:.6f}")
    print(f"Calibrated uncertainty: {calibrated_sigma:.6f}")



results["limitations"] = [
    "Input is a display-rendered GIF, not calibrated scientific data.",
    "Per-image standardization is exploratory.",
    "Predictions are not scientifically validated.",
    "Calibration factors were derived from synthetic validation data.",
    "Calibrated uncertainty does not establish real-domain coverage."
]

with open(OUTPUT_DIR / "exploratory_inference.json", "w") as f:
    json.dump(results, f, indent=4)

np.save(
    OUTPUT_DIR / "exploratory_mc_predictions.npy",
    all_predictions
)

print("\nExploratory inference completed.")
print("Results saved in:", OUTPUT_DIR)
