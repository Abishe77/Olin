
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model

from inference.config import (
    MODEL_PATH,
    TARGET_NORMALIZATION_PATH,
    UNCERTAINTY_CALIBRATION_PATH
)


TARGET_COLUMNS = [
    "theta_E",
    "e1",
    "e2",
    "gamma1",
    "gamma2"
]


# Monte Carlo Dropout layer
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


# Residual block
def residual_block(x, filters, stride=1):

    shortcut = x

    x = layers.Conv2D(
        filters, 3, strides=stride,
        padding="same", use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters, 3,
        padding="same", use_bias=False
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


# Build ResNet-18 with MC Dropout
def build_mc_resnet18():

    inputs = layers.Input(shape=(160, 160, 1))

    x = layers.Conv2D(
        64, 7, strides=2,
        padding="same", use_bias=False
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(
        3, strides=2, padding="same"
    )(x)

    for filters, stride in [
        (64, 1),
        (128, 2),
        (256, 2),
        (512, 2)
    ]:

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


# Load all inference assets
def load_inference_assets():

    required_files = [
        MODEL_PATH,
        TARGET_NORMALIZATION_PATH,
        UNCERTAINTY_CALIBRATION_PATH
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required inference asset not found: {path}"
            )

    # Load the original trained model
    trained_model = tf.keras.models.load_model(MODEL_PATH)

    # Rebuild the same architecture with MC Dropout
    mc_model = build_mc_resnet18()

    # Transfer the trained weights
    mc_model.set_weights(trained_model.get_weights())

    # Load target normalization
    with open(TARGET_NORMALIZATION_PATH, "r") as file:
        normalization = json.load(file)

    if normalization["target_columns"] != TARGET_COLUMNS:
        raise ValueError("Target parameter order mismatch.")

    target_mean = np.asarray(
        normalization["mean"], dtype=np.float32
    )

    target_std = np.asarray(
        normalization["std"], dtype=np.float32
    )

    # Load uncertainty calibration factors
    with open(UNCERTAINTY_CALIBRATION_PATH, "r") as file:
        calibration_factors = json.load(file)

    calibration_array = np.asarray(
        [calibration_factors[name] for name in TARGET_COLUMNS],
        dtype=np.float32
    )

    if (
        target_mean.shape != (5,)
        or target_std.shape != (5,)
        or np.any(target_std <= 0)
        or not np.all(np.isfinite(target_mean))
        or not np.all(np.isfinite(target_std))
    ):
        raise ValueError("Invalid target normalization values.")

    if (
        calibration_array.shape != (5,)
        or not np.all(np.isfinite(calibration_array))
        or np.any(calibration_array <= 0)
    ):
        raise ValueError("Invalid uncertainty calibration factors.")

    print("Inference assets loaded successfully.")

    return {
        "model": mc_model,
        "target_mean": target_mean,
        "target_std": target_std,
        "calibration_factors": calibration_array,
        "target_columns": TARGET_COLUMNS
    }
