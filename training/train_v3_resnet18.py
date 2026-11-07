
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import layers, Model


# Setting the project paths and training parameters
ROOT = Path("/mnt/d/Olin")
DATA_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "training" / "models"
RESULT_DIR = ROOT / "training" / "results" / "v3_resnet18"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "v3_resnet18.keras"

EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEED = 42

tf.keras.utils.set_random_seed(SEED)


# Loading the preprocessed images
x_train = np.load(DATA_DIR / "x_train_1.npy")
x_val = np.load(DATA_DIR / "x_validation_1.npy")
x_test = np.load(DATA_DIR / "x_test_1.npy")


# Loading all five target parameters using the same split as V2
labels = pd.read_csv(DATA_DIR / "dataset_v1" / "labels.csv")

indices = np.arange(len(labels))
rng = np.random.default_rng(SEED)
rng.shuffle(indices)

train_idx = indices[:8000]
val_idx = indices[8000:9000]
test_idx = indices[9000:10000]

target_columns = ["theta_E", "e1", "e2", "gamma1", "gamma2"]

y_train_raw = labels.iloc[train_idx][target_columns].to_numpy(dtype=np.float32)
y_val_raw = labels.iloc[val_idx][target_columns].to_numpy(dtype=np.float32)
y_test_raw = labels.iloc[test_idx][target_columns].to_numpy(dtype=np.float32)


# Checking whether the training split matches the previous V2 model
y_train_v2 = np.load(DATA_DIR / "y_train_1.npy")

assert np.allclose(y_train_raw[:, 0], y_train_v2), "Training split mismatch"


# Adding a channel dimension to the grayscale images
x_train = x_train[..., np.newaxis].astype(np.float32)
x_val = x_val[..., np.newaxis].astype(np.float32)
x_test = x_test[..., np.newaxis].astype(np.float32)


# Normalizing the target parameters using training data statistics
target_mean = y_train_raw.mean(axis=0)
target_std = y_train_raw.std(axis=0)

y_train = (y_train_raw - target_mean) / target_std
y_val = (y_val_raw - target_mean) / target_std

with open(RESULT_DIR / "target_normalization.json", "w") as f:
    json.dump({
        "target_columns": target_columns,
        "mean": target_mean.tolist(),
        "std": target_std.tolist()
    }, f, indent=4)


# Creating a residual block used in the ResNet architecture
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

    # Adjusting the shortcut when the input dimensions change
    if stride != 1 or shortcut.shape[-1] != filters:

        shortcut = layers.Conv2D(
            filters, 1, strides=stride,
            padding="same", use_bias=False
        )(shortcut)

        shortcut = layers.BatchNormalization()(shortcut)

    # Adding the original input to the output of the convolution layers
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


# Building the ResNet-18 model
def build_resnet18():

    inputs = layers.Input(shape=(160, 160, 1))

    # Initial convolution layer for grayscale images
    x = layers.Conv2D(
        64, 7, strides=2,
        padding="same", use_bias=False
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # ResNet-18 contains four stages with two residual blocks each
    for filters, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:

        x = residual_block(x, filters, stride)
        x = residual_block(x, filters)

    # Converting the feature maps into a compact feature vector
    x = layers.GlobalAveragePooling2D()(x)

    # Shared layer used by all three prediction heads
    shared = layers.Dense(256, activation="relu")(x)
    shared = layers.Dropout(0.3)(shared)

    # Predicting Einstein radius
    theta_head = layers.Dense(1, name="theta_E")(shared)

    # Predicting lens ellipticity components
    ellipticity_head = layers.Dense(2, name="ellipticity")(shared)

    # Predicting external shear components
    shear_head = layers.Dense(2, name="shear")(shared)

    outputs = layers.Concatenate()([
        theta_head,
        ellipticity_head,
        shear_head
    ])

    return Model(inputs, outputs)


model = build_resnet18()
model.summary()


# Compiling the model with Adam optimizer and mean squared error loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"]
)


# Setting callbacks to save the best model and control training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


# Training the model using the training and validation datasets
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)


# Saving the training history
with open(RESULT_DIR / "history.json", "w") as f:
    json.dump(
        {k: [float(v) for v in values]
         for k, values in history.history.items()},
        f,
        indent=4
    )


# Loading the best checkpoint for final evaluation
best_model = tf.keras.models.load_model(MODEL_PATH)

pred_scaled = best_model.predict(x_test, batch_size=BATCH_SIZE)

# Converting predictions back to their original parameter scales
predictions = pred_scaled * target_std + target_mean

errors = predictions - y_test_raw

mae = np.mean(np.abs(errors), axis=0)
rmse = np.sqrt(np.mean(errors ** 2, axis=0))

r2 = 1 - (
    np.sum(errors ** 2, axis=0) /
    np.sum((y_test_raw - y_test_raw.mean(axis=0)) ** 2, axis=0)
)

correlations = [
    np.corrcoef(y_test_raw[:, i], predictions[:, i])[0, 1]
    for i in range(5)
]


# Displaying the evaluation metrics for each parameter
metrics = {}

for i, name in enumerate(target_columns):

    metrics[name] = {
        "MAE": float(mae[i]),
        "RMSE": float(rmse[i]),
        "R2": float(r2[i]),
        "correlation": float(correlations[i])
    }

    print(f"\n{name}")
    print(f"MAE: {mae[i]:.6f}")
    print(f"RMSE: {rmse[i]:.6f}")
    print(f"R2: {r2[i]:.6f}")
    print(f"Correlation: {correlations[i]:.6f}")


# Saving the evaluation metrics and test predictions
with open(RESULT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

prediction_df = pd.DataFrame()

for i, name in enumerate(target_columns):
    prediction_df[f"true_{name}"] = y_test_raw[:, i]
    prediction_df[f"pred_{name}"] = predictions[:, i]

prediction_df.to_csv(RESULT_DIR / "test_predictions.csv", index=False)

print("\nResNet-18 training and evaluation completed.")
