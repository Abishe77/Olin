
#Loading the prepared datasets
import numpy as np
from pathlib import Path
import tensorflow as tf #for cnn
import json

tf.keras.utils.set_random_seed(42) #to reproduce the same results


datasets_dir = Path("../datasets")

x_train = np.load(datasets_dir / "x_train_0.npy")
y_train = np.load(datasets_dir / "y_train_0.npy")


x_validation = np.load(datasets_dir / "x_validation_0.npy")
y_validation = np.load(datasets_dir / "y_validation_0.npy")

x_test = np.load(datasets_dir / "x_test_0.npy")
y_test = np.load(datasets_dir / "y_test_0.npy")


#now the shapes are basically(no of sample , no of array rows , no of array cols)
#but CNN needs four things and the last one is channel
#normally RGB images have 3 , but since this is
#scientific grayscale image we'll keep it 1
#so the shape becomes(s,m,n,1)

x_train = x_train[...,np.newaxis]
x_validation = x_validation[...,np.newaxis]
x_test = x_test[...,np.newaxis]
# ... means keep the size same , newaxis just one dimension extra


model = tf.keras.Sequential([

    tf.keras.Input(shape=(160,160,1)),

    tf.keras.layers.Conv2D(16,(3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    #16 different filters of size 3x3 , so op has 16 channels

    tf.keras.layers.Conv2D(32,(3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    #takes those 16 channels or feature maps and learns 32 new filters

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.summary()

model.compile(
    optimizer="adam", #to adjust weights efficiently during training
    loss="mse", #squared error
    metrics=["mae"] #absolute error
)


#saving the best model based on validation MAE
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

results_dir = Path("results/cnn_1")
results_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = models_dir / "cnn_1_best.keras"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_mae",
    mode="min",
    save_best_only=True,
    verbose=1
)


#stopping training when validation performance stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_mae",
    mode="min",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


#reducing learning rate when validation performance plateaus
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_mae",
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


#training the model
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_validation,y_validation),
    epochs=30,
    batch_size=32,
    callbacks=[
        checkpoint_callback,
        early_stopping,
        reduce_lr
    ]
)


#loading the best model saved during training
best_model = tf.keras.models.load_model(checkpoint_path)


#evaluating the best model on the test dataset
test_loss, test_mae = best_model.evaluate(
    x_test,
    y_test,
    verbose=1
)


#saving the training history
history_data = {
    "loss": [float(x) for x in history.history["loss"]],
    "mae": [float(x) for x in history.history["mae"]],
    "val_loss": [float(x) for x in history.history["val_loss"]],
    "val_mae": [float(x) for x in history.history["val_mae"]]
}

with open(results_dir / "history.json", "w") as f:
    json.dump(history_data, f, indent=4)


#saving final evaluation metrics
metrics = {
    "test_mse": float(test_loss),
    "test_mae": float(test_mae),
    "epochs_trained": len(history.history["loss"]),
    "best_validation_mae": float(min(history.history["val_mae"]))
}

with open(results_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


print("CNN_1 best model saved")
print("Test MSE:", test_loss)
print("Test MAE:", test_mae)


predictions = best_model.predict(x_test)

for i in range(10):
    print(
        f"True: {y_test[i]:.4f} | "
        f"Predicted: {predictions[i, 0]:.4f} | "
        f"Error: {predictions[i, 0] - y_test[i]:.4f}"
    )
