import numpy as np
import tensorflow as tf

x_test = np.load("../datasets/x_test_0.npy")
x_test = x_test[...,np.newaxis]

cnn_0 = tf.keras.models.load_model("models/cnn_0.keras")
predictions_0 = cnn_0.predict(x_test).flatten()

np.save("results/cnn_1/predictions_cnn_0.npy" , predictions_0)

y_test = np.load("../datasets/y_test_0.npy")
predictions = np.load("results/cnn_1/predictions.npy")

print(y_test.shape)
print(predictions.shape)

#to take certain intervals of theta_e from our min and max itself
#and check how  cnn_1  perfomrs on each

bins = [
    (1.6, 1.9),
    (1.9, 2.2),
    (2.2, 2.5),
    (2.5, 2.8)
]

errors = predictions - y_test
absolute_errors = np.abs(errors)

errors_0 = predictions_0 - y_test
absolute_errors_0 = np.abs(errors_0)

for lower , upper in bins:
    mask = (y_test >= lower) & (y_test < upper)
    #test value should not exceed the values within the range we choose

    bin_errors_0 = errors_0[mask]
    bin_absolute_errors_0 = absolute_errors_0[mask]

    mae_0 = np.mean(bin_absolute_errors_0)
    bias_0 = np.mean(bin_errors_0)

    bin_errors = errors[mask]
    bin_absolute_errors = absolute_errors[mask]

    count = np.sum(mask)

    mae = np.mean(bin_absolute_errors)
    bias = np.mean(bin_errors)


    print(f"\nRange: {lower} - {upper}")
    print(f"Samples: {count}")

    print(f"CNN_0 MAE: {mae_0:.6f}")
    print(f"CNN_0 Bias: {bias_0:.6f}")

    print(f"CNN_1 MAE: {mae:.6f}")
    print(f"CNN_1 Bias: {bias:.6f}")