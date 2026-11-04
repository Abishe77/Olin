import numpy as np
import tensorflow as tf

# Load test data
x_test = np.load("../datasets/x_test_0.npy")
y_test = np.load("../datasets/y_test_0.npy")

# Add channel dimension
x_test = x_test[..., np.newaxis]

# Load trained CNN
model = tf.keras.models.load_model("models/cnn_1_best.keras")

# Generate predictions
predictions = model.predict(x_test)

# Remove unnecessary output dimension
predictions = predictions.flatten()

print("Test images:", x_test.shape)
print("Actual values:", y_test.shape)
print("Predictions:", predictions.shape)
#calculating prediction errors
errors = predictions - y_test
absolute_errors = np.abs(errors)

#calculating evaluation metrics
mean_absolute_error = np.mean(absolute_errors)
median_absolute_error = np.median(absolute_errors)

percentile_90 = np.percentile(absolute_errors,90)
percentile_95 = np.percentile(absolute_errors, 95)

maximum_error = np.max(absolute_errors)
mean_signed_error = np.mean(errors)

np.save("results/cnn_1/predictions.npy", predictions)

print("\nCNN_1 Test Error Analysis")
print(f"Mean Absolute Error:   {mean_absolute_error:.6f} arcsec")
print(f"Median Absolute Error: {median_absolute_error:.6f} arcsec")
print(f"90th Percentile Error: {percentile_90:.6f} arcsec")
print(f"95th Percentile Error: {percentile_95:.6f} arcsec")
print(f"Maximum Absolute Error:{maximum_error:.6f} arcsec")
print(f"Mean Signed Error:     {mean_signed_error:.6f} arcsec")

print("\nPrecision check")
print(f"Mean:   {np.mean(absolute_errors):.10f}")
print(f"Median: {np.median(absolute_errors):.10f}")


