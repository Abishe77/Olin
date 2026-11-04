import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from FM2.reverse_reconstruction_model import reconstruct_image

ROOT = Path(__file__).resolve().parent.parent

# Load CNN₁ prediction and original observation
data = np.load(ROOT / "FM2" / "cnn1_prediction.npz")

image_id = str(data["image_id"])
true_theta_E = float(data["true_theta_E"])
predicted_theta_E = float(data["predicted_theta_E"])
observed_image = data["observed_image"]

print("Image ID:", image_id)
print("Ground truth:", true_theta_E)
print("CNN₁ prediction:", predicted_theta_E)

# Reconstruct using CNN₁ prediction
predicted_reconstruction = reconstruct_image(predicted_theta_E)

# Reconstruct using ground truth
true_reconstruction = reconstruct_image(true_theta_E)

print("\nReconstruction shape:", predicted_reconstruction.shape)
print("Predicted reconstruction contains NaN:",
      np.isnan(predicted_reconstruction).any())


residual = observed_image - predicted_reconstruction

fig, axes = plt.subplots(1, 5, figsize=(22, 5))

images = [
    observed_image,
    true_reconstruction,
    predicted_reconstruction,
    residual,
    observed_image - true_reconstruction
]

titles = [
    "Noisy Observation",
    "Ground-truth Reconstruction",
    "CNN₁ Reconstruction",
    "CNN₁ Residual",
    "Ground-truth Residual"
]

for ax, image, title in zip(axes, images, titles):
    im = ax.imshow(image, origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.show()