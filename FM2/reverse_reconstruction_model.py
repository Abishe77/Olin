#This is file we are going to use as reverse contruction engine
# when CNN predicts parameters this is sent here 
#the image is resconstructed using these predicted params
#and compared against original grounf truth images


import numpy as np
import matplotlib.pyplot as plt

from FM1.dataset_generation_forward_model import(
    build_image_model,
    make_lens_kwargs,
    make_source_kwargs
)

image_model = build_image_model()

def reconstruct_image(theta_E):

    kwargs_lens = make_lens_kwargs(theta_E=theta_E)
    kwargs_source = make_source_kwargs()

    reconstructed_image = image_model.image(
        kwargs_lens = kwargs_lens,
        kwargs_source = kwargs_source
    )

    return reconstructed_image

if __name__ == "__main__":
    image = reconstruct_image(theta_E=2.2)

    print("Image shape:", image.shape)
    print("Minimum pixel value:", np.min(image))
    print("Maximum pixel value:", np.max(image))
    print("Contains NaN:", np.isnan(image).any())
    print("Contains Inf:", np.isinf(image).any())

    plt.figure(figsize=(6, 6))
    plt.imshow(image, origin="lower", cmap="magma")
    plt.colorbar(label="Intensity")
    plt.title("FM2 Reconstruction (theta_E = 2.2)")
    plt.xlabel("Pixel")
    plt.ylabel("Pixel")
    plt.show()