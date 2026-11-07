
import numpy as np
from scipy.optimize import least_squares

from FM1.dataset_generation_forward_model import (
    build_image_model,
    make_lens_kwargs,
    make_source_kwargs,
    num_pix
)


def reconstruct(observed_image, lens_predictions):
    """
    Fit a parametric Sérsic source using fixed CNN-predicted
    lens parameters, then generate a reconstructed image.

    IMPORTANT:
    observed_image must be in the original FM1 intensity scale,
    NOT the standardized image tensor passed to the CNN.
    """

    observed_image = np.asarray(observed_image, dtype=np.float64)

    # Validate input
    if observed_image.shape != (num_pix, num_pix):
        raise ValueError(
            f"Expected image shape {(num_pix, num_pix)}, "
            f"received {observed_image.shape}"
        )

    if not np.all(np.isfinite(observed_image)):
        raise ValueError("Input image contains NaN or Inf.")

    # Build lens model using CNN predictions
    kwargs_lens = make_lens_kwargs(
        theta_E=lens_predictions["theta_E"],
        e1=lens_predictions["e1"],
        e2=lens_predictions["e2"],
        gamma1=lens_predictions["gamma1"],
        gamma2=lens_predictions["gamma2"]
    )

    image_model = build_image_model()

    # Parameters being fitted:
    # [source_x, source_y, R_sersic, n_sersic, amplitude]

    def render(parameters):
        center_x, center_y, radius, index, amplitude = parameters

        kwargs_source = make_source_kwargs(
            amp=amplitude,
            R_sersic=radius,
            n_sersic=index,
            e1=0.0,
            e2=0.0,
            center_x=center_x,
            center_y=center_y
        )

        return image_model.image(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source
        )

    def residual_function(parameters):
        model_image = render(parameters)
        return (model_image - observed_image).ravel()

    # Initial estimates
    initial = np.array([
        0.0,   # source centre x
        0.0,   # source centre y
        0.30,  # Sérsic radius
        2.0,   # Sérsic index
        1.0    # amplitude
    ])

    # Bounds based on the synthetic training population
    lower_bounds = np.array([
        -0.8, -0.8, 0.15, 1.0, 0.01
    ])

    upper_bounds = np.array([
         0.8,  0.8, 0.50, 4.0, 100.0
    ])

    # Fit source parameters while keeping lens parameters fixed
    result = least_squares(
        residual_function,
        initial,
        bounds=(lower_bounds, upper_bounds),
        x_scale=np.array([0.5, 0.5, 0.2, 1.0, 1.0]),
        max_nfev=100
    )

    fitted = result.x

    reconstructed_image = render(fitted)
    residual_image = observed_image - reconstructed_image

    return {
        "source_parameters": {
            "center_x": float(fitted[0]),
            "center_y": float(fitted[1]),
            "R_sersic": float(fitted[2]),
            "n_sersic": float(fitted[3]),
            "amp": float(fitted[4])
        },
        "reconstructed_image": reconstructed_image,
        "residual_image": residual_image,
        "metrics": {
            "rmse": float(np.sqrt(np.mean(residual_image ** 2))),
            "mae": float(np.mean(np.abs(residual_image)))
        },
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "evaluations": int(result.nfev)
        }
    }
