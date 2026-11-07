
import json
from pathlib import Path

import numpy as np

from inference.preprocessing import load_image, preprocess_image
from inference.predictor import predict
from inference.reconstruction import reconstruct
from inference.report import generate_pdf


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "inference" / "results"


def run_pipeline(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load the original image for reconstruction.
    observed_image, input_type = load_image(image_path)

    # Prepare the image for the neural network.
    model_input = preprocess_image(image_path)

    # Predict the five lens parameters.
    prediction_result = predict(model_input)

    lens_parameters = {
        name: float(values["prediction"])
        for name, values in prediction_result["parameters"].items()
    }

    # Reconstruct the image using the predicted lens parameters.
    reconstruction_result = reconstruct(
        observed_image,
        lens_parameters
    )

    # Create output directory.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_dir = RESULTS_DIR / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save reconstruction and residual.
    reconstruction_path = output_dir / "reconstructed_image.npy"
    residual_path = output_dir / "residual_image.npy"
    report_path = output_dir / "result.json"

    np.save(
        reconstruction_path,
        reconstruction_result["reconstructed_image"]
    )

    np.save(
        residual_path,
        reconstruction_result["residual_image"]
    )

    # Distinguish synthetic validation from exploratory image inference.
    if input_type == "synthetic":
        evaluation_type = "Synthetic-domain validation"
        limitations = [
            "Results apply to the synthetic data distribution.",
            "Uncertainty calibration is based on synthetic validation data."
        ]
    else:
        evaluation_type = "Exploratory inference - uncalibrated image input"
        limitations = [
            "Input is an ordinary display image, not calibrated scientific data.",
            "Per-image standardization is used for exploratory inference.",
            "Predicted parameters have not been validated for this image domain.",
            "Uncertainty estimates are calibrated on synthetic data only.",
            "Reconstruction metrics are not scientifically interpretable."
        ]

    report = {
        "input_image": image_path.name,
        "input_type": input_type,
        "evaluation_type": evaluation_type,
        "model": prediction_result["model"],
        "mc_passes": prediction_result["mc_passes"],
        "lens_parameters": prediction_result["parameters"],
        "source_parameters": reconstruction_result["source_parameters"],
        "reconstruction_metrics": reconstruction_result["metrics"],
        "optimizer": reconstruction_result["optimizer"],
        "limitations": limitations,
        "outputs": {
            "reconstructed_image": str(reconstruction_path),
            "residual_image": str(residual_path)
        }
    }

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    pdf_path = output_dir / "report.pdf"

    generate_pdf(
    image_path=image_path,
    report=report,
    output_path=pdf_path
    )    

    print("\nOLIN PIPELINE ")
    print("Input:", image_path.name)
    print("Evaluation:", evaluation_type)

    print("\nLens parameters:")
    for name, values in prediction_result["parameters"].items():
        print(
            f"{name}: {values['prediction']:.5f} "
            f"+/- {values['calibrated_uncertainty']:.5f}"
        )

    print("\nReconstruction metrics:")
    for name, value in reconstruction_result["metrics"].items():
        print(f"{name.upper()}: {value:.8f}")

    print("\nOptimizer success:", reconstruction_result["optimizer"]["success"])
    print("Report saved:", report_path)
    print("PDF Report saved: ",pdf_path)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Olin inference pipeline."
    )

    parser.add_argument(
        "image",
        help="Input image: NPY, PNG, JPG, JPEG or WebP"
    )

    args = parser.parse_args()
    run_pipeline(args.image)

