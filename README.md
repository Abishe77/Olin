# Olin

## A Scientific Machine Learning Project for Inverse Inference in Strong Gravitational Lensing

Olin explores the use of computer vision and deep learning to infer physical parameters of gravitational lenses from astronomical observations.

The project combines established gravitational-lensing physics, synthetic observation generation, supervised deep learning, uncertainty estimation, and physics-based image reconstruction into a scientific inverse-inference pipeline.

**Physical Parameters → Gravitational Lensing → Synthetic Observations → Deep Learning Inference → Physics-Based Reconstruction → Scientific Reporting**

Unlike a conventional image-classification project, Olin treats deep learning as a tool for investigating an established scientific inverse problem.

---

## Project Status

**Active Development — Scientific Inference Pipeline Completed; Application Development in Progress**

Olin has progressed from a controlled one-parameter synthetic experiment to multi-parameter inference, uncertainty estimation, reconstruction, exploratory real-observation testing, and automated scientific reporting.

### Completed

- First-principles gravitational-lensing physics experiments
- Singular Isothermal Ellipsoid (SIE) lens modelling
- Elliptical Sérsic source modelling
- External shear modelling
- PSF-based observation modelling
- Photon-counting and detector read-noise simulation
- FM1 synthetic observation generation
- Dataset V0 containing 10,000 labelled observations
- CNN₀ baseline and CNN₁ improved regression experiments
- CNN₀ vs CNN₁ quantitative evaluation
- FM2 deterministic physics-based reconstruction
- Ground-truth control reconstruction and residual analysis
- Dataset V1 with additional physical parameter variation
- Multi-output regression experiments
- V3 ResNet-18 multi-parameter inference
- Monte Carlo uncertainty estimation and calibration
- Image preprocessing for PNG, JPEG and WebP inputs
- Inference and reconstruction pipeline
- Automated scientific PDF report generation
- Exploratory inference using a real astronomical observation

### Current Development

- FastAPI application foundation
- PostgreSQL database integration
- Authentication and ownership-based access
- Upload and inference API integration
- Protected analysis and report retrieval
- Application-level testing and reproducibility

### Future Directions

- More rigorous validation using real astronomical observations
- Evaluation against additional observational datasets
- Investigation of parameter degeneracies and distribution shifts
- Improved real-observation compatibility
- Possible Gaia Focused Product Release evaluation in a future version

The current synthetic experiments provide controlled quantitative validation. Exploratory inference on real astronomical images is not yet evidence of general-purpose accuracy on observational data.

---

## 1. Scientific Objective

Gravitational lensing occurs when the gravitational field of a foreground mass distribution alters the apparent position and brightness distribution of light from a background source.

In strong gravitational lensing, this can produce structures such as Einstein rings, arcs, and multiple images.

The physical process can be represented by the lens equation:

$$
\boldsymbol{\beta}
=
\boldsymbol{\theta}
-
\boldsymbol{\alpha}(\boldsymbol{\theta})
$$

where:

- $\boldsymbol{\beta}$ represents the source-plane position.
- $\boldsymbol{\theta}$ represents the observed image-plane position.
- $\boldsymbol{\alpha}$ represents the gravitational deflection angle.

The forward problem uses a known lens mass distribution and source configuration to calculate the resulting lensed image.

Olin investigates the corresponding inverse problem:

> Given an observed lensed image, can a machine-learning model infer the physical parameters responsible for generating it?

The project progressively investigates this problem, beginning with Einstein-radius estimation and extending to multiple physical parameters.

Olin does not attempt to introduce new gravitational-lensing physics. It uses established physical models to investigate machine-learning-based inverse inference.

---

## 2. Physical Forward Model — FM1

FM1 is the synthetic observation generation component of Olin.

It converts physical lens and source parameters into simulated astronomical observations.

The production forward model uses established gravitational-lensing implementations provided by [lenstronomy](https://github.com/lenstronomy/lenstronomy).

### Forward Simulation Pipeline

```text
Lens Parameters
      +
Source Parameters
      |
      v
Gravitational Lensing
      |
      v
Ideal Lensed Image
      |
      v
Observation Effects
      |
      v
Point Spread Function (PSF)
      |
      v
Photon-Counting Noise
      |
      v
Detector Read Noise
      |
      v
Synthetic Observation
```

### Physical Configuration

**Lens model**

- Singular Isothermal Ellipsoid (SIE)
- External shear

**Background source**

- Elliptical Sérsic brightness profile

**Observation model**

- Image dimensions: 160 × 160 pixels
- Pixel scale: 0.05 arcseconds per pixel
- Field of view: 8 × 8 arcseconds
- Gaussian PSF with 0.4 arcsecond FWHM
- Photon-counting noise
- Gaussian detector read noise

The observation model is intentionally controlled. It is not intended to represent a complete telescope or detector simulator.

---

## 3. Synthetic Datasets

Synthetic observations provide an important advantage during initial model development: the physical parameters used to generate each image are known.

This allows inferred parameters to be evaluated directly against their ground-truth values.

### Dataset V0 — Controlled Einstein-Radius Inference

Dataset V0 contains 10,000 synthetic gravitational-lensing observations.

| Parameter | Configuration |
|---|---|
| Regression target | Einstein radius $\theta_E$ |
| Einstein-radius range | 1.6–2.8 arcseconds |
| Image dimensions | 160 × 160 |
| Pixel scale | 0.05 arcseconds/pixel |
| Field of view | 8 × 8 arcseconds |
| Lens model | SIE + external shear |
| Source model | Elliptical Sérsic |
| Dataset size | 10,000 |
| Training samples | 8,000 |
| Validation samples | 1,000 |
| Test samples | 1,000 |
| Random seed | 42 |

The initial dataset varied the Einstein radius while keeping the remaining physical and observational parameters controlled.

### Dataset V1 — Multi-Parameter Inference

Dataset V1 extends the physical variation beyond the Einstein radius.

The dataset contains 10,000 observations with the following supervised targets:

- Einstein radius ($\theta_E$)
- Lens ellipticity component 1 ($e_1$)
- Lens ellipticity component 2 ($e_2$)
- External shear component 1 ($\gamma_1$)
- External shear component 2 ($\gamma_2$)

The dataset includes additional variations in lens ellipticity, external shear and lens-centre position.

| Parameter | Configuration |
|---|---|
| Dataset size | 10,000 |
| Training samples | 8,000 |
| Validation samples | 1,000 |
| Test samples | 1,000 |
| Random seed | 42 |
| Image dimensions | 160 × 160 |
| Number of prediction targets | 5 |

Dataset V1 introduces a more challenging inverse problem because multiple physical parameters can influence the observed image morphology.

### Dataset Preparation

The dataset preparation pipeline:

1. Loads generated images and corresponding physical labels.
2. Creates reproducible training, validation and test splits.
3. Calculates normalization statistics using the training data.
4. Applies the same normalization to validation and test data.
5. Stores prepared arrays and metadata for model training.

Generated datasets and prepared NumPy arrays are excluded from version control because they are derived experimental data.

---

## 4. Machine-Learning Pipeline

Olin uses convolutional neural networks to estimate physical parameters from synthetic gravitational-lensing observations.

The experiments progressed from single-parameter regression to multi-output inference.

### CNN₀ — Initial Baseline

CNN₀ establishes the initial regression baseline.

**Architecture**

```text
Input: 160 × 160 × 1
          |
          v
Conv2D — 16 filters, 3×3, ReLU
          |
          v
MaxPooling — 2×2
          |
          v
Conv2D — 32 filters, 3×3, ReLU
          |
          v
MaxPooling — 2×2
          |
          v
Flatten
          |
          v
Dense — 64, ReLU
          |
          v
Dense — 1
          |
          v
Predicted Einstein Radius
```

CNN₀ provides the initial baseline against which subsequent training improvements are evaluated.

### CNN₁ — Improved Regression Model

CNN₁ introduces improved training control and validation-based monitoring.

The training pipeline incorporates early stopping to reduce unnecessary training after validation performance stops improving.

CNN₁ achieved substantially better test performance than CNN₀.

### V2 Multi-Output Regression Experiment

A multi-output regression model was developed to investigate simultaneous estimation of five physical parameters.

The model predicts:

$$
[\theta_E,e_1,e_2,\gamma_1,\gamma_2]
$$

This experiment investigates whether a shared image representation can learn multiple related physical parameters.

### V3 — ResNet-18 Multi-Parameter Inference

V3 uses a ResNet-18 architecture for multi-parameter regression.

It is the current primary inference model.

**Training configuration**

| Setting | Configuration |
|---|---|
| Architecture | ResNet-18 |
| Input dimensions | 160 × 160 × 1 |
| Output parameters | 5 |
| Epochs | 40 |
| Batch size | 32 |
| Initial learning rate | 0.001 |
| Uncertainty estimation | Monte Carlo dropout |
| Monte Carlo passes | 30 |

The model predicts the five physical parameters simultaneously.

Target normalization is used during training and reversed during inference to obtain predictions in physical parameter units.

---

## 5. Experimental Results

### CNN₀ vs CNN₁ — Dataset V0

Both models were evaluated using the same held-out test dataset.

| Metric | CNN₀ | CNN₁ |
|---|---:|---:|
| Test MAE | 0.111396 arcsec | 0.019245 arcsec |
| Test MSE | 0.01388 | 0.00057045 |

CNN₁ achieved:

- Approximately 82.7% reduction in test MAE.
- Approximately 5.8 times lower test MAE than CNN₀.
- Best validation MAE of 0.018489 arcseconds at epoch 9.
- Early stopping after 14 epochs.

These results demonstrate improved parameter estimation under the controlled synthetic V0 experiment. They should not be interpreted as equivalent accuracy on real astronomical observations.

### V3 — Dataset V1 Multi-Parameter Results

The following metrics were obtained on the held-out test dataset.

| Parameter | MAE | R² |
|---|---:|---:|
| Einstein radius ($\theta_E$) | 0.034362 | 0.983177 |
| Ellipticity ($e_1$) | 0.036626 | 0.882142 |
| Ellipticity ($e_2$) | 0.036234 | 0.877282 |
| Shear ($\gamma_1$) | 0.016809 | 0.754748 |
| Shear ($\gamma_2$) | 0.018192 | 0.689337 |

The results indicate that the model performs strongly on Einstein-radius estimation within the synthetic validation domain.

The lower R² values for the shear components also highlight differences in prediction difficulty across physical parameters.

These metrics describe performance on the generated synthetic dataset, not generalization to arbitrary real astronomical images.

---

## 6. Uncertainty Estimation

A point prediction alone does not communicate how variable a model's output is under repeated stochastic inference.

Olin incorporates Monte Carlo dropout to investigate predictive variability.

### Monte Carlo Inference

During inference, the model performs multiple forward passes with dropout enabled.

For each physical parameter, the resulting predictions are used to calculate:

- Mean prediction
- Predictive standard deviation

The current implementation uses 30 Monte Carlo passes.

### Uncertainty Calibration

Calibration factors derived from synthetic validation data are applied to the estimated uncertainty.

The calibration procedure uses a target coverage of approximately 68.27%, corresponding to the nominal one-standard-deviation interval for a Gaussian distribution.

The resulting uncertainty estimates are useful for communicating model variability within the validated synthetic domain.

**Scientific limitation:** The calibration is based on synthetic validation data. It has not established calibrated uncertainty for arbitrary real astronomical observations or distribution shifts.

Uncertainty estimates should therefore be interpreted alongside the domain and limitations of the model.

---

## 7. Physics-Based Reconstruction — FM2

Parameter prediction alone does not provide a complete scientific evaluation of an inverse-inference model.

FM2 uses inferred physical parameters to generate a reconstruction through the established forward model.

### Reconstruction Pipeline

```text
Original Observation
        |
        v
   V3 Inference
        |
        v
 Predicted Parameters
        |
        v
       FM2
        |
        v
Physics-Based Reconstruction
        |
        v
 Reconstruction and Residual Analysis
```

### Reconstruction Process

The reconstruction component:

1. Receives predicted physical parameters.
2. Passes them into the gravitational-lensing model configuration.
3. Renders a deterministic image using the forward model.
4. Compares the reconstructed image against the input observation.
5. Generates residual information for analysis.

The reconstruction does not introduce a separate neural image-generation model.

### Residual Analysis

The image residual is defined as:

$$
R(x,y)=I_{\mathrm{observed}}(x,y)-I_{\mathrm{reconstructed}}(x,y)
$$

Residual analysis helps investigate differences between the observed image and the reconstruction.

A small residual alone does not establish that the inferred physical parameters are correct. Different parameter configurations may potentially produce similar observations.

Reconstruction analysis must therefore be considered alongside parameter-estimation error and the limitations of the forward model.

### Current Reconstruction Pipeline

The inference pipeline integrates:

- Image preprocessing
- V3 model prediction
- Uncertainty estimation
- Physics-based reconstruction
- Residual generation
- Numerical result generation
- Scientific report generation

Synthetic `.npy` observations are used for controlled quantitative validation. Ordinary image inputs are handled as exploratory observational inputs.

---

## 8. Exploratory Real-Observation Evaluation

Olin has been tested using a real astronomical gravitational-lensing observation.

### CASTLES — HE0435-1223

The exploratory evaluation used observational material associated with the HE0435-1223 gravitational lens system from the CASTLES collection.

The experiment investigated whether the existing inference pipeline could process a real astronomical image and produce:

- Physical parameter predictions
- Uncertainty estimates
- A physics-based reconstruction
- Residual information
- A scientific PDF report

This experiment demonstrated that the software pipeline can process an external image, but the resulting reconstruction was not sufficiently convincing to establish reliable physical inference.

### Scientific Limitations

Real astronomical images can differ substantially from the synthetic training observations in:

- Instrument response
- Pixel scale
- PSF characteristics
- Noise distribution
- Image calibration
- Source and lens morphology
- Background and foreground contamination

The current model was trained and calibrated using controlled synthetic observations.

Consequently, real-image predictions are exploratory and must not be interpreted as scientifically validated measurements.

The experiment highlights the importance of domain compatibility and real-observation validation before claiming generalization.

---

## 9. Inference and Scientific Reporting Pipeline

Olin provides an inference pipeline that connects image preprocessing, parameter prediction, reconstruction and report generation.

### Supported Image Inputs

The user-facing image pipeline supports:

- PNG
- JPEG
- WebP

Synthetic `.npy` inputs are reserved for internal controlled validation rather than ordinary image uploads.

### Pipeline Workflow

```text
Input Image
    |
    v
Image Validation
    |
    v
Image Preprocessing
    |
    v
160 × 160 Model Input
    |
    v
V3 Model Inference
    |
    v
Physical Parameter Predictions
    |
    v
Uncertainty Estimation
    |
    v
FM2 Reconstruction
    |
    v
Residual Analysis
    |
    v
Scientific Results
    |
    v
PDF Report Generation
```

### Scientific PDF Report

The reporting component generates a PDF containing relevant analysis outputs, including:

- Input image
- Reconstructed image
- Residual image
- Predicted physical parameters
- Uncertainty information
- Reconstruction metrics
- Relevant inference and reconstruction details
- Limitations and reproducibility information

The pipeline stores generated results in its local results directory during development.

The application is intended to expose reports through protected, ownership-checked retrieval rather than public filesystem access.

---

## 10. Application Architecture — Olin Web Application

The scientific inference pipeline is being integrated into a web application.

The application separates user-facing functionality from the scientific inference implementation.

### Planned Architecture

```text
              User
               |
               v
       HTML / CSS / JavaScript
               |
               v
            FastAPI
               |
       +-------+-------+
       |               |
       v               v
 Authentication   Analysis API
       |               |
       v               v
   PostgreSQL     Inference Pipeline
                       |
                 +-----+-----+
                 |           |
                 v           v
               V3 Model     FM2
                 |           |
                 +-----+-----+
                       |
                       v
                Scientific Report
                       |
                       v
                Protected Retrieval
```

### Application Technologies

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Database | PostgreSQL |
| Frontend | HTML, CSS, JavaScript |
| Scientific inference | TensorFlow / Keras |
| Image processing | Pillow, NumPy |
| Gravitational-lens modelling | lenstronomy |
| PDF generation | ReportLab |
| Version control | Git / GitHub |

### Application Responsibilities

**Authentication**

- Email and password authentication
- Google OAuth
- Email verification
- Password reset
- Access and refresh token handling

**Image analysis**

- Image upload and validation
- Preprocessing
- Model inference
- Uncertainty estimation
- Physics-based reconstruction

**Scientific reporting**

- PDF report generation
- Analysis metadata
- Reproducibility information
- Protected report retrieval

**Data management**

- User and analysis associations
- Report references and metadata
- Ownership-based access control

### Architectural Principles

- Keep the inference pipeline independent of authentication and application logic.
- Load trained model artifacts for inference rather than training during user requests.
- Use PostgreSQL for persistent application metadata.
- Store generated report files separately from database metadata.
- Enforce ownership checks for analysis and report retrieval.
- Keep the initial backend architecture simple and avoid unnecessary distributed infrastructure.

The application is currently under development. Planned application functionality should not be confused with completed scientific inference functionality.

---

## 11. Repository Structure

The repository is organized around the scientific pipeline, experimental outputs, documentation and application implementation.

```text
Olin/
│
├── simulation/
│
├── FM1/
│
├── FM2/
│
├── datasets/
│
├── training/
│   ├── models/
│   └── results/
│
├── inference/
│   └── results/
│
├── real_data/
│
├── docs/
│
├── Documents and updations/
│
├── olin_app/
│
├── README.md
└── .gitignore
```

### Component Responsibilities

**Simulation**

Contains exploratory physical implementations developed to understand gravitational-lensing physics.

**FM1**

Generates synthetic observations from known lens and source parameters.

**FM2**

Provides deterministic physics-based reconstruction using inferred physical parameters.

**Datasets**

Contains locally generated synthetic observations, labels and prepared arrays.

**Training**

Contains model training, dataset preparation and evaluation artifacts.

**Inference**

Contains the production inference, preprocessing, prediction, reconstruction and scientific reporting pipeline.

**Real Data**

Contains exploratory external observational material used for testing. Such material is not automatically included in version control.

**Docs and Documents and updations**

Contain scientific software documentation, experiment reports and project documentation.

**Olin App**

Contains the application backend and database integration.

Generated datasets, model weights, temporary prediction files and other large derived artifacts are excluded from version control where appropriate.

---

## 12. Scientific Design Principles and Limitations

Olin follows several principles throughout development.

1. **Physics before machine learning**

   The physical forward process is established before training an inverse model.

2. **Controlled experiments**

   Initial datasets provide known conditions for evaluating model behaviour.

3. **Known ground truth**

   Synthetic observations provide physical parameters for quantitative evaluation.

4. **Physics-based reconstruction**

   Inferred parameters are passed through an established forward model to investigate the resulting image.

5. **Progressive complexity**

   Additional physical variations and parameter degeneracies are introduced progressively.

6. **Uncertainty awareness**

   Predictions are accompanied by uncertainty estimates, with calibration limitations explicitly acknowledged.

7. **Separation of simulation and inference**

   FM1, training, inference and FM2 have distinct responsibilities.

8. **Scientific honesty**

   Synthetic performance and exploratory real-image results are reported separately.

9. **Depth over feature count**

   The project prioritizes scientific understanding and validation rather than unrelated application features.

### Current Limitations

- The synthetic observation model does not reproduce every property of real astronomical instruments.
- The model's quantitative performance is established primarily within its synthetic training domain.
- Real-observation inference has not yet been validated across a representative collection of systems.
- Uncertainty calibration is based on synthetic validation data.
- Reconstruction similarity alone cannot establish physical correctness.
- Parameter degeneracies and out-of-distribution behaviour require further investigation.

---

## 13. Future Directions

### Scientific Evaluation

- Expand validation using real astronomical observations.
- Investigate observational data compatibility.
- Study parameter degeneracies.
- Evaluate behaviour under distribution shifts.
- Investigate the limitations of the current forward model.

### Possible V2 — Real-Observation Evaluation

A possible future extension is to investigate the Gaia Focused Product Release gravitational-lens search catalogue.

The catalogue may provide a starting point for identifying candidate systems and locating corresponding observational data.

This would require obtaining suitable image observations and checking their compatibility with Olin's physical forward model.

The Gaia catalogue is not currently a direct training dataset for Olin's image-based inference model.

### Application Development

- Complete the FastAPI application.
- Finalize database schema and authentication.
- Integrate the scientific inference pipeline with the API.
- Implement protected analysis and report retrieval.
- Test the complete user workflow.

Future development will prioritize scientific validity, reproducibility and reliable software integration rather than simply increasing model complexity or application features.

---

## 14. Documentation

The project's scientific software requirements, architecture and development considerations are documented in:

[Olin Scientific Software Report](Documents%20and%20updations/Olin_scientific_sw_report.pdf)

The CNN experiment report:

[Experiment 1 — CNN₀ vs CNN₁](Documents%20and%20updations/Experiment%20Reports/Experiment_1_CNN0_vs_CNN1.pdf)

The Dataset V0 reconstruction-validation report:

[Dataset V0 — Reconstruction Validation](docs/dataset_v0_validation.md)

---

## 15. References

### Gravitational-Lensing Modelling

Birrer, S., & Amara, A. (2018). *lenstronomy: Multi-purpose gravitational lens modelling software package*. Physics of the Dark Universe, 22, 189–201.

DOI: https://doi.org/10.1016/j.dark.2018.11.002

### Software

[lenstronomy](https://github.com/lenstronomy/lenstronomy) — Open-source Python software for gravitational-lens modelling and forward simulations.

---

**Olin — Connecting gravitational-lensing physics with machine-learning inverse inference.**
