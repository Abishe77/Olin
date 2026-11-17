# Olin

**Olin** is a scientific machine-learning application exploring **inverse inference in strong gravitational lensing**.

The project investigates whether a computer-vision model can infer selected physical lens parameters from astronomical observations and examine those estimates through an established gravitational-lensing forward model.

![Olin Galaxy Animation](docs/images/olin-galaxy.gif)

> **Physical parameters → Gravitational lensing → Synthetic observation → Deep-learning inference → Physics-based reconstruction**

Olin connects gravitational-lensing physics with a machine-learning inference pipeline and an integrated web application for analysing compatible astronomical images.

## Project Status

**Olin V1 — Core Scientific Pipeline and Integrated Web Application Implemented**

The first complete application version brings together the scientific inference pipeline, authenticated user workflows, analysis visualisation, and PDF reporting.

### Completed

- First-principles gravitational-lensing experiments.
- Singular Isothermal Ellipsoid (SIE) lens modelling.
- Elliptical Sérsic source modelling.
- External shear.
- PSF-based observation modelling.
- Photon-counting and detector read-noise simulation.
- FM1 synthetic observation generation.
- Synthetic dataset generation and preparation.
- CNN₀ and CNN₁ Einstein-radius regression experiments.
- Multi-output physical parameter inference.
- V3 ResNet-18 model for five-parameter inference.
- Monte Carlo dropout uncertainty estimation.
- Uncertainty calibration using synthetic validation data.
- FM2 physics-based reconstruction.
- Residual generation and analysis.
- Exploratory testing using a real astronomical observation.
- FastAPI backend and PostgreSQL integration.
- User authentication and account workflows.
- Image upload and analysis creation.
- Analysis history and individual analysis viewing.
- Prediction and uncertainty presentation.
- Reconstruction and residual visualisation.
- Scientific PDF report generation and retrieval.
- Integrated frontend and backend workflow.

### Current Development

The core V1 application is implemented. Further work is focused on:

- Deployment preparation.
- Additional integration and security testing.
- Evaluation on a broader range of real astronomical observations.
- Further scientific validation and model generalisation experiments.

## Scientific Objective

Gravitational lensing is a physical process in which the gravitational field of a foreground mass changes the path followed by light from a more distant source.

Olin approaches the inverse problem:

> Given an observed gravitationally lensed image, can a machine-learning model estimate selected physical parameters of the lens system?

The forward problem starts with known physical parameters and generates an observation. The inverse problem attempts to recover selected parameters from that observation.

The gravitational lens equation is:

$$
\boldsymbol{\beta}
=
\boldsymbol{\theta}
-
\boldsymbol{\alpha}(\boldsymbol{\theta})
$$

where:

- $\boldsymbol{\beta}$ represents the source-plane position.
- $\boldsymbol{\theta}$ represents the image-plane position.
- $\boldsymbol{\alpha}(\boldsymbol{\theta})$ represents the deflection angle.

The inverse problem is challenging because different physical configurations can produce similar observed image structures. Olin investigates a constrained version of this problem using supervised deep learning and physics-based reconstruction.

## Forward Model — FM1

The production forward model uses established gravitational-lensing implementations from `lenstronomy`.

The lens model combines:

- Singular Isothermal Ellipsoid (SIE).
- External shear.

The background source is represented by an elliptical Sérsic profile.

The observation pipeline follows:

```text
Lens parameters
      +
Source parameters
      ↓
Gravitational lensing
      ↓
Ideal lensed image
      ↓
PSF convolution
      ↓
Photon-counting noise
      ↓
Detector read noise
      ↓
Synthetic observation
```

The controlled observation model includes:

- Image size: 160 × 160 pixels.
- Pixel scale: 0.05 arcseconds per pixel.
- Field of view: 8 × 8 arcseconds.
- Gaussian PSF with 0.4 arcsecond FWHM.
- Photon-counting noise.
- Gaussian detector read noise.

The observation model is intentionally controlled. It is not intended to represent every instrumental or observational effect present in real astronomical data.

## Synthetic Datasets

Synthetic observations provide known ground-truth parameters, making controlled quantitative evaluation possible.

### Dataset V0

The initial dataset was designed to isolate the relationship between Einstein radius and image morphology.

| Parameter | Configuration |
|---|---|
| Prediction target | Einstein radius ($\theta_E$) |
| Einstein-radius range | 1.6–2.8 arcseconds |
| Image size | 160 × 160 |
| Pixel scale | 0.05 arcseconds/pixel |
| Field of view | 8 × 8 arcseconds |
| Lens model | SIE + external shear |
| Source model | Elliptical Sérsic |
| Dataset size | 10,000 |
| Train / validation / test | 8,000 / 1,000 / 1,000 |
| Random seed | 42 |

### Dataset V1

Dataset V1 introduces additional physical variation to make the inverse problem more challenging.

The dataset contains 10,000 synthetic observations and includes variation in:

- Einstein radius.
- Lens ellipticity components.
- External shear components.
- Lens centre coordinates.
- Sérsic source properties.

The expanded dataset supports multi-parameter inference rather than restricting the model to a single regression target.

Generated datasets and prepared training arrays are excluded from version control.

## Machine-Learning Experiments

Olin was developed through successive experiments, beginning with a controlled single-parameter baseline and progressing towards multi-parameter inference.

### CNN₀ — Initial Baseline

CNN₀ was developed for continuous Einstein-radius regression.

Its architecture consists of convolutional layers, max-pooling, and dense regression layers.

The initial experiment demonstrated that a CNN could learn the relationship between the controlled synthetic observations and the Einstein radius.

However, its fixed training schedule allowed validation performance to deteriorate after the early epochs.

**Test results:**

| Metric | Result |
|---|---:|
| Test MAE | 0.111396 arcseconds |
| Test MSE | 0.01388 |

CNN₀ served as the initial baseline for subsequent training improvements.

### CNN₁ — Improved Training

CNN₁ introduced improved training control and early stopping.

| Metric | Result |
|---|---:|
| CNN₀ test MAE | 0.111396 |
| CNN₁ test MAE | 0.019245 |
| Best validation MAE | 0.018489 |
| Best epoch | 9 |

CNN₁ reduced test MAE by approximately **82.7%** relative to CNN₀.

This experiment established the importance of training control before increasing model complexity.

### V2 — Multi-output Inference

V2 extended the prediction task to five physical parameters:

- Einstein radius ($\theta_E$).
- Lens ellipticity ($e_1$, $e_2$).
- External shear ($\gamma_1$, $\gamma_2$).

Rather than training five independent models, V2 explored a shared multi-output architecture.

### V3 — ResNet-18

V3 uses a ResNet-18 architecture for multi-parameter inference.

The model predicts the five selected physical parameters from a normalised grayscale observation.

**Training configuration:**

| Setting | Value |
|---|---|
| Architecture | ResNet-18 |
| Epochs | 40 |
| Batch size | 32 |
| Initial learning rate | 0.001 |
| Prediction targets | 5 |
| Uncertainty passes | 30 |

**Held-out synthetic test results:**

| Parameter | MAE | R² |
|---|---:|---:|
| $\theta_E$ | 0.034362 | 0.983177 |
| $e_1$ | 0.036626 | 0.882142 |
| $e_2$ | 0.036234 | 0.877282 |
| $\gamma_1$ | 0.016809 | 0.754748 |
| $\gamma_2$ | 0.018192 | 0.689337 |

These metrics describe performance on the held-out synthetic test dataset. They should not be interpreted as measured accuracy on real astronomical observations.

## Uncertainty Estimation

A point prediction alone does not communicate how uncertain a model is about its estimate.

Olin uses Monte Carlo dropout to obtain multiple predictions for an input observation.

The model is evaluated repeatedly with dropout active, producing a distribution of predictions rather than a single output.

The resulting variation is used to estimate predictive uncertainty.

The uncertainty estimates are calibrated using the synthetic validation data. Calibration is intended to improve the relationship between predicted uncertainty and observed prediction errors within the evaluated data distribution.

This calibration does not guarantee equivalent uncertainty coverage under distribution shift or on real astronomical observations.

## Physics-Based Reconstruction — FM2

The machine-learning model estimates physical parameters. FM2 then uses those estimates in a gravitational-lensing forward model to generate a corresponding reconstruction.

```text
Observed astronomical image
          ↓
Image preprocessing
          ↓
     V3 ResNet-18
          ↓
 Five physical parameters
          ↓
 Uncertainty estimation
          ↓
          FM2
          ↓
 Physics-based reconstruction
          ↓
 Comparison with observation
          ↓
        Residual
```

The reconstruction is not an image-enhancement or denoising operation. It is generated using the inferred physical parameters and the forward-model assumptions.

The residual provides a visual representation of the difference between the observation and the reconstructed image.

![Original observation, physics-based reconstruction, and residual](docs/images/reconstruction-comparison.png)

A close reconstruction does not independently establish that the inferred parameters are uniquely correct. Different parameter combinations may produce similar image structures, and the result remains conditional on the assumptions of the forward model.

## Exploratory Real-Observation Testing

Olin's inference pipeline was also tested using the real gravitational-lens observation **HE0435-1223**, obtained from the CASTLES dataset.

This experiment was used to examine whether the preprocessing and inference workflow could be applied to an observational image.

The reconstruction did not provide sufficient qualitative agreement to establish reliable real-observation performance.

This is therefore treated as exploratory pipeline testing, not scientific validation or evidence of generalisation to real astronomical systems.

Further observational evaluation is required before making claims about real-data inference accuracy.

## Web Application

Olin V1 includes an integrated web application that exposes the scientific inference pipeline through an authenticated workflow.

### Application workflow

```text
User authentication
        ↓
Dashboard
        ↓
Upload astronomical image
        ↓
Image preprocessing
        ↓
V3 inference
        ↓
Parameter predictions
        ↓
Uncertainty estimation
        ↓
FM2 reconstruction
        ↓
Residual generation
        ↓
Scientific PDF report
```

### Frontend

The frontend uses:

- HTML.
- CSS.
- Vanilla JavaScript.

The application includes:

- Landing page.
- Registration and login.
- Google OAuth authentication.
- Email verification and OTP workflow.
- Password recovery and reset.
- Dashboard.
- New analysis and image upload.
- Analysis history.
- Individual analysis results.
- Prediction and uncertainty visualisation.
- Reconstruction and residual viewing.
- Report access.

### Backend

The backend uses FastAPI and PostgreSQL.

Implemented capabilities include:

- User authentication and authorisation.
- Access and refresh token workflows.
- Protected analysis endpoints.
- Analysis ownership checks.
- Image upload and processing.
- Analysis metadata persistence.
- PDF report generation and retrieval.
- Rate limiting on protected operations.

### API Overview

The application exposes the following principal analysis and reporting endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/analyses` | Create an analysis |
| GET | `/analyses` | Retrieve the authenticated user's analyses |
| GET | `/analyses/{analysis_id}` | Retrieve an individual analysis |
| POST | `/analyses/{analysis_id}/report` | Generate an analysis report |
| GET | `/reports` | Retrieve available reports |
| GET | `/reports/{report_id}/download` | Download a report |

Access to user-specific resources is protected by authentication and ownership checks.

## Technologies

| Area | Technologies |
|---|---|
| Scientific computing | Python, NumPy |
| Gravitational lens modelling | `lenstronomy` |
| Machine learning | TensorFlow, Keras |
| Computer vision | ResNet-18 |
| Image processing | Pillow |
| Backend | FastAPI |
| Database | PostgreSQL |
| Frontend | HTML, CSS, JavaScript |
| 3D galaxy visualisation | Three.js |
| Scientific reporting | PDF generation |
| Version control | Git, GitHub |

## Repository Structure

The following is a simplified overview of the project:

```text
Olin/
│
├── docs/
│   └── images/
│       ├── olin-galaxy.gif
│       └── reconstruction-comparison.png
│
├── frontend/
│   ├── css/
│   ├── js/
│   ├── pages/
│   ├── index.html
│   └── galaxy-only.html
│
├── olin_app/
│   ├── main.py
│   └── database.py
│
├── inference/
│   ├── config.py
│   ├── preprocessing.py
│   ├── model_loader.py
│   ├── predictor.py
│   └── ...
│
├── FM1/
├── FM2/
├── simulation/
├── training/
├── datasets/                 # Generated locally; not tracked
│
├── Documents and updations/
│   ├── Olin_scientific_sw_report.pdf
│   └── Experiment Reports/
│
├── README.md
└── .gitignore
```

Some generated datasets, trained model weights, and intermediate artifacts are intentionally excluded from version control.

## Scientific Design Principles

Olin follows several principles throughout its development:

1. **Physics before ML** — establish the physical forward process before training the inverse model.
2. **Controlled experiments** — introduce physical variation progressively so that model behaviour can be interpreted.
3. **Known ground truth** — use synthetic observations with known parameters for quantitative evaluation.
4. **Physics-based reconstruction** — examine predicted parameters by passing them through a forward model.
5. **Uncertainty awareness** — communicate uncertainty rather than presenting predictions as exact physical measurements.
6. **Scientific limitations** — distinguish synthetic evaluation from real-observation validation.
7. **Depth over feature count** — prioritise the scientific inference pipeline over unrelated application features.
8. **Established physics** — use established gravitational-lensing theory rather than claiming novel cosmological physics.

## Limitations

Olin V1 is a constrained research and scientific software project. Its results should be interpreted within the following limitations:

- The inference model is trained on synthetic observations generated from a defined forward model.
- The predicted parameters describe the selected lens parameterisation, not an unrestricted recovery of the complete mass distribution.
- Synthetic test metrics do not establish real-observation accuracy.
- Uncertainty calibration on synthetic validation data may not remain reliable under distribution shift.
- Parameter degeneracies can produce similar reconstructed observations.
- The observation model does not reproduce every effect present in real telescope data.
- The exploratory HE0435-1223 test does not constitute scientific validation on real observations.
- Datasets and trained model artifacts are not included in the Git repository. They must be provisioned separately to execute the corresponding inference and training workflows.

## Future Directions

Future work will focus on scientific validation and reliability rather than simply increasing application complexity.

Potential directions include:

- Broader real-observation evaluation.
- Improved handling of observational effects.
- Further parameter-degeneracy experiments.
- Out-of-distribution testing.
- Improved uncertainty evaluation under distribution shift.
- Additional physical variation in synthetic datasets.
- Deployment and operational hardening of the web application.

## Documentation

The project includes additional scientific and development documentation:

- `Documents and updations/Olin_scientific_sw_report.pdf` — scientific software requirements and design.
- `Documents and updations/Experiment Reports/Experiment_1_CNN0_vs_CNN1.pdf` — initial CNN experiment report.
- `docs/dataset_v0_validation.md` — Dataset V0 validation documentation.

## References

### Gravitational-lensing modelling

1. Birrer, S., & Amara, A. (2018). *lenstronomy: Multi-purpose gravitational lens modelling software package*. **Physics of the Dark Universe, 22**, 189–201. https://doi.org/10.1016/j.dark.2018.11.002
2. Birrer, S., et al. (2021). *lenstronomy II: A gravitational lensing software ecosystem*. **Journal of Open Source Software**.
3. Birrer, S. (2024). *Strong gravitational lensing's 'external shear' is not shear*. **Monthly Notices of the Royal Astronomical Society**.

### Software

- [lenstronomy](https://github.com/lenstronomy/lenstronomy) — open-source Python software for gravitational-lens modelling.
- [Three.js](https://threejs.org/) — JavaScript 3D graphics library.

---

**Olin — Gravitational Lensing Inverse Inference**

A physics-grounded machine-learning application for exploring the relationship between gravitationally lensed observations and the physical parameters behind them.
