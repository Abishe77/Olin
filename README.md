# Olin

**A Scientific Machine Learning Project for Inverse Inference in Strong Gravitational Lensing**

Olin explores the use of computer vision and deep learning to infer physical parameters of gravitational lenses from astronomical observations.

The project combines established gravitational-lensing physics, synthetic observation generation, convolutional neural networks, uncertainty estimation, and physics-based image reconstruction into a scientific inference pipeline.

> **Physical Parameters → Gravitational Lensing → Synthetic Observations → Deep Learning Inference → Uncertainty Estimation → Physics-Based Reconstruction → Scientific Reporting**

Unlike a conventional image-classification project, Olin treats deep learning as a tool for investigating an established scientific inverse problem.

---

## Project Status

**Active Development — Scientific Inference Pipeline Completed; Web Application in Progress**

Olin has progressed from controlled synthetic experiments to a multi-parameter inference and reconstruction pipeline with automated scientific reporting.

### Completed

- First-principles gravitational-lensing physics sandbox
- Point-mass lensing and Einstein-radius calculations
- Extended-source modelling
- Distributed-mass lensing experiments
- Singular Isothermal Ellipsoid (SIE) lens modelling
- Elliptical Sérsic source modelling
- External shear
- PSF-based observation modelling
- Photon-counting and detector read-noise simulation
- FM1 synthetic observation generation
- Dataset V0 and V1 generation and preparation
- CNN₀ baseline regression model
- CNN₁ improved regression model
- Multi-parameter inference experiments
- V3 ResNet-18 five-parameter inference model
- Monte Carlo Dropout uncertainty estimation
- Synthetic-domain model evaluation
- FM2 physics-based reconstruction
- Image loading and preprocessing for PNG, JPEG and WebP
- End-to-end inference pipeline
- Reconstruction and residual image generation
- Automated JSON results and scientific PDF report generation

### Current Development

- FastAPI application integration
- PostgreSQL database and persistence
- Email/password authentication
- Google OAuth authentication
- Ownership-based access to analyses and reports
- Frontend image submission and report retrieval

### Scientific Limitations

The model has been evaluated using controlled synthetic observations. Inference on ordinary real astronomical display images remains exploratory.

Synthetic-domain validation does not establish equivalent accuracy or uncertainty calibration on arbitrary real astronomical observations.

---

## 1. Scientific Objective

Gravitational lensing occurs when the gravitational field of a foreground mass distribution alters the apparent position and brightness distribution of light from a background source.

In strong gravitational lensing, this can produce structures such as Einstein rings, arcs, and multiple images.

The physical process can be represented by the lens equation:

\[
\boldsymbol{\beta}
=
\boldsymbol{\theta}
-
\boldsymbol{\alpha}(\boldsymbol{\theta})
\]

where:

- \(\boldsymbol{\beta}\) represents the source-plane position.
- \(\boldsymbol{\theta}\) represents the observed image-plane position.
- \(\boldsymbol{\alpha}\) represents the gravitational deflection angle.

The forward problem uses a known lens mass distribution and source configuration to calculate the resulting lensed image.

Olin investigates the corresponding inverse problem:

> Given an observed lensed image, can a machine-learning model infer the physical parameters responsible for generating it?

The project initially investigated a controlled one-parameter inverse problem using the Einstein radius, \(\theta_E\), as the regression target.

Subsequent experiments introduced additional physical variation and extended the inference task to five parameters.

Olin does not attempt to introduce new gravitational-lensing physics. It uses established physical models to investigate a machine-learning inverse-inference problem.

---

## 2. Physical Forward Model — FM1

FM1 is the synthetic observation generation component of Olin.

It converts physical lens and source parameters into simulated astronomical observations.

The production forward model uses established gravitational-lensing implementations provided by [lenstronomy](https://github.com/lenstronomy/lenstronomy).

### Forward simulation pipeline

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
Instrument / Observation Effects
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

### Physical configuration

**Lens model**

- Singular Isothermal Ellipsoid (SIE)
- External shear

**Background source**

- Elliptical Sérsic brightness profile

**Observation model**

- Fixed pixel scale
- Fixed image field of view
- Gaussian PSF
- Photon-counting noise
- Gaussian detector read noise

The observation model is intentionally controlled. It is not intended to represent a complete telescope or detector simulator.

### Example synthetic observation

The forward model generates an ideal lensed image and applies the observation model to produce a noisy synthetic observation.

![Example of Olin's clean and noisy synthetic observation](assets/dataset_v0_sample.png)

---

## 3. Synthetic Datasets

Synthetic observations provide an important advantage during initial model development: the physical parameters used to generate each image are known.

This allows inferred parameters to be evaluated directly against their ground-truth values.

### Dataset V0 — Controlled one-parameter experiment

The first dataset contains **10,000 synthetic gravitational-lensing observations**.

| Parameter | Configuration |
|---|---|
| Regression target | Einstein radius \(\theta_E\) |
| Einstein-radius range | 1.6″–2.8″ |
| Image dimensions | 160 × 160 |
| Pixel scale | 0.05″ / pixel |
| Field of view | 8″ × 8″ |
| Lens model | SIE + external shear |
| Source model | Elliptical Sérsic |
| PSF | Gaussian |
| Dataset size | 10,000 |
| Training samples | 8,000 |
| Validation samples | 1,000 |
| Test samples | 1,000 |

The initial experiment varies the Einstein radius while holding the remaining physical and observational parameters fixed.

This controlled setup isolates the relationship between the Einstein radius and image morphology before introducing additional physical variations.

### Dataset V1 — Expanded physical variation

Dataset V1 introduces variation across multiple lens and source parameters.

The dataset contains 10,000 synthetic observations, divided into:

- 8,000 training samples
- 1,000 validation samples
- 1,000 test samples

The parameter configuration includes:

| Parameter | Variation |
|---|---|
| Einstein radius | 1.6–2.8 |
| Lens ellipticity \(e_1,e_2\) | Approximately 0.05–0.30 in magnitude |
| External shear \(\gamma_1,\gamma_2\) | Approximately 0.01–0.10 in magnitude |
| Lens centre coordinates | −0.8 to 0.8 |
| Sérsic radius | Varied |
| Sérsic index | Varied |

The dataset uses a fixed random seed for reproducibility.

Normalization statistics are calculated using the training data and applied consistently to the validation and test sets.

### Dataset preparation

The dataset preparation pipeline:

1. Loads generated images and physical labels.
2. Shuffles samples using a reproducible random seed.
3. Creates training, validation and test splits.
4. Calculates normalization statistics using the training set.
5. Applies the same normalization to validation and test data.
6. Saves prepared arrays and metadata for model training.

Generated datasets and prepared NumPy arrays are excluded from version control because they are derived experimental data.

---

## 4. Machine-Learning Pipeline

Olin uses convolutional neural networks to estimate physical parameters from synthetic gravitational-lensing observations.

The model development progressed through controlled regression, expanded parameter estimation and a multi-output residual network.

### CNN₀ — Initial baseline

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

The architecture contains approximately 2.96 million trainable parameters.

**Training configuration**

| Setting | Configuration |
|---|---|
| Optimizer | Adam |
| Loss | Mean Squared Error |
| Evaluation metric | Mean Absolute Error |
| Epochs | 10 |
| Batch size | 32 |
| Random seed | 42 |
| Input | Normalized grayscale observation |
| Target | Einstein radius \(\theta_E\) |

CNN₀ demonstrated that the network could learn a relationship between simulated image morphology and the Einstein radius.

### CNN₁ — Improved regression model

CNN₁ introduced improved training control and validation-based monitoring.

The training pipeline incorporated early stopping to reduce unnecessary training after validation performance stopped improving.

CNN₁ substantially improved test performance over CNN₀.

### V1 — Expanded parameter experiment

The V1 experiment introduced physical variation beyond the controlled V0 dataset.

A single-parameter regression experiment on V1 achieved an Einstein-radius MAE of approximately **0.08818**.

This experiment demonstrated the increased difficulty of inference when additional physical parameters vary.

### V2 — Multi-output regression

V2 extended the inference task to five simultaneous outputs:

- Einstein radius \(\theta_E\)
- Lens ellipticity \(e_1\)
- Lens ellipticity \(e_2\)
- External shear \(\gamma_1\)
- External shear \(\gamma_2\)

Instead of training five independent models, V2 investigated a shared multi-output architecture.

This allowed the network to learn a common image representation while predicting multiple physical parameters.

### V3 — ResNet-18

V3 introduced a ResNet-18 architecture for multi-parameter inference.

The model predicts the five lens parameters simultaneously and serves as the current inference model.

**Training configuration**

| Setting | Configuration |
|---|---|
| Architecture | ResNet-18 |
| Input dimensions | 160 × 160 × 1 |
| Output parameters | 5 |
| Epochs | 40 |
| Batch size | 32 |
| Initial learning rate | 0.001 |
| Uncertainty method | Monte Carlo Dropout |
| Monte Carlo passes | 30 |

The trained model is stored locally at:

```text
training/models/v3_resnet18.keras
```

The model artifact and supporting normalization and calibration files are excluded from version control.

---

## 5. Experimental Results

### CNN₀ vs CNN₁

Both models were evaluated using the same held-out V0 test dataset.

| Metric | CNN₀ | CNN₁ |
|---|---:|---:|
| Test MAE | 0.111396″ | **0.019245″** |
| Test MSE | 0.01388 | **0.00057045** |

CNN₁ achieved approximately:

- **82.7% reduction in test MAE**
- **5.8× lower test MAE** compared with CNN₀

CNN₁ achieved its best validation MAE of **0.018489″ at epoch 9** and stopped training after 14 epochs through early stopping.

These results demonstrate a substantial improvement under the controlled synthetic V0 experiment.

They should not be interpreted as equivalent accuracy on real astronomical observations.

### V3 — Multi-parameter evaluation

V3 was evaluated on the held-out synthetic test data.

| Predicted parameter | Test MAE | R² |
|---|---:|---:|
| Einstein radius \(\theta_E\) | 0.034362 | 0.983177 |
| Ellipticity \(e_1\) | 0.036626 | 0.882142 |
| Ellipticity \(e_2\) | 0.036234 | 0.877282 |
| External shear \(\gamma_1\) | 0.016809 | 0.754748 |
| External shear \(\gamma_2\) | 0.018192 | 0.689337 |

The results show strong performance for Einstein-radius estimation within the synthetic evaluation domain, while the shear components remain more challenging.

The differences in performance also highlight that not all physical parameters are equally identifiable from the simulated image distribution.

### Interpretation

These results establish performance within the generated synthetic domain.

They do not independently establish:

- Generalization to all real astronomical observations.
- Physical identifiability under arbitrary parameter combinations.
- Reliability under unmodelled observational effects.
- Valid uncertainty calibration outside the synthetic validation distribution.

---

## 6. Uncertainty Estimation

Olin incorporates Monte Carlo Dropout to investigate predictive uncertainty.

Instead of performing a single forward pass, the model is evaluated repeatedly with dropout active.

The resulting predictions are used to estimate the spread of model outputs.

### Procedure

```text
Input Observation
       |
       v
Trained V3 Model
       |
       v
Monte Carlo Dropout
       |
       +---- Prediction 1
       +---- Prediction 2
       +---- Prediction 3
       |
       +---- ...
       |
       +---- Prediction 30
       |
       v
Prediction Distribution
       |
       v
Uncertainty Estimation
```

The implementation uses 30 Monte Carlo passes.

Calibration factors are estimated using synthetic validation data to adjust the uncertainty estimates.

### Interpretation and limitations

The resulting uncertainty estimates describe predictive variability under the model and its synthetic calibration procedure.

They should not be interpreted as universally calibrated physical confidence intervals.

In particular, synthetic-domain calibration does not establish reliable uncertainty coverage for arbitrary real astronomical images or observations outside the training distribution.

---

## 7. Physics-Based Reconstruction — FM2

Parameter prediction alone does not provide a complete scientific evaluation of an inverse-inference model.

A model may predict a parameter value, but it is also important to investigate what happens when that inferred parameter is passed back through the physical forward model.

FM2 is Olin's physics-based reconstruction component.

It uses inferred lens parameters to generate a reconstructed image through the established gravitational-lensing model.

### Reconstruction pipeline

```text
Original Observation
        |
        v
    V3 Model
        |
        v
Predicted Lens Parameters
        |
        v
       FM2
        |
        v
Physics-Based Reconstruction
        |
        v
 Reconstructed Image
        |
        v
Residual Analysis
```

### Reconstruction implementation

FM2 reuses the physical model and configuration established in FM1.

It does not duplicate the gravitational-lensing equations or introduce a separate neural reconstruction network.

The reconstruction process:

1. Receives the inferred lens parameters.
2. Passes the parameters into the physical lens configuration.
3. Renders a deterministic image using the forward model.
4. Produces a reconstruction for comparison with the input observation.

The deterministic image-rendering operation avoids introducing another independent detector-noise realization.

### Residual analysis

The image residual is defined as:

\[
R(x,y)=I_{\mathrm{observed}}(x,y)-I_{\mathrm{reconstructed}}(x,y)
\]

Residual analysis helps investigate differences between the observation and the reconstructed image.

A small residual alone does not establish that the inferred physical parameters are correct. Different parameter configurations can potentially produce similar observations.

Reconstruction analysis must therefore be considered alongside direct parameter-estimation error and the assumptions of the physical model.

---

## 8. Application Inference and Scientific Reporting

Olin includes an end-to-end inference pipeline that connects image loading, preprocessing, parameter estimation, uncertainty estimation, reconstruction and scientific reporting.

The application pipeline is implemented independently of the model-training workflow.

### Supported image inputs

Ordinary application inputs:

- PNG
- JPG / JPEG
- WebP

NumPy arrays are retained for internal synthetic validation and are not required from ordinary application users.

### Inference workflow

```text
Uploaded Image
      |
      v
Image Loading
      |
      v
Image Preprocessing
      |
      v
V3 ResNet-18 Model
      |
      v
Five Parameter Predictions
      |
      v
Monte Carlo Dropout
      |
      v
Uncertainty Estimation
      |
      v
Lenstronomy Reconstruction
      |
      v
Reconstruction and Residual Images
      |
      v
JSON Results
      |
      v
Scientific PDF Report
```

### Image preprocessing

The preprocessing pipeline supports common image formats and converts compatible observations into the model's expected input representation.

The pipeline includes:

- Image loading and validation.
- EXIF orientation handling.
- Grayscale conversion.
- Resizing to 160 × 160 pixels.
- Image-specific standardization for ordinary display images.
- Rejection of near-constant images.

Synthetic NumPy observations use the established dataset normalization metadata.

### Inference outputs

The pipeline produces:

- Five predicted lens parameters.
- Predictive uncertainty estimates.
- Reconstructed image.
- Residual image.
- Reconstruction diagnostics.
- Structured JSON results.
- Scientific PDF report.

### Automated scientific PDF reporting

Olin automatically generates a PDF report containing:

- Analysis information.
- Report generation timestamp.
- Predicted physical parameters.
- Uncertainty estimates and interpretation caveats.
- Source parameters used during reconstruction.
- Original input image.
- Physics-based reconstructed image.
- Residual visualization.
- Reconstruction metrics.
- Optimizer information.
- Scientific limitations and interpretation notes.

The PDF is generated programmatically using ReportLab.

### Scientific interpretation of uploaded images

There is an important distinction between synthetic validation and ordinary image inference.

**Synthetic NumPy observations**

These use the established dataset normalization and belong to the controlled synthetic evaluation domain.

**PNG, JPEG and WebP observations**

These undergo image-specific preprocessing. Their pixel scale, point spread function, detector response, background, noise characteristics and scientific calibration may not match the synthetic training configuration.

Consequently, inference on ordinary uploaded images is exploratory.

Uncertainty calibration is based on synthetic validation data. Reconstruction metrics for arbitrary display images are descriptive rather than measures of established physical accuracy.

A visually plausible reconstruction does not independently demonstrate that the inferred parameters are physically correct.

---

## 9. Complete Scientific Workflow

The current scientific workflow connects physical simulation, machine-learning inference, uncertainty estimation and reconstruction.

```text
       PHYSICAL PARAMETERS
                |
                v
               FM1
                |
                v
      Synthetic Observations
                |
                v
       Dataset Preparation
                |
                v
       Train / Validation /
          Test Split
                |
                v
          CNN Training
                |
                v
        V3 ResNet-18
                |
                v
       Five-Parameter Inference
                |
                v
       Uncertainty Estimation
                |
                v
               FM2
                |
                v
       Reconstructed Image
                |
                v
        Residual Analysis
                |
                v
      Scientific Evaluation
```

The application workflow extends this research pipeline:

```text
       User Authentication
                |
                v
        Observation Upload
                |
                v
        Image Validation
                |
                v
       Inference Pipeline
                |
                v
       Results and Report
                |
                v
      Protected Report Access
```

The scientific inference pipeline is implemented independently so that application functionality does not alter the underlying model or physical reconstruction process.

---

## 10. Repository Structure

The repository separates physical simulation, dataset generation, model training, inference and application development.

```text
Olin/
│
├── assets/
│
├── simulation/
│   ├── point_mass.py
│   ├── einstein_ring.py
│   ├── extended_source.py
│   ├── distributed_mass.py
│   ├── sie.py
│   └── sersic.py
│
├── FM1/
│   ├── dataset_generation_forward_model.py
│   ├── generate_dataset.py
│   └── README.md
│
├── FM2/
│   ├── reverse_reconstruction_model.py
│   ├── prepare_validation_data.py
│   ├── select_validation_samples.py
│   ├── plot_validation_examples.py
│   └── test_cnn_inference.py
│
├── training/
│   ├── dataset preparation scripts
│   ├── model training scripts
│   ├── evaluation scripts
│   └── results/
│
├── inference/
│   ├── __init__.py
│   ├── config.py
│   ├── model_loader.py
│   ├── preprocessing.py
│   ├── predictor.py
│   ├── reconstruction.py
│   ├── pipeline.py
│   └── report.py
│
├── olin_app/
│   └── main.py
│
├── datasets/                  # Generated locally; not tracked
│
├── real_data/                 # Exploratory real-observation data
│
├── Documents and updations/
│   ├── Olin_scientific_sw_report.pdf
│   └── Experiment Reports/
│       └── Experiment_1_CNN0_vs_CNN1.pdf
│
├── README.md
└── .gitignore
```

### Component responsibilities

**Simulation**

Contains first-principles exploratory implementations developed to understand gravitational-lensing physics.

**FM1**

Generates synthetic observations from known lens and source parameters using the production forward model.

**Training**

Contains dataset preparation, model training and evaluation scripts.

**FM2**

Provides deterministic physics-based reconstruction using inferred physical parameters.

**Inference**

Connects image preprocessing, trained-model loading, parameter prediction, uncertainty estimation, reconstruction and scientific report generation.

**Olin App**

Contains the FastAPI application layer. It is responsible for authentication, user requests, persistence and protected access to analysis results. It does not perform model training.

**Documents and updations**

Contains scientific software documentation and experiment reports.

Generated datasets, prepared arrays, model weights and temporary prediction files are kept locally and excluded from version control.

---

## 11. Application Architecture

The application is being developed as a lightweight web interface around the existing scientific inference pipeline.

### Planned architecture

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | FastAPI |
| Database | PostgreSQL |
| Database connectivity | psycopg2 |
| Authentication | Email/password and Google OAuth |
| Password hashing | bcrypt |
| Scientific inference | TensorFlow / Keras |
| Physical reconstruction | lenstronomy |
| PDF generation | ReportLab |

### Architectural principles

- The scientific inference pipeline remains independent of authentication.
- The trained model is loaded as a validated artifact.
- Model training does not occur during user requests.
- User data and analysis ownership are persisted in PostgreSQL.
- Users can access only their own analyses and reports.
- Original uploads are treated as temporary processing inputs.
- Generated reports are retrieved through protected application endpoints.
- The initial application uses a single FastAPI backend rather than microservices.

The web application is still under development; these are the intended application components, not a claim that all application features are already operational.

---

## 12. Scientific Design Principles

Olin follows several principles throughout its development.

1. **Physics before machine learning**

   The physical forward process is established and understood before training an inverse model.

2. **Controlled experiments**

   Initial datasets vary physical parameters in a controlled manner to make model behaviour interpretable.

3. **Known ground truth**

   Synthetic observations provide physical parameters for quantitative evaluation.

4. **Physics-based reconstruction**

   Inferred parameters are passed back through an established forward model to examine the resulting reconstruction.

5. **Progressive complexity**

   Additional physical variation, noise, uncertainty and parameter degeneracies are introduced progressively.

6. **Depth over feature count**

   The project prioritizes scientific understanding and evaluation rather than unrelated application features.

7. **Separation of simulation and inference**

   FM1, model training, inference and FM2 have distinct responsibilities.

8. **Scientific honesty**

   Synthetic-domain performance is not presented as proof of equivalent real-observation performance.

9. **No claims of novel cosmological physics**

   Olin uses established gravitational-lensing physics as the scientific foundation for investigating an inverse machine-learning problem.

---

## 13. Future Directions

Future work will prioritize scientific validity and the controlled extension of the inverse problem.

### Scientific modelling

- Introduce additional lens and source parameter variation where scientifically justified.
- Investigate lens-parameter degeneracies.
- Examine the effects of observational variations.
- Evaluate generalization beyond the training distribution.

### Machine learning

- Investigate prediction reliability under distribution shifts.
- Compare additional model configurations only where scientifically justified.
- Examine parameter-specific inference limitations.

### Reconstruction and validation

- Perform systematic residual analysis.
- Quantify reconstruction quality within appropriate validation domains.
- Separate parameter-estimation errors from forward-model limitations.
- Investigate performance on real astronomical observations with appropriate scientific calibration.

### Software development

- Complete authenticated web application integration.
- Implement persistent analysis and report metadata storage.
- Provide secure analysis history and report retrieval.
- Validate application behaviour with end-to-end tests.

Future development will prioritize making the inverse problem scientifically more meaningful rather than simply increasing neural-network complexity.

---

## 14. Technologies

| Area | Technologies |
|---|---|
| Programming | Python |
| Numerical computation | NumPy |
| Data handling | Pandas |
| Visualization | Matplotlib |
| Gravitational-lensing modelling | lenstronomy |
| Deep learning | TensorFlow / Keras |
| Uncertainty estimation | Monte Carlo Dropout |
| Scientific reporting | ReportLab |
| Backend | FastAPI |
| Database | PostgreSQL |
| Database connectivity | psycopg2 |
| Frontend | HTML / CSS / JavaScript |
| Version control | Git / GitHub |

---

## 15. Documentation

The project's scientific software requirements, architecture and development considerations are documented in:

[Olin Scientific Software Report](Documents%20and%20updations/Olin_scientific_sw_report.pdf)

The initial CNN experiment report is available here:

[Experiment 1 — CNN₀ vs CNN₁ (PDF)](Documents%20and%20updations/Experiment%20Reports/Experiment_1_CNN0_vs_CNN1.pdf)

---

## 16. References

### Gravitational-lensing modelling

1. Birrer, S., & Amara, A. (2018). *lenstronomy: Multi-purpose gravitational lens modelling software package*. Physics of the Dark Universe, 22, 189–201.  
   DOI: https://doi.org/10.1016/j.dark.2018.11.002

2. Birrer, S., et al. (2021). *lenstronomy II: A gravitational lensing software ecosystem*. Journal of Open Source Software.

3. Birrer, S. (2024). *Strong gravitational lensing's 'external shear' is not shear*. Monthly Notices of the Royal Astronomical Society.

The lenstronomy documentation recommends citing the relevant software publications when using the package in scientific work.

### Software

- [lenstronomy](https://github.com/lenstronomy/lenstronomy) — Open-source Python software for gravitational-lens modelling and forward simulations.
- [TensorFlow](https://www.tensorflow.org/) — Deep-learning framework.
- [FastAPI](https://fastapi.tiangolo.com/) — Python web framework.
- [PostgreSQL](https://www.postgresql.org/) — Relational database system.

---

**Olin — Connecting gravitational-lensing physics with machine-learning inverse inference.**
