# Olin

## A Scientific Machine Learning Project for Inverse Inference in Strong Gravitational Lensing

Olin explores the use of computer vision and deep learning to infer physical parameters of gravitational lenses from simulated astronomical observations.

The project combines established gravitational-lensing physics, synthetic observation generation, convolutional neural networks, and physics-based image reconstruction into a controlled scientific inference pipeline.

**Physical Parameters → Gravitational Lensing → Synthetic Observations → CNN Inference → Physics-Based Reconstruction**

Unlike a conventional image-classification project, Olin treats deep learning as a tool for investigating an established scientific inverse problem.

---

## Project Status

**Active Development — Dataset V0 Inference and Reconstruction Pipeline Completed**

The current implementation establishes a working synthetic-data-to-inference-to-reconstruction workflow, including an initial quantitative reconstruction-validation experiment.

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
- FM1 synthetic observation generator
- Dataset V0 containing 10,000 labelled observations
- Train/validation/test preparation pipeline
- CNN₀ baseline regression model
- CNN₁ improved regression model
- CNN₀ vs CNN₁ quantitative evaluation
- Parameter-wise error analysis
- FM2 physics-based reconstruction engine
- CNN₁ → FM2 inference and reconstruction integration
- 30-sample stratified reconstruction validation
- Quantitative reconstruction residual analysis
- Visual comparison of predicted and ground-truth reconstructions

### Current Development

- Scientific design of Dataset V1
- Investigation of parameter degeneracies
- Generalization testing under increased physical variation

### Planned

- More realistic lens and source variations
- Uncertainty estimation
- Out-of-distribution evaluation
- Validation using real astronomical observations
- Automated scientific reporting
- Olin web application

The present implementation is a controlled synthetic experiment. It is not yet a validated system for estimating parameters from real astronomical observations.

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

The initial experiment focuses on estimating the **Einstein radius**, \(\theta_E\), as a continuous regression target.

The initial dataset varies only \(\theta_E\), while the remaining lens, source, and observational parameters are held fixed.

This controlled setup isolates the relationship between the Einstein radius and image morphology before introducing additional physical variations and parameter degeneracies.

Olin does not attempt to introduce new gravitational-lensing physics. It uses established physical models to investigate a machine-learning inverse-inference problem.

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

### Current Physical Configuration

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

### Example Synthetic Observation

The forward model generates an ideal lensed image and applies the observation model to produce a noisy synthetic observation.

![Example of Olin's clean and noisy synthetic observation](assets/dataset_v0_sample.png)

---

## 3. Synthetic Dataset

Synthetic observations provide an important advantage during initial model development: the physical parameters used to generate each image are known exactly.

This allows the inferred parameters to be evaluated directly against their ground-truth values.

### Dataset V0

The first dataset contains **10,000 synthetic gravitational-lensing observations**.

| Parameter | Configuration |
|---|---|
| Regression target | Einstein radius \(\theta_E\) |
| Einstein-radius range | 1.6″ – 2.8″ |
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

The initial parameter range was selected after examining the generated observations and ensuring that the lensing structures remained sufficiently distinguishable for the controlled V0 experiment.

### Dataset Preparation

The dataset preparation pipeline:

1. Loads the generated images and corresponding physical labels.
2. Shuffles the samples using a reproducible random seed.
3. Creates training, validation, and test splits.
4. Calculates normalization statistics using the training set.
5. Applies the same normalization to the validation and test sets.
6. Saves the prepared arrays for model training.

The split is performed using a fixed random seed to ensure reproducibility.

Generated images and prepared NumPy arrays are excluded from version control because they are derived experimental data.

---

## 4. Machine-Learning Pipeline

Olin uses convolutional neural networks to estimate the Einstein radius from synthetic gravitational-lensing observations.

Two model versions have been developed.

### CNN₀ — Initial Baseline

CNN₀ establishes the initial regression baseline.

#### Architecture

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

The architecture contains approximately **2.96 million trainable parameters**.

#### Training Configuration

| Setting | Configuration |
|---|---|
| Optimizer | Adam |
| Loss | Mean Squared Error (MSE) |
| Evaluation metric | Mean Absolute Error (MAE) |
| Epochs | 10 |
| Batch size | 32 |
| Random seed | 42 |
| Input | Normalized grayscale observation |
| Target | Einstein radius \(\theta_E\) |

CNN₀ demonstrated that the network could learn a relationship between the simulated image morphology and the Einstein radius.

However, its fixed training schedule allowed validation performance to deteriorate after the early epochs.

It therefore serves as the initial baseline rather than the final model.

### CNN₁ — Improved Regression Model

CNN₁ introduces improved training control and evaluation.

The objective was to address the limitations observed in CNN₀ and obtain more reliable parameter estimates.

The training pipeline incorporates validation-based monitoring and early stopping.

CNN₁ achieved substantially better test performance than the baseline.

---

## 5. CNN₀ vs CNN₁ — Experimental Results

Both models were evaluated using the same held-out test dataset.

### Quantitative Comparison

| Metric | CNN₀ | CNN₁ |
|---|---:|---:|
| Test MAE | 0.111396″ | **0.019245″** |
| Test MSE | 0.01388 | **0.00057045** |

CNN₁ achieved approximately:

- **82.7% reduction in test MAE**
- **5.8× lower test MAE** compared with CNN₀

CNN₁ achieved its best validation MAE of **0.018489″ at epoch 9** and stopped training after 14 epochs through early stopping.

The evaluation also showed that CNN₁ outperformed CNN₀ across the tested Einstein-radius bins.

The results demonstrate a substantial improvement in parameter estimation under the controlled synthetic V0 experiment.

These results should not be interpreted as evidence of equivalent accuracy on real astronomical observations. The initial experiment uses a restricted physical parameter space with most parameters held fixed.

### Evaluation Artifacts

Training and evaluation outputs are available in:

```text
training/
├── results/
│   └── cnn_1/
│       ├── history.json
│       ├── metrics.json
│       ├── cnn_1_learning_curves.png
│       └── test_error_analysis.png
```

The detailed CNN₀ vs CNN₁ experiment report is available here:

[Experiment 1 — CNN₀ vs CNN₁ (PDF)](Documents%20and%20updations/Experiment%20Reports/Experiment_1_CNN0_vs_CNN1.pdf)

---

## 6. Physics-Based Reconstruction — FM2

Parameter prediction alone does not provide a complete scientific evaluation of an inverse-inference model.

A model may predict a parameter value, but it is also important to investigate what happens when that inferred parameter is passed back through the physical forward model.

FM2 is Olin's reverse-inference reconstruction component.

It takes the Einstein radius predicted by CNN₁ and uses the established gravitational-lensing model to generate a reconstructed image.

### Reconstruction Pipeline

```text
Original Synthetic Observation
             |
             v
            CNN₁
             |
             v
    Predicted Einstein Radius
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
      Comparison with Input
```

### FM2 Implementation

FM2 reuses the physical model and configuration established in FM1.

It does not duplicate the gravitational-lensing equations or introduce a separate neural reconstruction network.

The reconstruction process:

1. Receives the predicted Einstein radius.
2. Passes the parameter into the lens configuration.
3. Renders a deterministic image using the forward model.
4. Returns the reconstructed image for comparison.

The reconstruction uses the deterministic image-rendering operation rather than generating a new noisy detector observation.

This allows the reconstructed image to be compared against the original noisy synthetic observation without introducing another independent noise realization.

### Ground-Truth Control Reconstruction

FM2 also generates a second reconstruction using the actual ground-truth Einstein radius.

This creates two independent reconstructions:

- **Predicted reconstruction:** Uses the parameter inferred by CNN₁.
- **Ground-truth reconstruction:** Uses the original parameter used to generate the observation.

The ground-truth reconstruction provides a reference for separating CNN parameter-estimation error from limitations of the forward reconstruction process.

### Residual Analysis

The image residual is defined as:

\[
R(x,y)
=
I_{\mathrm{observed}}(x,y)
-
I_{\mathrm{reconstructed}}(x,y)
\]

Residual analysis helps investigate differences between the original observation and the reconstructed image.

A small residual alone does not establish that the inferred physical parameters are correct, since different parameter configurations can potentially produce similar observations.

For this reason, reconstruction analysis must be considered alongside direct parameter-estimation error.

### Dataset V0 Reconstruction Validation

A stratified validation experiment was conducted using **30 held-out test observations**, with six samples selected from each of five equal-width Einstein-radius bins.

For each observation, FM2 generated:

- A ground-truth reconstruction using the original Einstein radius.
- A predicted reconstruction using the Einstein radius inferred by CNN₁.

The experiment evaluated both direct parameter-estimation error and image residuals against the original noisy observation.

#### Quantitative Results

| Metric | Result |
|---|---:|
| Mean absolute Einstein-radius error | 0.017951″ |
| Median absolute Einstein-radius error | 0.014763″ |
| Mean ground-truth reconstruction MAE | 0.00090402 |
| Mean CNN reconstruction MAE | 0.00090494 |
| Mean ground-truth reconstruction RMSE | 0.00117784 |
| Mean CNN reconstruction RMSE | 0.00118010 |

The CNN-based reconstructions had slightly higher mean residuals than the ground-truth reconstructions in this sample.

The differences were small, but this should not be interpreted as proof that the CNN has recovered the exact physical configuration.

Image residuals are affected by observational noise and by the difference between deterministic model images and noisy observations.

Direct Einstein-radius error therefore remains an important independent evaluation measure.

#### Visual Validation

The validation experiment included comparisons of:

- Original noisy observations
- Ground-truth reconstructions
- CNN-predicted reconstructions
- Ground-truth residual images
- CNN reconstruction residual images

The figure below contains three representative observations: a low-error case, a typical case, and the highest-error case from the selected validation sample.

![Dataset V0 reconstruction validation examples](FM2/results/validation_examples.png)

#### Validation Interpretation

The results establish that:

- CNN₁ predictions can be passed successfully into FM2.
- The physical reconstruction pipeline operates across the selected Einstein-radius range.
- Predicted-parameter reconstructions are numerically comparable to ground-truth reconstructions in this controlled experiment.
- Direct parameter error and image residuals provide complementary evaluation information.

This is an initial integration and reconstruction-validation experiment, not a comprehensive scientific validation.

The experiment does not establish robustness to additional lens or source variations, parameter degeneracies, different observational conditions, or real astronomical observations.

The detailed methodology, selected examples, quantitative results, and limitations are documented in:

[Dataset V0 — Reconstruction Validation](docs/dataset_v0_validation.md)

This completes the initial reconstruction-validation milestone for Dataset V0. Broader tests involving additional physical variation and parameter degeneracies remain future work.

---

## 7. Complete Olin Workflow

The current project architecture connects the physical simulation, machine-learning inference, and reconstruction stages.

```text
       PHYSICAL PARAMETERS
                |
                v
               FM1
                |
                v
      Synthetic Dataset V0
          (10,000 images)
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
          +-----+-----+
          |           |
          v           v
        CNN₀         CNN₁
          |           |
          v           v
       Baseline    Improved
       Inference   Inference
                      |
                      v
                Predicted θ_E
                      |
                      v
                     FM2
                      |
                      v
             Reconstructed Image
                      |
                      v
              Ground-Truth Comparison
                      |
                      v
             Parameter and Residual
                   Analysis
                      |
                      v
              Scientific Validation
```

The project currently focuses on a controlled one-parameter inverse problem.

Future experiments will introduce additional physical variation and investigate the resulting inference ambiguity.

---

## 8. Repository Structure

```text
Olin/
│
├── assets/
│
├── datasets/
│   └── dataset_v0/
│       └── images/
│
├── docs/
│   └── dataset_v0_validation.md
│
├── FM1/
│   └── dataset_generation_forward_model.py
│
├── FM2/
│   ├── reverse_reconstruction_model.py
│   ├── test_cnn1_fm2.py
│   ├── select_validation_samples.py
│   ├── prepare_validation_data.py
│   ├── validate_reconstructions.py
│   ├── plot_validation_examples.py
│   └── results/
│
├── simulation/
│   └── point_mass.py
│
├── training/
│   ├── models/
│   └── results/
│       └── cnn_1/
│
├── Documents and updations/
│   └── Experiment Reports/
│
├── olin_app/
│
├── README.md
└── .gitignore
```

### Component Responsibilities

**Simulation**

Contains first-principles exploratory implementations developed to understand gravitational-lensing physics before moving to the production modelling layer.

**FM1**

Generates synthetic observations from known lens and source parameters using the production forward model.

**FM2**

Provides deterministic physics-based reconstruction using inferred physical parameters, along with validation and residual-analysis scripts.

**Training**

Contains dataset preparation, CNN training, and evaluation scripts.

**Datasets**

Contains locally generated synthetic observations and prepared arrays. Large generated datasets are excluded from version control.

**Docs**

Contains focused technical documentation for individual project milestones and validation experiments.

**Documents and updations**

Contains the project's scientific software documentation and experiment reports.

Generated datasets, prepared arrays, model weights, and temporary prediction files are kept locally and excluded from version control.

---

## 9. Scientific Design Principles

Olin follows several principles throughout its development.

1. **Physics before machine learning**

   The physical forward process is established and understood before training an inverse model.

2. **Controlled experiments**

   Initial datasets vary one important parameter at a time to make model behaviour interpretable.

3. **Known ground truth**

   Synthetic observations provide exact physical parameters for quantitative evaluation.

4. **Physics-based reconstruction**

   Inferred parameters are passed back through an established forward model to examine the resulting reconstruction.

5. **Progressive complexity**

   Additional physical variation, noise, uncertainty, and parameter degeneracies are introduced progressively.

6. **Depth over feature count**

   The project prioritizes scientific understanding and evaluation rather than adding unrelated application features.

7. **Separation of physical simulation and inference**

   FM1, the CNN models, and FM2 have distinct responsibilities.

8. **No claims of novel cosmological physics**

   Olin uses established gravitational-lensing physics as the scientific foundation for investigating an inverse machine-learning problem.

---

## 10. Future Directions

The next development stages focus on increasing the scientific difficulty and realism of the inverse problem.

### Scientific Modelling — Dataset V1

- Define the physical parameters to vary beyond Einstein radius.
- Select justified lens and source parameter ranges.
- Investigate the effects of additional physical variation.
- Design experiments to identify parameter degeneracies.
- Establish validation criteria before generating the next dataset.

### Machine Learning

- Investigate uncertainty estimation.
- Evaluate prediction reliability under distribution shifts.
- Compare additional model configurations only where scientifically justified.

### Reconstruction and Validation

- Evaluate reconstruction quality under increased physical variation.
- Separate parameter-estimation errors from forward-model limitations.
- Investigate performance on real astronomical observations.
- Examine the limitations of inference when multiple physical configurations produce similar observations.

### Software Development

- Automate scientific PDF reporting.
- Develop a web-based interface for image submission and inference.
- Provide reproducible experiment records.

Future development will prioritize making the inverse problem scientifically more meaningful rather than simply increasing neural-network complexity.

---

## 11. Technologies

| Area | Technologies |
|---|---|
| Programming | Python |
| Numerical computation | NumPy |
| Data handling | Pandas |
| Visualization | Matplotlib |
| Gravitational lens modelling | lenstronomy |
| Deep learning | TensorFlow / Keras |
| Version control | Git / GitHub |

### Planned Application Technologies

- FastAPI
- PostgreSQL
- HTML / CSS / JavaScript

These application technologies are planned components, not part of the currently completed inference and reconstruction pipeline.

---

## 12. Documentation

The project's scientific software requirements, architecture, and development considerations are documented in:

[Olin Scientific Software Report](Documents%20and%20updations/Olin_scientific_sw_report.pdf)

The CNN experiment report is available here:

[Experiment 1 — CNN₀ vs CNN₁](Documents%20and%20updations/Experiment%20Reports/Experiment_1_CNN0_vs_CNN1.pdf)

The Dataset V0 reconstruction-validation report is available here:

[Dataset V0 — Reconstruction Validation](docs/dataset_v0_validation.md)

---

## 13. References

### Gravitational-Lensing Modelling

1. Birrer, S., & Amara, A. (2018). *lenstronomy: Multi-purpose gravitational lens modelling software package*. Physics of the Dark Universe, 22, 189–201.

   DOI: https://doi.org/10.1016/j.dark.2018.11.002

2. Birrer, S., et al. (2021). *lenstronomy II: A gravitational lensing software ecosystem*. Journal of Open Source Software.

3. Birrer, S. (2024). *Strong gravitational lensing's 'external shear' is not shear*. Monthly Notices of the Royal Astronomical Society.

The lenstronomy documentation recommends citing the relevant software publications when using the package in scientific work.

### Software

- [lenstronomy](https://github.com/lenstronomy/lenstronomy) — Open-source Python software for gravitational-lens modelling and forward simulations.

---

**Olin — Connecting gravitational-lensing physics with machine-learning inverse inference.**
