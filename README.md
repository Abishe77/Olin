# Olin

**Olin** is a scientific machine-learning project exploring **inverse inference in strong gravitational lensing**.

The project investigates whether a computer-vision model can infer physical lens parameters from simulated astronomical observations generated from an explicit gravitational-lensing forward model.

> **Physical parameters → gravitational lensing → synthetic observation → CNN inference → physics-based reconstruction**

Olin is designed as a first-principles scientific software project rather than a generic image-classification application. The goal is to connect established lensing physics with a complete machine-learning inference pipeline.

## Project Status

**Early development — forward-model and synthetic-data pipeline**

Current progress:

- [x] First-principles gravitational-lensing sandbox
- [x] Point-mass lensing
- [x] Extended-source modelling
- [x] Distributed-mass lensing experiments
- [x] Singular Isothermal Ellipsoid (SIE) lens model
- [x] Sérsic source model
- [x] External shear
- [x] PSF-based observation modelling
- [x] Photon-counting and detector-noise simulation
- [x] FM1 forward model
- [x] Synthetic dataset generation pipeline
- [x] Initial 100-sample pilot dataset
- [ ] CNN baseline for Einstein-radius regression
- [ ] Large-scale synthetic dataset
- [ ] Uncertainty estimation
- [ ] Degeneracy and generalization experiments
- [ ] Real-observation validation
- [ ] FM2 inference/reconstruction model
- [ ] Web application and scientific reporting

The repository currently focuses on establishing a scientifically controlled forward model and dataset before moving to machine-learning inference.

## Scientific Objective

Gravitational lensing is a forward physical process: the mass distribution of a foreground lens alters the apparent light distribution of a background source.

Olin approaches the inverse problem:

> **Given an observed lensed image, can a machine-learning model infer the physical parameters that generated it?**

The initial experiment focuses on estimating the **Einstein radius** $\theta_E$ as a continuous regression target.

The first controlled dataset varies only $\theta_E$, while lens, source, and observation parameters remain fixed. This isolates the relationship between the Einstein radius and the resulting image before introducing additional sources of ambiguity.

## Forward Model

The current forward pipeline is based on established gravitational-lensing models implemented with `lenstronomy`.

```text
Lens parameters
      +
Source parameters
      ↓
Gravitational lensing
      ↓
Ideal lensed image
      ↓
PSF / observation effects
      ↓
Photon-counting noise
      ↓
Detector read noise
      ↓
Synthetic observation
```

The current lens model combines:

- Singular Isothermal Ellipsoid (SIE)
- External shear

The background source is represented using an elliptical Sérsic profile.

The observation layer currently includes:

- fixed pixel scale
- fixed image field of view
- Gaussian PSF
- photon-counting noise
- Gaussian detector read noise

The observation model is intentionally controlled rather than a complete telescope or detector simulator.

## Machine-Learning Pipeline

The planned inverse-inference pipeline is:

```text
Synthetic observation
        ↓
       CNN
        ↓
 Predicted θ_E
        ↓
      FM2
        ↓
Physics-based reconstruction
        ↓
Comparison with observation
```

The first ML experiment will treat $\theta_E$ as a continuous regression variable.

### Initial V0 configuration

| Parameter | V0 |
|---|---|
| Target | Einstein radius $\theta_E$ |
| Training range | 1.6″ – 2.8″ |
| Image size | 160 × 160 |
| Pixel scale | 0.05″ / pixel |
| Field of view | 8″ × 8″ |
| Lens model | SIE + external shear |
| Source model | Elliptical Sérsic |
| PSF | Gaussian |
| Dataset pilot | 100 observations |

The initial range was selected after visually examining the generated observations and ensuring that the lensing structure remained sufficiently distinguishable for the controlled V0 experiment.

## Dataset Design

Synthetic data is used as the initial training foundation because the forward model provides exact ground-truth parameters.

Each generated observation has a corresponding label:

```text
image → θ_E
```

The generated dataset is intentionally excluded from version control because the image files are derived artifacts rather than source code.

The dataset-generation structure is:

```text
datasets/
└── dataset_v0/
    ├── images/
    │   ├── 000001.npy
    │   ├── 000002.npy
    │   └── ...
    └── labels.csv
```

The repository stores the generator and scientific assumptions, while generated datasets remain local.

## Repository Structure

```text
Olin/
│
├── FM1/
│   ├── dataset_generation_forward_model.py
│   ├── generate_dataset.py
│   └── README.md
│
├── simulation/
│   ├── point_mass.py
│   ├── einstein_ring.py
│   ├── extended_source.py
│   ├── distributed_mass.py
│   ├── sie.py
│   └── sersic.py
│
├── datasets/                  # generated locally, not tracked
│
├── Documents and updations/
│   └── Olin_scientific_sw_report.pdf
│
├── README.md
└── .gitignore
```

### FM1

**FM1** is the dataset-generation forward-model side of Olin.

It converts physical lens/source parameters into synthetic observations and provides the simulation interface used by the dataset-generation script.

### Simulation

The `simulation/` directory contains first-principles exploratory implementations developed to understand the underlying physics before moving to the production modelling layer.

These experiments include point-mass lensing, distributed mass, SIE, and Sérsic source modelling.

They are primarily educational and exploratory rather than the final production pipeline.

## Scientific Design Principles

Olin follows several principles:

1. **Physics before ML**  
   The physical forward process is established before training the inverse model.

2. **Controlled experiments**  
   Initial datasets vary one important parameter at a time so that model behaviour can be interpreted.

3. **Known ground truth**  
   Synthetic observations provide exact physical parameters for quantitative evaluation.

4. **Physics-based validation**  
   Predicted parameters are intended to be passed through a separate inference/reconstruction forward model.

5. **Progressive complexity**  
   Additional physical variation, noise, uncertainty, and parameter degeneracies will be introduced only after the baseline experiment is understood.

6. **No claims of novel cosmological physics**  
   Olin uses established gravitational-lensing physics as the scientific foundation for investigating an inverse machine-learning problem.

## Future Directions

Planned extensions include:

- CNN-based Einstein-radius regression
- larger synthetic datasets
- uncertainty estimation
- parameter-wise evaluation
- controlled lens/source variation
- degeneracy experiments
- out-of-distribution testing
- real astronomical observation validation
- physics-based reconstruction
- scientific PDF reporting
- deployment through the Olin web application

The project will prioritize **depth of the scientific inference pipeline** over adding unrelated application features.

## Technologies

- Python
- NumPy
- Matplotlib
- `lenstronomy`
- PyTorch *(planned for the CNN stage)*
- FastAPI *(planned)*
- PostgreSQL *(planned)*
- Vanilla HTML/CSS/JavaScript *(planned)*

## Documentation

The current scientific software requirements and architecture are documented in:

`Documents and updations/Olin_scientific_sw_report.pdf`

## References

### Gravitational-lensing modelling

1. Birrer, S., & Amara, A. (2018). *lenstronomy: Multi-purpose gravitational lens modelling software package*. **Physics of the Dark Universe, 22**, 189–201.  
   DOI: `10.1016/j.dark.2018.11.002`

2. Birrer, S., et al. (2021). *lenstronomy II: A gravitational lensing software ecosystem*. **Journal of Open Source Software**.

3. Birrer, S. (2024). *Strong gravitational lensing's 'external shear' is not shear*. **Monthly Notices of the Royal Astronomical Society**.

The `lenstronomy` documentation recommends citing the relevant software publications when using the package in scientific work.

### Software

- `lenstronomy` — open-source Python software for modelling strong gravitational lenses and performing forward modelling of lens mass and surface-brightness distributions.
