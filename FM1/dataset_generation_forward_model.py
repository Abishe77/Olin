import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.psf import PSF


# all observation settings live here
# so changing the instrument means changing only these values
num_pix = 160
pixel_size = 0.05  # arcsec per pixel
fwhm = 0.4  # PSF width in arcsec
photon_scale = 1000  # converts simulated intensity into expected photon counts
read_noise = 1.0  # gaussian detector noise in photon-count units

def border_flux_fraction(img, width=3):
    edge = np.zeros(img.shape, dtype=bool)
    edge[:width, :] = edge[-width:, :] = edge[:, :width] = edge[:, -width:] = True
    return img[edge].sum() / img.sum()


lens_model = LensModel(["SIE", "SHEAR"])  # requesting SIE + external shear
source_model = LightModel(["SERSIC_ELLIPSE"])  # using an elliptical Sersic source


# setting the physical parameters of the SIE lens
# kept as a function so that later we can feed random values into it
def make_lens_kwargs(theta_E=1.6, e1=0.15, e2=0.0,
                     gamma1=0.15, gamma2=0.05,
                     center_x=0.0, center_y=0.0):

    return [{
        "theta_E": theta_E,  # controls the main lensing scale
        "e1": e1,
        "e2": e2,
        # e1 and e2 together describe the lens ellipticity
        # their combination also determines the orientation of the ellipse
        "center_x": center_x,
        "center_y": center_y
    },

    # external shear represents the tidal gravitational influence
    # from mass structures that are not explicitly included in the main lens
    {
        "gamma1": gamma1,
        "gamma2": gamma2
    }]


# setting the physical parameters of the background galaxy
# kept as a function so that later we can randomize the source population
def make_source_kwargs(amp=1.0, R_sersic=0.3, n_sersic=2.0,
                       e1=0.0, e2=0.0,
                       center_x=0.6, center_y=0.3):

    return [{
        "amp": amp,
        "R_sersic": R_sersic,  # overall brightness scale 
        "n_sersic": n_sersic,  # controls how concentrated the source is
        "e1": e1,
        "e2": e2,
        "center_x": center_x,
        "center_y": center_y
    }]


def build_image_model():

    # read noise is added in photon-count units later
    # lenstronomy expects background_rms in the final image units
    background_rms = read_noise / photon_scale

    # pixel (0,0) is half a pixel inside the negative edge
    # this makes the coordinate grid symmetric around (0,0)
    start = -(num_pix - 1) / 2 * pixel_size

    blank = np.zeros((num_pix, num_pix))

    data = ImageData(
        blank,

        transform_pix2angle=np.array([
            [pixel_size, 0],
            [0, pixel_size]
        ]),

        # tells lenstronomy where pixel (0,0) sits in angular coordinates
        ra_at_xy_0=start,
        dec_at_xy_0=start,

        # RMS level used by the noise/likelihood model
        background_rms=background_rms
    )

    # PSF describes how the instrument spreads light
    # gaussian PSF is a simple approximation of telescope/instrument blurring
    psf = PSF(
        psf_type="GAUSSIAN",
        fwhm=fwhm,  # full width at half maximum
        pixel_size=pixel_size
    )

    # combines the coordinate system, PSF, lens model and source model
    # to turn the physical parameters into an image
    image_model = ImageModel(
        data_class=data,
        psf_class=psf,
        lens_model_class=lens_model,
        source_model_class=source_model
    )

    return image_model


def add_noise(clean_image, rng):

    # the clean image has already gone through the lensing and PSF stages
    # noise is added afterwards because this function represents detector measurement

    # convert simulated intensity into expected photon counts
    # poisson sampling cannot work with negative values
    photon_counts = np.clip(clean_image * photon_scale, 0, None)

    # photons arrive as individual random events
    # poisson sampling gives the random fluctuation in the detected photon count
    photon_counts = rng.poisson(photon_counts)

    # the detector itself adds another source of gaussian read noise
    # this noise is independent of the photon-counting process
    photon_counts = photon_counts + rng.normal(
        0,
        read_noise,
        clean_image.shape
    )

    # convert photon counts back into the intensity units used by the image
    return photon_counts / photon_scale


def simulate(image_model, kwargs_lens, kwargs_source, seed=0):

    # using a seeded generator means the same seed produces the same noise
    # this is important when we want reproducible synthetic datasets
    rng = np.random.default_rng(seed)

    # generate the ideal lensed observation after the PSF
    image = image_model.image(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source
    )

    # keep the clean version so we can compare it with the noisy observation
    clean_image = image.copy()

    # add detector-level noise
    noisy_image = add_noise(clean_image, rng)

    return clean_image, noisy_image


if __name__ == "__main__":

    image_model = build_image_model()

    kwargs_lens = make_lens_kwargs()
    kwargs_source = make_source_kwargs()

    clean_image, image = simulate(
        image_model,
        kwargs_lens,
        kwargs_source,
        seed=42
    )

    # sanity check:
    # compare the signal strength with the detector noise level
    print("clean image max:", clean_image.max())
    print("read noise rms:", read_noise / photon_scale)

    # for a ray position at (x,y), calculate the deflection
    # produced by the complete lens model at that position
    alpha_x, alpha_y = lens_model.alpha(
        2.0,
        1.0,
        kwargs_lens
    )

    print("deflection at (2,1):", alpha_x, alpha_y)

    # physical angular extent of the image
    half_fov = num_pix * pixel_size / 2
    extent = [
        -half_fov,
        half_fov,
        -half_fov,
        half_fov
    ]

    # origin="lower" makes the vertical axis increase upward
    # which matches the usual x-y coordinate convention
    plt.subplot(1, 2, 1)
    plt.imshow(
        clean_image,
        origin="lower",
        extent=extent
    )
    plt.title("Clean")

    plt.subplot(1, 2, 2)
    plt.imshow(
        image,
        origin="lower",
        extent=extent
    )
    plt.title("Noisy")

    plt.show()



