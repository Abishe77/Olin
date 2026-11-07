import numpy as np

def sample_ellipticity(rng):
    # Sample ellipticity magnitude
    e = rng.uniform(0.05, 0.30)

    # Sample orientation in radians
    phi = rng.uniform(0, np.pi)

    # Convert to Lenstronomy ellipticity components
    e1 = e * np.cos(2 * phi)
    e2 = e * np.sin(2 * phi)

    return e1, e2





 

def sample_shear(rng):

    # Sample shear magnitude
    gamma = rng.uniform(0.01, 0.10)

    # Sample shear orientation
    phi = rng.uniform(0, np.pi)

    # Convert into shear components
    gamma1 = gamma * np.cos(2 * phi)
    gamma2 = gamma * np.sin(2 * phi)

    return gamma1, gamma2   



def sample_source_position(rng):

    center_x = rng.uniform(-0.8, 0.8)
    center_y = rng.uniform(-0.8, 0.8)

    return center_x, center_y    




def sample_source_size(rng):

    R_sersic = rng.uniform(0.15, 0.50)

    return R_sersic 


 

def sample_sersic_index(rng):

    n_sersic = rng.uniform(1.0, 4.0)

    return n_sersic    



def sample_theta_E(rng):

    theta_E = rng.uniform(1.6, 2.8)

    return theta_E 


def sample_parameters(rng):

    # Target parameter
    theta_E = sample_theta_E(rng)

    # Lens ellipticity
    e1, e2 = sample_ellipticity(rng)

    # External shear
    gamma1, gamma2 = sample_shear(rng)

    # Source position
    center_x, center_y = sample_source_position(rng)

    # Source structure
    R_sersic = sample_source_size(rng)
    n_sersic = sample_sersic_index(rng)

    return {
        "theta_E": theta_E,
        "e1": e1,
        "e2": e2,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "center_x": center_x,
        "center_y": center_y,
        "R_sersic": R_sersic,
        "n_sersic": n_sersic
    }


