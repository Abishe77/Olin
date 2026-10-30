import numpy as np


def rotate_to_lens_frame(x,y,phi):
    x_prime = x * np.cos(phi * (np.pi / 180)) + y * np.sin(phi * (np.pi/180))
    y_prime = -x * np.sin(phi * (np.pi / 180)) + y * np.cos(phi * (np.pi/180))
    return x_prime , y_prime

def elliptical_radius(x_prime,y_prime,q):
    psi = np.sqrt((q**2 * x_prime**2) + (y_prime**2))
    return psi

def sie_deflection_aligned(x_prime, y_prime, theta_e, q):

    if q <= 0 or q > 1:
        raise ValueError("q must satisfy 0 < q <= 1")

    if 1 - q < 1e-6:
        r = np.sqrt(x_prime**2 + y_prime**2)

        if r == 0:
            return 0 , 0

        alpha_x_prime = theta_e * x_prime / r
        alpha_y_prime = theta_e * y_prime / r

        return alpha_x_prime, alpha_y_prime

    e = np.sqrt(1 - q**2)
    psi = elliptical_radius(x_prime, y_prime, q)

    if psi == 0:
        return 0 , 0

    alpha_x_prime = (theta_e * q / e) * np.arctan(
        e * x_prime / psi
    )

    alpha_y_prime = (theta_e * q / e) * np.arctanh(
        e * y_prime / psi
    )

    return alpha_x_prime, alpha_y_prime   

def rotate_from_lens_frame(alpha_x_prime , alpha_y_prime , phi):
    phi = phi * (np.pi / 180)
    alpha_x = alpha_x_prime * np.cos(phi) - alpha_y_prime * np.sin(phi)
    alpha_y = alpha_x_prime * np.sin(phi) + alpha_y_prime * np.cos(phi)

    return alpha_x , alpha_y

def sie_deflection_(x , y , theta_e, q , phi):
    x_prime , y_prime =rotate_to_lens_frame(x , y , phi)
    alpha_x_prime , alpha_y_prime = sie_deflection_aligned(x_prime,y_prime,theta_e,q)
    alpha_x , alpha_y = rotate_from_lens_frame(alpha_x_prime,alpha_y_prime,phi)
    return alpha_x , alpha_y



