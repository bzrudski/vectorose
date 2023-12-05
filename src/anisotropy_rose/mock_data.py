"""
Anisotropy Rose - mock data creator

Joseph Deering, Benjamin Rudski
2023

This module provides tools to create artificial vectors for testing.
"""

import numpy as np

from .core import convert_spherical_to_cartesian_coordinates


def create_vectors_with_primary_orientation(
    phi: float,
    theta: float,
    number_of_vectors: int = 1000,
    phi_std: float = 1.0,
    theta_std: float = 1.0,
    magnitude: float = 1.0,
    magnitude_std: float = 0.5,
    inversion_prob: float = 0.5,
    use_degrees: bool = False,
) -> np.ndarray:
    """
    Create a noisy set of vectors.

    Create a set of noisy vectors with a dominant magnitude and a single
    dominant orientation. Gaussian noise is applied in both angular
    directions and in the magnitude. These vectors are all assumed to be
    located at the origin. This function does not consider any spatial
    distribution.

    For more information on the procedure for converting spherical
    coordinates to cartesian coordinates, see
    ``.core.convert_spherical_to_cartesian_coordinates``.

    :param phi: Dominant phi orientation.
    :param theta: Dominant theta orientation.
    :param number_of_vectors: Number of vectors to produce.
    :param phi_std: Standard deviation of Gaussian noise applied to
        the phi angles.
    :param theta_std: Standard deviation of Gaussian noise applied to
        the theta angles.
    :param magnitude: Average length of the vectors.
    :param magnitude_std: Standard deviation of Gaussian noise applied
        to the magnitude.
    :param inversion_prob: Probability of inverting the sense of the
        vector, following a Bernoulli random variable.
    :param use_degrees: Indicate that angles are in degrees.
    :return: NumPy array with ``number_of_vectors`` rows and three
        columns, containing the respective ``x,y,z`` components of the
        produced vectors.
    """

    # Create the phi angles
    phi_angles = np.random.default_rng().normal(
        loc=phi, scale=phi_std, size=number_of_vectors
    )

    # Create the theta angles
    theta_angles = np.random.default_rng().normal(
        loc=theta, scale=theta_std, size=number_of_vectors
    )

    # Create the magnitudes
    magnitudes = np.random.default_rng().normal(
        loc=magnitude, scale=magnitude_std, size=number_of_vectors
    )

    # Flip some of the directions
    number_of_sites_to_flip = np.random.default_rng().binomial(
        n=number_of_vectors, p=inversion_prob
    )

    all_sites = np.arange(number_of_vectors)

    sites_to_flip = np.random.default_rng().choice(
        all_sites, size=number_of_sites_to_flip, replace=False
    )

    magnitudes[sites_to_flip] *= -1

    # Convert the spherical coordinates to Cartesian coordinates
    angular_coordinates = np.stack([phi_angles, theta_angles], axis=-1)

    if use_degrees:
        # Convert the angles to radians if they've been provided in deg.
        angular_coordinates = np.radians(angular_coordinates)

    cartesian_coordinates = convert_spherical_to_cartesian_coordinates(
        angular_coordinates=angular_coordinates, radius=magnitudes
    )

    return cartesian_coordinates
