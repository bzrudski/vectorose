"""
Anisotropy Rose - mock data creator

Joseph Deering, Benjamin Rudski
2023

This module provides tools to create artificial vectors for testing.
"""
from numbers import Real
from typing import Sequence

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


def create_vectors_multiple_orientations(
    phis: Sequence[float] | np.ndarray,
    thetas: Sequence[float] | np.ndarray,
    numbers_of_vectors: Sequence[int] | np.ndarray | Real = 1000,
    phi_stds: float | Sequence[float] | np.ndarray = 1.0,
    theta_stds: float | Sequence[float] | np.ndarray = 1.0,
    magnitudes: float | Sequence[float] | np.ndarray = 1.0,
    magnitude_stds: float | Sequence[float] | np.ndarray = 0.5,
    inversion_probs: float | Sequence[float] | np.ndarray = 0.5,
    use_degrees: bool = False,
) -> np.ndarray:
    """
    Create a noisy set of vectors with multiple orientations.

    Create a set of noisy vectors with dominant magnitudes and multiple
    dominant orientations. Gaussian noise is applied in both angular
    directions and in the magnitude. These vectors are all assumed to be
    located at the origin. This function does not consider any spatial
    distribution. Arguments aside from phi and theta can be passed as
    sequence or array types in order to have different properties
    for each dominant direction.

    :param phis: Dominant phi orientation.
    :param thetas: Dominant theta orientation.
    :param numbers_of_vectors: Number of vectors to produce.
    :param phi_stds: Standard deviation of Gaussian noise applied to
        the phi angles.
    :param theta_stds: Standard deviation(s) of Gaussian noise applied
        to the theta angles.
    :param magnitudes: Average length(s) of the vectors.
    :param magnitude_stds: Standard deviation(s) of Gaussian noise
        applied to the magnitude.
    :param inversion_probs: Probability of inverting the sense of the
        vectors.
    :param use_degrees: Indicate that angles are in degrees.
    :return: NumPy array with ``sum(numbers_of_vectors)`` rows and three
        columns, containing the respective ``x,y,z`` components of the
        produced vectors.
    """

    # Convert the arguments to NumPy arrays
    phi_array = np.array(phis)
    theta_array = np.array(thetas)

    # Get the number of vector families
    number_of_families = len(phi_array)

    # Convert the remaining parameters to arrays
    arguments = {
        "number_of_vectors": numbers_of_vectors,
        "phi_std": phi_stds,
        "theta_std": theta_stds,
        "magnitude": magnitudes,
        "magnitude_std": magnitude_stds,
        "inversion_prob": inversion_probs,
    }

    corrected_argument_arrays = {}

    for arr_name in arguments:
        arg = arguments[arr_name]

        if isinstance(arg, Real):
            corrected_arr = arg * np.ones(number_of_families)
        else:
            corrected_arr = arg
        corrected_argument_arrays[arr_name] = corrected_arr

    numbers_of_vectors_array = corrected_argument_arrays["number_of_vectors"].astype(
        int
    )
    phi_std_array = corrected_argument_arrays["phi_std"]
    theta_std_array = corrected_argument_arrays["theta_std"]
    magnitude_array = corrected_argument_arrays["magnitude"]
    magnitude_std_array = corrected_argument_arrays["magnitude_std"]
    inversion_probs_array = corrected_argument_arrays["inversion_prob"]

    # And now, we can iterate:
    vector_results = [
        create_vectors_with_primary_orientation(
            phi=phi_array[i],
            theta=theta_array[i],
            number_of_vectors=numbers_of_vectors_array[i],
            phi_std=phi_std_array[i],
            theta_std=theta_std_array[i],
            magnitude=magnitude_array[i],
            magnitude_std=magnitude_std_array[i],
            inversion_prob=inversion_probs_array[i],
            use_degrees=use_degrees,
        )
        for i in range(number_of_families)
    ]

    all_vectors = np.concatenate(vector_results, axis=0)

    return all_vectors
