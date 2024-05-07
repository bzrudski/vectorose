# Copyright (c) 2023-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.


"""
Mock vector data creator.

This module provides tools to create artificial vectors for testing.
"""
from numbers import Real
from typing import Sequence, Union

import numpy as np
from scipy.stats import vonmises_fisher

from .vectorose import convert_spherical_to_cartesian_coordinates


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
    """Create a noisy set of vectors.

    Create a set of noisy vectors with a dominant magnitude and a single
    dominant orientation. Gaussian noise is applied in both angular
    directions and in the magnitude. These vectors are all assumed to be
    located at the origin. This function **does not** consider any spatial
    distribution.

    Parameters
    ----------
    phi
        Dominant :math:`\\phi` orientation.
    
    theta
        Dominant :math:`\\theta` orientation.
    
    number_of_vectors
        Number of vectors to produce.
    
    phi_std
        Standard deviation of Gaussian noise applied to the :math:\\`phi` 
        angles.
    
    theta_std
        Standard deviation of Gaussian noise applied to the :math:`\\theta`
        angles.
    
    magnitude
        Average length of the vectors.
    
    magnitude_std
        Standard deviation of Gaussian noise applied to the magnitude.
    
    inversion_prob
        Probability of inverting the sense of the vector, following a
        Bernoulli random variable.
    
    use_degrees
        Indicate that angles passed for :math:`\\phi` and :math:`\\theta` 
        are in degrees.
    
    Returns
    -------
    numpy.ndarray
        NumPy array with ``number_of_vectors`` rows and three
        columns, containing the respective ``x,y,z`` components of the
        produced vectors.

    See Also
    --------
    .convert_spherical_to_cartesian_coordinates:
        Function used to convert spherical coordinates in 
        :math:`(\\phi, \\theta)` to cartesian :math:`(x,y,z)` coordinates.
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
    phis: Union[Sequence[float], np.ndarray],
    thetas: Union[Sequence[float], np.ndarray],
    numbers_of_vectors: Union[Sequence[int], np.ndarray, Real] = 1000,
    phi_stds: Union[float, Sequence[float], np.ndarray] = 1.0,
    theta_stds: Union[float, Sequence[float], np.ndarray] = 1.0,
    magnitudes: Union[float, Sequence[float], np.ndarray] = 1.0,
    magnitude_stds: Union[float, Sequence[float], np.ndarray] = 0.5,
    inversion_probs: Union[float, Sequence[float], np.ndarray] = 0.5,
    use_degrees: bool = False,
) -> np.ndarray:
    """Create a noisy set of vectors with multiple orientations.

    Create a set of noisy vectors with dominant magnitudes and multiple
    dominant orientations. Gaussian noise is applied in both angular
    directions and in the magnitude. These vectors are all assumed to be
    located at the origin. This function does not consider any spatial
    distribution. Arguments aside from phi and theta can be passed as
    sequence or array types in order to have different properties
    for each dominant direction.

    Parameters
    ----------
    phis
        Collection of dominant :math:`\\phi` orientations.
    
    thetas
        Collection of dominant :math:`\\theta` orientations.
    
    numbers_of_vectors
        Number of vectors to produce for each set of parameters.
    
    phi_stds
        Standard deviation(s) of Gaussian noise applied to the
        :math:`\\phi` angles.
    
    theta_stds
        Standard deviation(s) of Gaussian noise applied to the
        :math:`\\theta` angles.
    
    magnitudes
        Average length of the vectors for each set of parameters.
    
    magnitude_stds
        Standard deviation(s) of Gaussian noise applied to the magnitude.
    
    inversion_probs
        Probability of inverting the sense of the vectors.
    
    use_degrees
        Indicate that angles are in degrees.

    Returns
    -------
    numpy.ndarray
        NumPy array with ``sum(numbers_of_vectors)`` rows and three
        columns, containing the respective ``x,y,z`` components of the
        produced vectors.

    See Also
    --------
    create_vectors_with_primary_orientation:
        function used to create each family of directed vectors for the
        provided parameter sets.
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


def create_vonmises_fisher_vectors_single_direction(
    phi: float, theta: float, kappa: float, number_of_points: int,
    magnitude: float = 1.0, magnitude_std: float = 0,
    use_degrees: bool = False,
) -> np.ndarray:
    """Create a set of vectors using a von Mises-Fisher distribution.

    Draw a set of random orientations from a von Mises-Fisher distribution
    on the unit sphere. The magnitude of these vectors can be modified
    using a normal distribution. These vectors are represented by
    components without any spatial coordinates.

    Parameters
    ----------
    phi
        Mean phi value, where phi reflects the inclination from the
        positive z-axis.
    theta
        Mean theta value, where theta reflects the in-plane angle clockwise
        from the positive y-axis.
    kappa
        Concentration parameter for the von Mises-Fisher distribution.
    number_of_points
        Number of points to draw from the distribution.
    magnitude
        Average magnitude for the computed vectors.
    magnitude_std
        Standard deviation for the magnitude distribution. If this is
        a positive value, the magnitudes follow a Gaussian distribution
        with mean `magnitude`.
    use_degrees
        Indicate whether the angles `phi` and `theta` are provided in
        degrees.

    Returns
    -------
    numpy.ndarray
        Array of shape (`number_of_points`, 3) containing the generated
        vectors.

    See Also
    --------
    scipy.stats.vonmises_fisher :
        Function used to generate the von Mises-Fisher distribution.

    Notes
    -----
    Unlike :func:`.create_vectors_with_primary_orientation`, this function
    relies on the von Mises-Fisher distribution, which is a true
    probability distribution on the sphere. As a result, weird effects
    observed at the poles in data generated with the other method do not
    appear in points generated using this function.

    """

    # Convert the mean direction to cartesian coordinates
    mu_spherical = np.array([phi, theta])

    if use_degrees:
        mu_spherical = np.radians(mu_spherical)

    mu = convert_spherical_to_cartesian_coordinates(angular_coordinates=mu_spherical)

    # Generate the von Mises-Fisher distribution
    vmf = vonmises_fisher(mu=mu, kappa=kappa)

    # Sample the distribution
    sampled_points = vmf.rvs(size=number_of_points)

    # Play with the magnitude, if applicable.
    if magnitude_std > 0:
        # Sample the magnitudes from a Gaussian distribution
        magnitudes = np.random.default_rng().normal(
            loc=magnitude, scale=magnitude_std, size=number_of_points
        )

        # Multiply the components by the respective magnitudes
        sampled_points *= magnitudes[:, None]
    else:
        # Rescale the vectors by the specified magnitude
        sampled_points *= magnitude

    # Return the sampled points
    return sampled_points
