# Copyright (c) 2023-, Benjamin Rudski, Joseph Deering
#
# This code is licensed under the MIT License. See the `LICENSE` file for
# more details about copying.


"""
Mock vector data creator.

This module provides tools to create artificial vectors for testing.

References
----------
.. [#fisher-lewis-embleton] Fisher, N. I., Lewis, T., & Embleton, B. J.
   J. (1993). Statistical analysis of spherical data ([New ed.], 1.
   paperback ed). Cambridge Univ. Press.
"""
from collections.abc import Collection
from typing import List, Optional, Union

import numpy as np
from scipy.stats import vonmises_fisher

from . import util
from .util import convert_spherical_to_cartesian_coordinates


def create_vonmises_fisher_vectors_single_direction(
    phi: float,
    theta: float,
    kappa: float,
    number_of_points: int,
    magnitude: float = 1.0,
    magnitude_std: float = 0,
    use_degrees: bool = False,
    seed: Optional[int] = None
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
    seed
        Optional seed for random number generation.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(number_of_points, 3)`` containing the generated
        vectors.

    See Also
    --------
    scipy.stats.vonmises_fisher :
        Function used to generate the von Mises-Fisher distribution.

    Notes
    -----
    This function relies on the von Mises-Fisher distribution, which is a
    true probability distribution on the sphere.

    """

    # Convert the mean direction to cartesian coordinates
    mu_spherical = np.array([phi, theta])

    if use_degrees:
        mu_spherical = np.radians(mu_spherical)

    mu = convert_spherical_to_cartesian_coordinates(angular_coordinates=mu_spherical)

    # Generate the von Mises-Fisher distribution
    vmf = vonmises_fisher(mu=mu, kappa=kappa, seed=seed)

    # Sample the distribution
    sampled_points = vmf.rvs(size=number_of_points)

    # Play with the magnitude, if applicable.
    if magnitude_std > 0:
        # Sample the magnitudes from a Gaussian distribution
        magnitudes = np.random.default_rng(seed).normal(
            loc=magnitude, scale=magnitude_std, size=number_of_points
        )

        # Multiply the components by the respective magnitudes
        sampled_points *= magnitudes[:, None]
    else:
        # Rescale the vectors by the specified magnitude
        sampled_points *= magnitude

    # Return the sampled points
    return sampled_points


def convert_args_to_length(
    n: int, *args: Union[float, Collection[float]]
) -> tuple[np.ndarray, ...]:
    """Standardise the length of all arguments.

    Convert the provided numbers or collections of numbers to NumPy arrays
    of a specified length.

    Parameters
    ----------
    n
        The length to which all fields will be standardised.
    *args
        The arguments to convert to arrays of a specified length. Each must
        either be a single value (or a collection of length 1) or a
        collection of length `n`.

    Returns
    -------
    tuple of numpy.ndarray
        The converted values in the same order they were originally passed.
        Any NumPy arrays passed will **not be copied** and the original
        arrays will simply be returned.

    Raises
    ------
    ValueError
        If collections passed in have a length that is not 1 or `n`.
    """

    converted_arguments: List[np.ndarray] = []

    for arg in args:
        # Check to see if we have a collection
        if isinstance(arg, Collection):
            if isinstance(arg, np.ndarray):
                converted_arg = arg
            else:
                converted_arg = np.array(arg)

                if len(converted_arg) == 1:
                    converted_arg = np.tile(converted_arg, n)

            if converted_arg.ndim > 1 or len(converted_arg) != n:
                raise ValueError("The passed arguments must have length 1 or `n`!")
        else:
            converted_arg = np.tile(arg, n)

        converted_arguments.append(converted_arg)

    return tuple(converted_arguments)


def create_von_mises_fisher_vectors_multiple_directions(
    phis: Collection[float],
    thetas: Collection[float],
    kappas: Collection[float],
    numbers_of_vectors: Union[int, Collection[int]] = 1000,
    magnitudes: Union[float, Collection[float]] = 1.0,
    magnitude_stds: Union[float, Collection[float]] = 0.5,
    use_degrees: bool = False,
    seeds: Optional[Collection[int]] = None
) -> np.ndarray:
    """Create vectors drawn from multiple von Mises-Fisher distributions.

    Using the supplied arguments, generate a collection of vectors drawn
    from multiple von Mises-Fisher distributions. These vectors may have
    non-unit magnitude, determined using a Gaussian distribution.

    Parameters
    ----------
    phis
        The set of ``phi`` values for the mean direction.
    thetas
        The set of ``theta`` values for the mean direction.
    kappas
        The set of concentration parameters for the distributions. If a
        single :class:`float` is passed, the same concentration parameter
        will be used for each set of vectors.
    numbers_of_vectors
        Number of vectors to produce for each parameter set. If a single
       :class:`int` is passed, the same number of vectors will be generated
       for each parameter set.
    magnitudes
        The average magnitude of the vectors produced for each parameter
        set. If a single :class:`float` is passed, then the same average
        magnitude is used for all parameter sets.
    magnitude_stds
        The standard deviation of the magnitude for each parameter set. If
        greater than zero, then the magnitudes are drawn from a normal
        distribution. If a single :class:`float` is passed, then the same
        standard deviation is used for all parameter sets.
    use_degrees
        Indicate whether the provided angles are in degrees. If `False`,
        the angles are assumed to be in radians.
    seeds
        Optional seeds for the random number generation for
        reproducibility.


    Returns
    -------
    numpy.ndarray
        The generated vectors drawn from different von Mises-Fisher
        distributions.

    Warnings
    --------
    The array-like arguments must **all** have the same length, unless a
    single value is provided.

    See Also
    --------
    create_vonmises_fisher_vectors_single_direction :
        Function that generates vectors drawn from a single von
        Mises-Fisher distribution.

    """

    # Convert everything to arrays
    phi_array: np.ndarray = np.array(phis)
    theta_array: np.ndarray = np.array(thetas)
    kappa_array: np.ndarray = np.array(kappas)

    seed_array: Optional[np.ndarray]
    if seeds is not None:
        seed_array = np.array(seeds)
    else:
        seed_array = None

    # Get the number of vector families
    number_of_families = len(phi_array)

    # Convert the remaining arguments
    (
        number_of_vectors_array,
        magnitude_array,
        magnitude_std_array,
        seeds_array
    ) = convert_args_to_length(
        number_of_families, numbers_of_vectors, magnitudes, magnitude_stds, seed_array
    )

    # Now, build up the results
    vector_results = [
        create_vonmises_fisher_vectors_single_direction(
            phi_array[i],
            theta_array[i],
            kappa_array[i],
            number_of_vectors_array[i],
            magnitude_array[i],
            magnitude_std_array[i],
            use_degrees,
            seeds_array[i]
        )
        for i in range(number_of_families)
    ]

    all_vectors = np.concatenate(vector_results, axis=0)

    return all_vectors


def generate_watson_distribution(
    mean_direction: np.ndarray, kappa: float, n: int = 100000,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate points from a Watson distribution.

    Simulate a orientations from a Watson distribution using the steps
    presented by Fisher, Lewis and Embleton [#fisher-lewis-embleton]_ in
    section 3.6.2.

    Parameters
    ----------
    mean_direction
        Cartesian coordinates of the mean direction.
    kappa
        Shape parameter of the watson distribution.
    n
        Number of points to generate.
    seed
        Optional seed for the random number generator.

    Returns
    -------
    numpy.ndarray
        Array with `n` rows, corresponding to the 3D Cartesian coordinates
        of the pseudo-randomly generated points.
    """

    # random_vectors = np.zeros((n, 3))

    random_vector_list: List[np.ndarray] = []

    rng = np.random.default_rng(seed)

    for _ in range(n):
        # Check the shape parameter
        s = 0
        if kappa > 0:
            while True:
                # Construct the bipolar distribution
                c = 1 / (np.exp(kappa) - 1)
                u = rng.uniform()
                v = rng.uniform()
                s = (1 / kappa) * np.log(u / c + 1)
                if v <= np.exp(kappa * s * s - kappa * s):
                    break
        else:
            while True:
                # Construct the girdle distribution
                c1 = np.sqrt(np.abs(kappa))
                c2 = np.arctan(c1)
                u = rng.uniform()
                v = rng.uniform()
                s = (1 / c1) * np.tan(c2 * u)

                if v <= (1 - kappa * s * s) * np.exp(kappa * s * s):
                    break

        # Perform the common steps

        # Compute the co-latitude and the longitude - adapt for our
        # definition of phi and theta
        phi = np.arccos(s)
        theta = 2 * np.pi * rng.uniform()

        # Add the new vector spherical angles to the list
        new_vector = np.array([phi, theta])
        random_vector_list.append(new_vector)

    # Convert all new vectors to cartesian coordinates and rotate to mean
    random_vectors = np.stack(random_vector_list, axis=0)
    new_vectors_cartesian = util.convert_spherical_to_cartesian_coordinates(
        random_vectors
    )
    rotated_new_vectors = util.rotate_vectors(
        new_vectors_cartesian, new_pole=mean_direction
    )
    return rotated_new_vectors


def generate_watson_vectors_multiple_directions(
    phis: Collection[float],
    thetas: Collection[float],
    kappas: Collection[float],
    numbers_of_vectors: Union[int, Collection[int]] = 1000,
    use_degrees: bool = False,
    seeds: Optional[Collection[int]] = None
) -> np.ndarray:
    """Create vectors drawn from multiple von Watson distributions.

    Using the supplied arguments, generate a collection of vectors drawn
    from multiple Watson distributions.

    Parameters
    ----------
    phis
        The set of ``phi`` values for the mean direction.
    thetas
        The set of ``theta`` values for the mean direction.
    kappas
        The set of shape parameters for the distributions. If a single
        :class:`float` is passed, the same shape parameter will be used for
        each set of vectors.
    numbers_of_vectors
        Number of vectors to produce for each parameter set. If a single
       :class:`int` is passed, the same number of vectors will be generated
       for each parameter set.
    use_degrees
        Indicate whether the provided angles are in degrees. If `False`,
        the angles are assumed to be in radians.
    seeds
        Optional seeds for the random number generation for
        reproducibility.

    Returns
    -------
    numpy.ndarray
        The generated vectors drawn from different Watson distributions.

    Warnings
    --------
    The array-like arguments must **all** have the same length, unless a
    single value is provided.

    See Also
    --------
    generate_watson_distribution :
        Function that generates vectors drawn from a Watson distribution.
    create_von_mises_fisher_vectors_multiple_directions :
        Similar function for generating vectors from multiple von
        Mises-Fisher distributions.

    """

    # Convert everything to arrays
    phi_array: np.ndarray = np.array(phis)
    theta_array: np.ndarray = np.array(thetas)
    kappa_array: np.ndarray = np.array(kappas)

    seed_array: Optional[np.ndarray]
    if seeds is not None:
        seed_array = np.array(seeds)
    else:
        seed_array = None

    # Get the number of vector families
    number_of_families = len(phi_array)

    # Convert the remaining arguments
    (number_of_vectors_array, seeds_array ) = convert_args_to_length(
        number_of_families,
        numbers_of_vectors,
        seed_array
    )

    # Convert the phi and theta values to Cartesian coordinates
    stacked_angles = np.stack([phi_array, theta_array], axis=-1)

    # Convert the angles to radians, if necessary.
    if use_degrees:
        stacked_angles = np.radians(stacked_angles)

    mean_directions = util.convert_spherical_to_cartesian_coordinates(stacked_angles)

    # Now, build up the results
    vector_results = [
        generate_watson_distribution(
            mean_directions[i],
            kappa_array[i],
            number_of_vectors_array[i],
            seeds_array[i]
        )
        for i in range(number_of_families)
    ]

    all_vectors = np.concatenate(vector_results, axis=0)

    return all_vectors
