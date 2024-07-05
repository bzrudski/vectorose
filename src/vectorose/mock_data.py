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
from numbers import Real
from typing import List, Sequence, Union
from collections.abc import Collection

import numpy as np
from scipy.stats import vonmises_fisher

from . import util
from .util import convert_spherical_to_cartesian_coordinates


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

    # Ensure that all arguments have the correct length
    (
        numbers_of_vectors_array,
        phi_std_array,
        theta_std_array,
        magnitude_array,
        magnitude_std_array,
        inversion_probs_array,
    ) = convert_args_to_length(
        number_of_families,
        numbers_of_vectors,
        phi_stds,
        theta_stds,
        magnitudes,
        magnitude_stds,
        inversion_probs,
    )

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
    phi: float,
    theta: float,
    kappa: float,
    number_of_points: int,
    magnitude: float = 1.0,
    magnitude_std: float = 0,
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
    tuple of np.ndarray
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
    create_create_vectors_multiple_orientations :
        Naive, non-directional statistics approach for generating vectors
        with different directions by applying noise in phi and theta
        separately.

    """

    # Convert everything to arrays
    phi_array: np.ndarray = np.array(phis)
    theta_array: np.ndarray = np.array(thetas)
    kappa_array: np.ndarray = np.array(kappas)

    # Get the number of vector families
    number_of_families = len(phi_array)

    # Convert the remaining arguments
    (
        number_of_vectors_array,
        magnitude_array,
        magnitude_std_array,
    ) = convert_args_to_length(
        number_of_families, numbers_of_vectors, magnitudes, magnitude_stds
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
        )
        for i in range(number_of_families)
    ]

    all_vectors = np.concatenate(vector_results, axis=0)

    return all_vectors


def generate_watson_distribution(
    mean_direction: np.ndarray, kappa: float, n: int = 100000
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

    Returns
    -------
    numpy.ndarray
        Array with `n` rows, corresponding to the 3D Cartesian coordinates
        of the pseudo-randomly generated points.
    """

    # random_vectors = np.zeros((n, 3))

    random_vector_list: List[np.ndarray] = []

    for _ in range(n):
        # Check the shape parameter
        s = 0
        if kappa > 0:
            while True:
                # Construct the bipolar distribution
                c = 1 / (np.exp(kappa) - 1)
                u = np.random.default_rng().uniform()
                v = np.random.default_rng().uniform()
                s = (1 / kappa) * np.log(u / c + 1)
                if v <= np.exp(kappa * s * s - kappa * s):
                    break
        else:
            while True:
                # Construct the girdle distribution
                c1 = np.sqrt(np.abs(kappa))
                c2 = np.arctan(c1)
                u = np.random.default_rng().uniform()
                v = np.random.default_rng().uniform()
                s = (1 / c1) * np.tan(c2 * u)

                if v <= (1 - kappa * s * s) * np.exp(kappa * s * s):
                    break

        # Perform the common steps

        # Compute the co-latitude and the longitude - adapt for our
        # definition of phi and theta
        phi = np.arccos(s)
        theta = 2 * np.pi * np.random.default_rng().uniform()

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
