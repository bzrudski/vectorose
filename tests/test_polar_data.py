#  test_polar_data.py
#
#  Copyright (c) 2024-, Benjamin Z. Rudski, Joseph Deering
#
#  This code is licensed under the MIT License. See the `LICENSE` file for
#  more details about copying. For a copy of the MIT License, you may
#  also visit https://choosealicense.com/licenses/mit/.

"""Tests for VectoRose polar histogram construction.

This module contains the automated tests for the module
:mod:`vectorose.polar_data`, which provides the polar histogram
construction.
"""

import numpy as np
import vectorose as vr
import vectorose.mock_data

RANDOM_SEED = 20241219


def generate_test_vectors() -> np.ndarray:
    """Generate test vectors for the unit tests."""

    vectors = vr.mock_data.create_vonmises_fisher_vectors_single_direction(
        phi=50,
        theta=60,
        kappa=10,
        number_of_points=100_000,
        magnitude=0.5,
        magnitude_std=0.2,
        use_degrees=True,
        seed=RANDOM_SEED,
    )
    return vectors


def construct_phi_vectors_every_bin(
    polar_discretiser: vr.polar_data.PolarDiscretiser,
) -> np.ndarray:
    """Generate vectors in every phi bin.

    Generate a collection of vectors where one vector lands in each phi bin
    for the provided :class:`.PolarDiscretiser`.

    Parameters
    ----------
    polar_discretiser
        The polar discretiser for which to construct vectors.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 3)`` where ``n`` is the number of phi bins for
        the provided `polar_discretiser`.
    """

    number_of_phi_bins = polar_discretiser.number_of_phi_bins

    bin_increment = polar_discretiser.phi_increment

    lower_bin_values = np.arange(number_of_phi_bins) * bin_increment

    offsets = np.random.default_rng(RANDOM_SEED).uniform(
        low=0, high=bin_increment, size=number_of_phi_bins
    )

    phi = lower_bin_values + offsets

    # Special case scenario for the last bin because of angle restrictions
    max_angle = 90 if polar_discretiser.is_axial else 180
    phi[phi > max_angle] = max_angle

    theta = np.random.default_rng(RANDOM_SEED).uniform(
        low=0, high=360, size=number_of_phi_bins
    )

    spherical_coordinates = np.vstack([phi, theta]).T

    cartesian_coordinates = vr.util.convert_spherical_to_cartesian_coordinates(
        spherical_coordinates, use_degrees=True
    )

    return cartesian_coordinates


def construct_theta_vectors_every_bin(
    polar_discretiser: vr.polar_data.PolarDiscretiser,
) -> np.ndarray:
    """Generate vectors in every theta bin.

    Generate a collection of vectors where one vector lands in each theta bin
    for the provided :class:`.PolarDiscretiser`.

    Parameters
    ----------
    polar_discretiser
        The polar discretiser for which to construct vectors.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 3)`` where ``n`` is the number of theta bins
        for the provided `polar_discretiser`.
    """

    number_of_theta_bins = polar_discretiser.number_of_theta_bins

    bin_increment = polar_discretiser.theta_increment

    lower_bin_values = np.arange(number_of_theta_bins) * bin_increment

    offsets = np.random.default_rng(RANDOM_SEED).uniform(
        low=0, high=bin_increment, size=number_of_theta_bins
    )

    theta = lower_bin_values + offsets

    min_phi = 5

    if polar_discretiser.is_axial:
        max_phi = 90
    else:
        max_phi = 180

    phi = np.random.default_rng(RANDOM_SEED).uniform(
        low=min_phi, high=max_phi, size=number_of_theta_bins
    )

    spherical_coordinates = np.vstack([phi, theta]).T

    cartesian_coordinates = vr.util.convert_spherical_to_cartesian_coordinates(
        spherical_coordinates, use_degrees=True
    )

    return cartesian_coordinates


def construct_axial_polar_discretiser() -> vr.polar_data.PolarDiscretiser:
    """Create an axial polar discretiser for test.

    The constructed polar discretiser will have 18 phi bins and 72 theta
    bins.
    """

    phi_bins = 18
    theta_bins = 72
    is_axial = True

    polar_discretiser = vr.polar_data.PolarDiscretiser(phi_bins, theta_bins, is_axial)

    polar_discretiser.binning_precision = 6

    return polar_discretiser


def construct_vectorial_polar_discretiser() -> vr.polar_data.PolarDiscretiser:
    """Create a vectorial polar discretiser for test.

    The constructed polar discretiser will have 36 phi bins and 72 theta
    bins.
    """

    phi_bins = 36
    theta_bins = 72
    is_axial = False

    polar_discretiser = vr.polar_data.PolarDiscretiser(phi_bins, theta_bins, is_axial)

    polar_discretiser.binning_precision = 6

    return polar_discretiser


def test_number_of_phi_bins_axial():
    """Test the number of phi bins for axial data.

    Test for :prop:`PolarDiscretiser.number_of_phi_bins` for an axial
    discretiser.
    """

    number_of_phi_bins = 10
    number_of_theta_bins = 36
    is_axial = True

    polar_discretiser = vr.polar_data.PolarDiscretiser(
        number_of_phi_bins, number_of_theta_bins, is_axial
    )

    assert polar_discretiser.number_of_phi_bins == number_of_phi_bins


def test_number_of_phi_bins_vectorial():
    """Test the number of phi bins for vectorial data.

    Test for :prop:`PolarDiscretiser.number_of_phi_bins` for a vectorial
    discretiser.
    """

    number_of_phi_bins = 18
    number_of_theta_bins = 36
    is_axial = False

    polar_discretiser = vr.polar_data.PolarDiscretiser(
        number_of_phi_bins, number_of_theta_bins, is_axial
    )

    assert polar_discretiser.number_of_phi_bins == number_of_phi_bins


def test_number_of_theta_bins_axial():
    """Test the number of theta bins for axial data.

    Test for :prop:`PolarDiscretiser.number_of_theta_bins` for an axial
    discretiser.
    """

    number_of_phi_bins = 9
    number_of_theta_bins = 36
    is_axial = True

    polar_discretiser = vr.polar_data.PolarDiscretiser(
        number_of_phi_bins, number_of_theta_bins, is_axial
    )

    assert polar_discretiser.number_of_theta_bins == number_of_theta_bins


def test_number_of_theta_bins_vectorial():
    """Test the number of theta bins for vectorial data.

    Test for :prop:`PolarDiscretiser.number_of_theta_bins` for a vectorial
    discretiser.
    """

    number_of_phi_bins = 18
    number_of_theta_bins = 36
    is_axial = False

    polar_discretiser = vr.polar_data.PolarDiscretiser(
        number_of_phi_bins, number_of_theta_bins, is_axial
    )

    assert polar_discretiser.number_of_theta_bins == number_of_theta_bins


def test_assign_histogram_bins_axial_phi():
    """Test phi polar histogram bin assignment for axial data.

    Test for :meth:`.PolarDiscretiser.assign_histogram_bins` using axial
    data and testing for the phi bin assignment.
    """

    # Create the discretiser
    discretiser = construct_axial_polar_discretiser()

    # Generate a vector in every phi bin
    vectors = construct_phi_vectors_every_bin(discretiser)

    # We expect that the vectors will fall in each sequential bin
    expected_phi_indices = np.arange(discretiser.number_of_phi_bins)

    # Perform the bin assignment
    labelled_vectors = discretiser.assign_histogram_bins(vectors)

    # Check that the number of vectors is indeed the same
    assert len(labelled_vectors) == len(vectors)

    actual_phi_indices = labelled_vectors["phi_bin"].to_numpy()
    assert np.all(actual_phi_indices == expected_phi_indices)


def test_assign_histogram_bins_axial_theta():
    """Test theta polar histogram bin assignment for axial data.

    Test for :meth:`.PolarDiscretiser.assign_histogram_bins` using axial
    data and testing for the theta bin assignment.
    """

    # Create the discretiser
    discretiser = construct_axial_polar_discretiser()

    # Generate a vector in every phi bin
    vectors = construct_theta_vectors_every_bin(discretiser)

    # We expect that the vectors will fall in each sequential bin
    expected_theta_indices = np.arange(discretiser.number_of_theta_bins)

    # Perform the bin assignment
    labelled_vectors = discretiser.assign_histogram_bins(vectors)

    # Check that the number of vectors is indeed the same
    assert len(labelled_vectors) == len(vectors)

    actual_theta_indices = labelled_vectors["theta_bin"].to_numpy()
    assert np.all(actual_theta_indices == expected_theta_indices)


def test_assign_histogram_bins_vectorial_phi():
    """Test phi polar histogram bin assignment for vectorial data.

    Test for :meth:`.PolarDiscretiser.assign_histogram_bins` using
    vectorial data and testing for the phi bin assignment.
    """

    # Create the discretiser
    discretiser = construct_vectorial_polar_discretiser()

    # Generate a vector in every phi bin
    vectors = construct_phi_vectors_every_bin(discretiser)

    # We expect that the vectors will fall in each sequential bin
    expected_phi_indices = np.arange(discretiser.number_of_phi_bins)

    # Perform the bin assignment
    labelled_vectors = discretiser.assign_histogram_bins(vectors)

    # Check that the number of vectors is indeed the same
    assert len(labelled_vectors) == len(vectors)

    actual_phi_indices = labelled_vectors["phi_bin"].to_numpy()

    assert np.all(actual_phi_indices == expected_phi_indices)


def test_assign_histogram_bins_vectorial_theta():
    """Test theta polar histogram bin assignment for vectorial data.

    Test for :meth:`.PolarDiscretiser.assign_histogram_bins` using
    vectorial data and testing for the theta bin assignment.
    """

    # Create the discretiser
    discretiser = construct_vectorial_polar_discretiser()

    # Generate a vector in every phi bin
    vectors = construct_theta_vectors_every_bin(discretiser)

    # We expect that the vectors will fall in each sequential bin
    expected_theta_indices = np.arange(discretiser.number_of_theta_bins)

    # Perform the bin assignment
    labelled_vectors = discretiser.assign_histogram_bins(vectors)

    # Check that the number of vectors is indeed the same
    assert len(labelled_vectors) == len(vectors)

    actual_theta_indices = labelled_vectors["theta_bin"].to_numpy()
    assert np.all(actual_theta_indices == expected_theta_indices)


def test_construct_phi_histogram():
    """Test phi histogram construction.

    Test for :meth:`.PolarDiscretiser.construct_phi_histogram` with
    randomly-generated vectors.
    """

    discretiser = construct_vectorial_polar_discretiser()
    vectors = generate_test_vectors()

    labelled_vectors = discretiser.assign_histogram_bins(vectors)

    # Build the phi histogram
    phi_histogram = discretiser.construct_phi_histogram(labelled_vectors)

    # Check the shape
    number_of_phi_bins = discretiser.number_of_phi_bins

    assert len(phi_histogram) == number_of_phi_bins

    # Sum the values
    summed_values = phi_histogram.sum()

    # Check the frequency
    assert np.isclose(summed_values["frequency"], 1)

    # Check the count
    assert summed_values["count"] == len(vectors)


def test_construct_theta_histogram():
    """Test theta histogram construction.

    Test for :meth:`.PolarDiscretiser.construct_theta_histogram` with
    randomly-generated vectors.
    """

    discretiser = construct_vectorial_polar_discretiser()
    vectors = generate_test_vectors()

    labelled_vectors = discretiser.assign_histogram_bins(vectors)

    # Build the theta histogram
    theta_histogram = discretiser.construct_theta_histogram(labelled_vectors)

    # Check the shape
    number_of_theta_bins = discretiser.number_of_theta_bins

    assert len(theta_histogram) == number_of_theta_bins

    # Sum the values
    summed_values = theta_histogram.sum()

    # Check the frequency
    assert np.isclose(summed_values["frequency"], 1)

    # Check the count
    assert summed_values["count"] == len(vectors)
