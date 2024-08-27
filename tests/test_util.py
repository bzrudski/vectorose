"""Tests for VectoRose utilities.

This module contains the automated tests for the :mod:`vectorose.util`
module.
"""


import numpy as np
import pytest
import scipy

from vectorose import util


RANDOM_SEED = 20240818


def test_flatten_vector_field():
    """Test the vector field flattening.

    Unit test for :func:`util.flatten_vector_field`.
    """

    # Generate the test vector field
    my_random_vector_field = np.random.default_rng(RANDOM_SEED).random(
        size=(100, 100, 100, 3)
    )

    # Flatten the vector field
    my_flat_vector_field = util.flatten_vector_field(my_random_vector_field)

    # Test to make sure the dimensionality is correct
    assert my_flat_vector_field.ndim == 2

    # Test to make sure the flat and original vector fields contain the
    # same number of elements.
    assert my_flat_vector_field.size == my_random_vector_field.size


def test_remove_zero_vectors():
    """Test the zero-vector removal.

    Unit test for :func:`util.remove_zero_vectors`.
    """

    # Generate random non-zero vectors
    my_random_non_zero_vectors = np.random.default_rng(RANDOM_SEED).uniform(
        low=1e-2, high=1, size=(10000, 3)
    )

    # Generate zero-vectors
    my_zero_vectors = np.zeros_like(my_random_non_zero_vectors)

    # Combine the two sets of vectors
    my_combined_vectors = np.concatenate(
        [my_random_non_zero_vectors, my_zero_vectors], axis=0
    )

    # Remove the zero vectors
    non_zero_vectors_cleaned = util.remove_zero_vectors(my_combined_vectors)

    # Check to make sure the two arrays have the same shape
    assert my_random_non_zero_vectors.shape == non_zero_vectors_cleaned.shape

    # Check that the non-zero vectors all have a magnitude greater than zero
    non_zero_norms = np.linalg.norm(non_zero_vectors_cleaned, axis=-1)
    assert np.all(non_zero_norms > 0)

    # Check that the non-zero vectors are all equal to the originals
    assert np.all(my_random_non_zero_vectors == non_zero_vectors_cleaned)


def test_normalise_array_flat():
    """Test the array normalisation.

    Unit test for :func:`util.normalise_array` for a flat input.
    """

    # Generate a random array.
    my_random_array = np.random.default_rng(RANDOM_SEED).random(size=1000)

    # Normalise the array
    normalised_array = util.normalise_array(my_random_array)

    # Now, check to see if the entries sum to one
    sum_of_entries = normalised_array.sum()
    assert sum_of_entries == 1

    # Compute the original sum and multiply the normalised version by it.
    sum_of_originals = my_random_array.sum()

    assert np.all(np.isclose(normalised_array * sum_of_originals, my_random_array))


def test_normalise_array_2d():
    """Test the array normalisation.

    Unit test for :func:`util.normalise_array` for a 2D array.
    """

    # Generate a random array.
    my_random_array = np.random.default_rng(RANDOM_SEED).random(size=(1000, 3))

    # Normalise the array
    normalised_array = util.normalise_array(my_random_array, axis=-1)

    # Now, check to see if the entries sum to one
    sum_of_entries = normalised_array.sum(axis=-1)
    assert np.all(np.isclose(sum_of_entries, 1))

    # Compute the original sum and multiply the normalised version by it.
    sum_of_originals = my_random_array.sum(axis=-1)[:, None]

    assert np.all(np.isclose(sum_of_originals * normalised_array, my_random_array))


def test_normalise_vectors_no_zero():
    """Test the vector normalisation.

    Unit test for :func:`util.normalise_vectors`.

    Warnings
    --------
    This test only verifies the normalisation. As the magnitude is computed
    using :func:`numpy.linalg.norm`, we trust that the NumPy team has
    tested that behaviour.
    """

    # Generate random vectors
    my_random_vectors = np.random.default_rng(RANDOM_SEED).integers(
        low=2, high=100, size=(1000, 3)
    )

    # Normalise the vectors
    normalised_vectors, magnitudes = util.normalise_vectors(my_random_vectors)

    # Check the magnitudes of the normalised vectors
    normalised_vector_norms = np.linalg.norm(normalised_vectors, axis=-1)
    assert np.all(np.isclose(normalised_vector_norms, 1))


def test_normalise_vectors_with_zeros():
    """Test the vector normalisation (with zero vectors).

    Unit test for :func:`util.normalise_vectors` including inputs that have
    zero magnitude.
    """
    # Generate non-zero random vectors
    my_random_vectors = np.random.default_rng(RANDOM_SEED).integers(
        low=2, high=100, size=(1000, 3)
    )

    # Generate zero vectors
    my_zero_vectors = np.zeros_like(my_random_vectors)

    my_total_vectors = np.concatenate(
        [my_random_vectors, my_zero_vectors], axis=0
    )

    # Normalise the vectors
    normalised_vectors, magnitudes = util.normalise_vectors(
        my_total_vectors
    )

    # Check the magnitudes of the normalised vectors
    normalised_vector_norms = np.linalg.norm(normalised_vectors, axis=-1)

    # Check to make sure that there are the same number of normalised
    # vectors as non-normalised vectors
    assert normalised_vectors.shape == my_total_vectors.shape

    # Now, check the numbers of non-zero and zero vectors
    non_zero_norms = normalised_vector_norms[normalised_vector_norms > 0]
    zero_norms = normalised_vector_norms[normalised_vector_norms == 0]

    assert len(non_zero_norms) == len(my_random_vectors)
    assert len(zero_norms) == len(my_zero_vectors)

    # Finally, check to make sure all the non-zero norms are indeed one
    assert np.all(np.isclose(non_zero_norms, 1))


def test_generate_representative_unit_vectors_magnitudes():
    """Test the representative unit vector field generation.

    Test :func:`util.generate_representative_unit_vectors` to verify that
    all vectors have unit magnitude.
    """

    # Generate random vectors
    my_random_vectors = np.random.default_rng(RANDOM_SEED).integers(
        low=2, high=100, size=(1000, 3)
    )

    # Generate the representative unit vectors
    my_unit_vectors = util.generate_representative_unit_vectors(my_random_vectors, 3000)

    # Check the magnitudes
    magnitudes = np.linalg.norm(my_unit_vectors, axis=-1)
    assert np.all(np.isclose(magnitudes, 1))


def test_convert_vectors_to_axes():
    """Test the vector to axis conversion.

    Unit test for :func:`util.convert_vectors_to_axes` to ensure that after
    conversion, all z-component values are non-zero.
    """

    # Generate random vectors
    my_random_vectors = scipy.stats.uniform_direction(dim=3, seed=RANDOM_SEED).rvs(1000)

    # Ensure that there are vectors in the bottom hemisphere
    if not np.any(my_random_vectors[:, -1] < 0):
        indices_to_flip = np.random.default_rng(RANDOM_SEED).choice(
            np.arange(1000), size=300, replace=False,
        )

        my_random_vectors[tuple(indices_to_flip)] *= -1

    # Convert the vectors to axes
    my_axes = util.convert_vectors_to_axes(my_random_vectors)

    # Check to make sure the shape is the same
    assert my_axes.shape == my_random_vectors.shape

    # Check to make sure the norms are the same
    assert np.all(np.linalg.norm(my_axes, axis=-1) == np.linalg.norm(my_random_vectors, axis=-1))

    # Check to make sure that all the z-coordinates are non-negative
    assert np.all(my_axes[:, -1] >= 0)


def test_create_symmetric_vectors_from_axes():
    """Test the axis to symmetric vector conversion.

    Unit test for :func:`util.create_symmetric_vectors_from_axes` to ensure
    that the symmetric vector list contains twice the number of vectors as
    the original axes list, and that the order is preserved.
    """

    # Create random non-unit axes
    my_random_axes = np.random.default_rng(RANDOM_SEED).uniform(
        low=[-1, -1, 0], high=[1, 1, 1], size=(1000, 3)
    )

    # Convert these axes into vectors
    my_vectors = util.create_symmetric_vectors_from_axes(my_random_axes)

    # Check the shape of the output
    n_axes, d_axes = my_random_axes.shape
    n_vectors, d_vectors = my_vectors.shape

    assert n_vectors == 2 * n_axes and d_axes == d_vectors

    # Check to see that the order is preserved
    assert np.all(my_vectors[:n_axes] == my_random_axes)
    assert np.all(my_vectors[n_axes:] == - my_random_axes)


def test_convert_spherical_to_cartesian_coordinates():
    """Test the conversion from spherical to Cartesian coordinates.

    Unit test for :func:`util.convert_spherical_to_cartesian_coordinates`.
    """

    # Create some simple angles
    input_angles = np.radians([
        [0, 0],
        [180, 0],
        [90, 0],
        [90, 180],
        [90, 90],
        [90, 270]
    ])

    expected_output = np.array([
        [0, 0, 1],
        [0, 0, -1],
        [0, 1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [-1, 0, 0]
    ])

    # Convert the angles to Cartesian coordinates
    my_cartesian_coordinates = util.convert_spherical_to_cartesian_coordinates(
        input_angles, radius=1
    )

    # Compare to expected output
    assert np.all(np.isclose(my_cartesian_coordinates, expected_output))


def test_compute_vector_orientation_angles():
    """Test spherical coordinate calculation.

    Unit test for :func:`util.compute_vector_orientation_angles`.
    """

    # Create the test input vectors and output angles
    input_vectors = np.array([
        [0, 0, 1],
        [0, 0, -1],
        [0, 1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [-1, 0, 0]
    ])

    expected_output = np.radians([
        [0, 0],
        [180, 0],
        [90, 0],
        [90, 180],
        [90, 90],
        [90, 270]
    ])

    # Run the conversion
    computed_spherical_coords = util.compute_vector_orientation_angles(
        input_vectors
    )

    # Check the output
    assert np.all(computed_spherical_coords == expected_output)


def test_convert_to_math_spherical_coordinates():
    """Test conversion from custom spherical coordinates to FLE system.

    Unit test for :func:`util.convert_to_math_spherical_coordinates`.
    """

    # Define our input angles
    input_angles = np.radians([
        [0, 0],
        [180, 0],
        [90, 0],
        [45, 45],
        [90, 180],
        [90, 90],
        [90, 270]
    ])

    # Define the expected outputs
    expected_outputs = np.radians([
        [90, 0],
        [90, 180],
        [90, 90],
        [45, 45],
        [270, 90],
        [0, 90],
        [180, 90]
    ])

    # And now perform the conversion
    converted_angles = util.convert_to_math_spherical_coordinates(
        input_angles, use_degrees=False
    )

    # And check the equality
    assert np.all(converted_angles == expected_outputs)

def test_convert_to_vectorose_spherical_coordinates():
    """Test conversion from FLE system to the custom spherical coordinates.

    Unit test for
    :func:`util.convert_math_spherical_coordinates_to_vr_coordinates`.
    """

    # Define our input angles
    input_angles = np.radians([
        [90, 0],
        [90, 180],
        [90, 90],
        [45, 45],
        [270, 90],
        [0, 90],
        [180, 90]
    ])

    # Define the expected outputs
    expected_outputs = np.radians([
        [0, 0],
        [180, 0],
        [90, 0],
        [45, 45],
        [90, 180],
        [90, 90],
        [90, 270]
    ])

    # And now perform the conversion
    converted_angles = util.convert_math_spherical_coordinates_to_vr_coordinates(
        input_angles, use_degrees=False
    )

    # And check the equality
    assert np.all(converted_angles == expected_outputs)


def test_perform_binary_search():
    """Test the custom binary search implementation.

    Unit test for :func:`util.perform_binary_search`.
    """

    # Create a list
    my_list = [1, 2, 3, 4, 6, 7]

    # Search for an element in the list
    my_element = 3
    my_index = util.perform_binary_search(my_list, my_element)

    # Check the index
    assert my_index == 2

    # Search for an element not in the list
    my_element = 5
    my_index = util.perform_binary_search(my_list, my_element)

    # Check the index
    assert my_index == 3
