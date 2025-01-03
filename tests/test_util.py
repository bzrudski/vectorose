"""Tests for VectoRose utilities.

This module contains the automated tests for the :mod:`vectorose.util`
module.
"""


import numpy as np
import pytest
import scipy

import vectorose as vr


RANDOM_SEED = 20240818


@pytest.fixture
def random_vectors() -> np.ndarray:
    """Generate random vectors."""

    my_random_non_zero_vectors = np.random.default_rng(RANDOM_SEED).uniform(
        low=1e-2, high=1, size=(10000, 3)
    )

    return my_random_non_zero_vectors


@pytest.fixture
def random_vectors_with_locations(random_vectors: np.ndarray) -> np.ndarray:
    """Generate random vectors with spatial locations."""

    # Generate random locations
    locations = np.random.default_rng(RANDOM_SEED * 2).uniform(
        low=1e-2, high=1, size=random_vectors.shape
    )

    # Slide in the locations
    my_random_non_zero_vectors = np.concatenate([locations, random_vectors], axis=-1)

    return my_random_non_zero_vectors


def test_convert_vectors_to_data_frame_no_locations(random_vectors):
    """Test conversion to DataFrame.

    Unit test for :func:`util.convert_vectors_to_data_frame` without
    spatial locations.
    """

    vectors_df = vr.util.convert_vectors_to_data_frame(random_vectors)
    vectors_df_array = vectors_df.to_numpy()

    assert "vx" in vectors_df
    assert "vy" in vectors_df
    assert "vz" in vectors_df

    assert np.all(vectors_df_array == random_vectors)


def test_convert_vectors_to_data_frame_with_locations(random_vectors_with_locations):
    """Test conversion to DataFrame.

    Unit test for :func:`util.convert_vectors_to_data_frame` with spatial
    locations.
    """

    vectors_df = vr.util.convert_vectors_to_data_frame(random_vectors_with_locations)
    vectors_df_array = vectors_df.to_numpy()

    assert "x" in vectors_df
    assert "y" in vectors_df
    assert "z" in vectors_df
    assert "vx" in vectors_df
    assert "vy" in vectors_df
    assert "vz" in vectors_df

    assert np.all(vectors_df_array == random_vectors_with_locations)


def test_magnitude_computation():
    """Test the magnitude function.

    Unit test for :func:`util.compute_vector_magnitudes`.
    """

    # Define two vectors
    vectors = np.array([[0, 1, 0], [3, 4, 5]])

    expected_magnitudes_3d = [1, 5 * np.sqrt(2)]
    expected_magnitudes_2d = [1, 5]

    magnitudes = vr.util.compute_vector_magnitudes(vectors)

    # Check the 3D magnitudes
    computed_magnitudes_3d = magnitudes[:, vr.util.MagnitudeType.THREE_DIMENSIONAL]

    assert np.all(np.isclose(computed_magnitudes_3d, expected_magnitudes_3d))

    # Check the in-plane magnitudes
    computed_magnitudes_2d = magnitudes[:, vr.util.MagnitudeType.IN_PLANE]

    assert np.all(np.isclose(computed_magnitudes_2d, expected_magnitudes_2d))


def test_magnitude_computation_one_vector():
    """Test the magnitude function.

    Unit test for :func:`util.compute_vector_magnitudes` for a single
    vector.
    """

    # Define two vectors
    vectors = np.array([3, 4, 5])

    expected_magnitudes_3d = 5 * np.sqrt(2)
    expected_magnitudes_2d = 5

    magnitudes = vr.util.compute_vector_magnitudes(vectors)

    # Check the 3D magnitudes
    computed_magnitudes_3d = magnitudes[vr.util.MagnitudeType.THREE_DIMENSIONAL]

    assert np.isclose(computed_magnitudes_3d, expected_magnitudes_3d)

    # Check the in-plane magnitudes
    computed_magnitudes_2d = magnitudes[vr.util.MagnitudeType.IN_PLANE]

    assert np.isclose(computed_magnitudes_2d, expected_magnitudes_2d)


def test_flatten_vector_field():
    """Test the vector field flattening.

    Unit test for :func:`util.flatten_vector_field`.
    """

    # Generate the test vector field
    my_random_vector_field = np.random.default_rng(RANDOM_SEED).random(
        size=(100, 100, 100, 3)
    )

    # Flatten the vector field
    my_flat_vector_field = vr.util.flatten_vector_field(my_random_vector_field)

    # Test to make sure the dimensionality is correct
    assert my_flat_vector_field.ndim == 2

    # Test to make sure the flat and original vector fields contain the
    # same number of elements.
    assert my_flat_vector_field.size == my_random_vector_field.size


def test_remove_zero_vectors(random_vectors):
    """Test the zero-vector removal.

    Unit test for :func:`util.remove_zero_vectors`.
    """

    # Generate zero-vectors
    my_zero_vectors = np.zeros_like(random_vectors)

    # Combine the two sets of vectors
    my_combined_vectors = np.concatenate([random_vectors, my_zero_vectors], axis=0)

    # Remove the zero vectors
    non_zero_vectors_cleaned = vr.util.remove_zero_vectors(my_combined_vectors)

    # Check to make sure the two arrays have the same shape
    assert random_vectors.shape == non_zero_vectors_cleaned.shape

    # Check that the non-zero vectors all have a magnitude greater than zero
    non_zero_norms = np.linalg.norm(non_zero_vectors_cleaned, axis=-1)
    assert np.all(non_zero_norms > 0)

    # Check that the non-zero vectors are all equal to the originals
    assert np.all(random_vectors == non_zero_vectors_cleaned)


def test_remove_zero_vectors_location(random_vectors_with_locations):
    """Test the zero-vector removal with locations.

    Unit test for :func:`util.remove_zero_vectors`.
    """

    # Generate the spatial locations
    random_locations = random_vectors_with_locations[:, :3]

    # Generate zero-vectors
    my_zero_vectors = np.zeros((10000, 3))
    my_zero_vectors = np.concatenate([random_locations, my_zero_vectors], axis=-1)

    # Combine the two sets of vectors
    my_combined_vectors = np.concatenate(
        [random_vectors_with_locations, my_zero_vectors], axis=0
    )

    # Remove the zero vectors
    non_zero_vectors_cleaned = vr.util.remove_zero_vectors(my_combined_vectors)

    # Check to make sure the two arrays have the same shape
    assert random_vectors_with_locations.shape == non_zero_vectors_cleaned.shape

    # Check that the non-zero vectors all have a magnitude greater than zero
    non_zero_norms = np.linalg.norm(non_zero_vectors_cleaned, axis=-1)
    assert np.all(non_zero_norms > 0)

    # Check that the non-zero vectors are all equal to the originals
    assert np.all(random_vectors_with_locations == non_zero_vectors_cleaned)


def test_normalise_array_flat():
    """Test the array normalisation.

    Unit test for :func:`util.normalise_array` for a flat input.
    """

    # Generate a random array.
    my_random_array = np.random.default_rng(RANDOM_SEED).random(size=1000)

    # Normalise the array
    normalised_array = vr.util.normalise_array(my_random_array)

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
    normalised_array = vr.util.normalise_array(my_random_array, axis=-1)

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
    normalised_vectors, magnitudes = vr.util.normalise_vectors(my_random_vectors)

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

    my_total_vectors = np.concatenate([my_random_vectors, my_zero_vectors], axis=0)

    # Normalise the vectors
    normalised_vectors, magnitudes = vr.util.normalise_vectors(my_total_vectors)

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


def test_normalise_vectors_no_zero_with_location():
    """Test the vector normalisation.

    Unit test for :func:`util.normalise_vectors`.

    Warnings
    --------
    This test only verifies the normalisation. As the magnitude is computed
    using :func:`numpy.linalg.norm`, we trust that the NumPy team has
    tested that behaviour.
    """

    # Generate random vectors
    my_random_vectors = (
        np.random.default_rng(RANDOM_SEED)
        .integers(low=2, high=100, size=(1000, 3))
        .astype(float)
    )

    l = w = h = 10
    indices = np.indices((l, w, h))
    indices = np.moveaxis(indices, 0, -1).reshape(-1, 3)

    my_random_vectors = np.concatenate([indices, my_random_vectors], axis=-1)

    # Normalise the vectors
    normalised_vectors, magnitudes = vr.util.normalise_vectors(my_random_vectors)

    # Check the magnitudes of the normalised vectors
    normalised_vector_norms = np.linalg.norm(normalised_vectors[:, -3:], axis=-1)
    assert np.all(np.isclose(normalised_vector_norms, 1))


def test_normalise_one_vector_no_location():
    """Test the vector normalisation for a single vector.

    Unit test for :func:`util.normalise_vectors`.

    Warnings
    --------
    This test only verifies the normalisation. As the magnitude is computed
    using :func:`numpy.linalg.norm`, we trust that the NumPy team has
    tested that behaviour.
    """

    # Generate random vectors
    my_vector = np.array([1, 2, 2])
    expected_result = np.array([1 / 3, 2 / 3, 2 / 3])

    # Normalise the vectors
    normalised_vector, magnitude = vr.util.normalise_vectors(my_vector)

    # Check the magnitudes of the normalised vectors
    normalised_vector_norm = np.linalg.norm(normalised_vector, axis=-1)
    assert np.all(np.isclose(normalised_vector_norm, 1))

    # Check that the vector is correct
    assert np.all(np.isclose(normalised_vector, expected_result))


def test_normalise_one_vector_with_location():
    """Test the vector normalisation for a single vector.

    Unit test for :func:`util.normalise_vectors`.

    Warnings
    --------
    This test only verifies the normalisation. As the magnitude is computed
    using :func:`numpy.linalg.norm`, we trust that the NumPy team has
    tested that behaviour.
    """

    # Generate random vectors
    my_vector = np.array([1, 2, 3, 1, 2, 2]).astype(float)
    expected_result = np.array([1, 2, 3, 1 / 3, 2 / 3, 2 / 3])

    # Normalise the vectors
    normalised_vector, magnitude = vr.util.normalise_vectors(my_vector)

    # Check the magnitudes of the normalised vectors
    normalised_vector_norm = np.linalg.norm(normalised_vector[-3:], axis=-1)
    assert np.all(np.isclose(normalised_vector_norm, 1))

    # Check that the vector is correct
    assert np.all(np.isclose(normalised_vector, expected_result))


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
            np.arange(1000),
            size=300,
            replace=False,
        )

        my_random_vectors[tuple(indices_to_flip)] *= -1

    # Convert the vectors to axes
    my_axes = vr.util.convert_vectors_to_axes(my_random_vectors)

    # Check to make sure the shape is the same
    assert my_axes.shape == my_random_vectors.shape

    # Check to make sure the norms are the same
    assert np.all(
        np.linalg.norm(my_axes, axis=-1) == np.linalg.norm(my_random_vectors, axis=-1)
    )

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
    my_vectors = vr.util.create_symmetric_vectors_from_axes(my_random_axes)

    # Check the shape of the output
    n_axes, d_axes = my_random_axes.shape
    n_vectors, d_vectors = my_vectors.shape

    assert n_vectors == 2 * n_axes and d_axes == d_vectors

    # Check to see that the order is preserved
    assert np.all(my_vectors[:n_axes] == my_random_axes)
    assert np.all(my_vectors[n_axes:] == -my_random_axes)


def test_convert_spherical_to_cartesian_coordinates():
    """Test the conversion from spherical to Cartesian coordinates.

    Unit test for :func:`util.convert_spherical_to_cartesian_coordinates`.
    """

    # Create some simple angles
    input_angles = np.radians(
        [[0, 0], [180, 0], [90, 0], [90, 180], [90, 90], [90, 270]]
    )

    expected_output = np.array(
        [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
    )

    # Convert the angles to Cartesian coordinates
    my_cartesian_coordinates = vr.util.convert_spherical_to_cartesian_coordinates(
        input_angles, radius=1
    )

    # Compare to expected output
    assert np.all(np.isclose(my_cartesian_coordinates, expected_output))


def test_compute_vector_orientation_angles():
    """Test spherical coordinate calculation.

    Unit test for :func:`util.compute_vector_orientation_angles`.
    """

    # Create the test input vectors and output angles
    input_vectors = np.array(
        [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
    )

    expected_output = np.radians(
        [[0, 0], [180, 0], [90, 0], [90, 180], [90, 90], [90, 270]]
    )

    # Run the conversion
    computed_spherical_coords = vr.util.compute_vector_orientation_angles(input_vectors)

    # Check the output
    assert np.all(computed_spherical_coords == expected_output)


def test_compute_vector_orientation_angles_one_vector():
    """Test spherical coordinate calculation.

    Unit test for :func:`util.compute_vector_orientation_angles`.
    """

    # Create the test input vectors and output angles
    input_vectors = np.array([0, 0, 1])

    expected_output = np.radians([0, 0])

    # Run the conversion
    computed_spherical_coords = vr.util.compute_vector_orientation_angles(input_vectors)

    # Check the output
    assert np.all(computed_spherical_coords == expected_output)


def test_compute_spherical_coordinates():
    """Test spherical coordinate calculation.

    Unit test for :func:`util.compute_spherical_coordinates`.
    """

    # Create the test input vectors and output angles
    input_vectors = np.array(
        [[0, 0, 2], [0, 0, -4], [0, 0.5, 0], [0, -1, 0], [0.3, 0, 0], [-1, 0, 0]]
    )

    expected_output = np.array(
        [
            [0, 0, 2],
            [180, 0, 4],
            [90, 0, 0.5],
            [90, 180, 1],
            [90, 90, 0.3],
            [90, 270, 1],
        ]
    )

    # Run the conversion
    computed_spherical_coords = vr.util.compute_spherical_coordinates(
        input_vectors, use_degrees=True
    )

    # Check the output
    assert np.all(np.isclose(computed_spherical_coords, expected_output))


def test_compute_spherical_coordinates_one_vector():
    """Test spherical coordinate calculation.

    Unit test for :func:`util.compute_spherical_coordinates`.
    """

    # Create the test input vectors and output angles
    input_vectors = np.array([0, 0, 5])

    expected_output = np.array([0, 0, 5])

    # Run the conversion
    computed_spherical_coords = vr.util.compute_spherical_coordinates(
        input_vectors, use_degrees=True
    )

    # Check the output
    assert np.all(np.isclose(computed_spherical_coords, expected_output))


def test_convert_to_math_spherical_coordinates():
    """Test conversion from custom spherical coordinates to FLE system.

    Unit test for :func:`util.convert_to_math_spherical_coordinates`.
    """

    # Define our input angles
    input_angles = np.radians(
        [[0, 0], [180, 0], [90, 0], [45, 45], [90, 180], [90, 90], [90, 270]]
    )

    # Define the expected outputs
    expected_outputs = np.radians(
        [[90, 0], [90, 180], [90, 90], [45, 45], [270, 90], [0, 90], [180, 90]]
    )

    # And now perform the conversion
    converted_angles = vr.util.convert_to_math_spherical_coordinates(
        input_angles, use_degrees=False
    )

    # And check the equality
    assert np.all(converted_angles == expected_outputs)


def test_convert_to_math_spherical_coordinates_degrees():
    """Test conversion from custom spherical coordinates to FLE system.

    Unit test for :func:`util.convert_to_math_spherical_coordinates`.
    """

    # Define our input angles
    input_angles = np.array(
        [[0, 0], [180, 0], [90, 0], [45, 45], [90, 180], [90, 90], [90, 270]]
    )

    # Define the expected outputs
    expected_outputs = np.array(
        [[90, 0], [90, 180], [90, 90], [45, 45], [270, 90], [0, 90], [180, 90]]
    )

    # And now perform the conversion
    converted_angles = vr.util.convert_to_math_spherical_coordinates(
        input_angles, use_degrees=True
    )

    # And check the equality
    assert np.all(converted_angles == expected_outputs)


def test_convert_to_vectorose_spherical_coordinates():
    """Test conversion from FLE system to the custom spherical coordinates.

    Unit test for
    :func:`util.convert_math_spherical_coordinates_to_vr_coordinates`.
    """

    # Define our input angles
    input_angles = np.radians(
        [[90, 0], [90, 180], [90, 90], [45, 45], [270, 90], [0, 90], [180, 90]]
    )

    # Define the expected outputs
    expected_outputs = np.radians(
        [[0, 0], [180, 0], [90, 0], [45, 45], [90, 180], [90, 90], [90, 270]]
    )

    # And now perform the conversion
    converted_angles = vr.util.convert_math_spherical_coordinates_to_vr_coordinates(
        input_angles, use_degrees=False
    )

    # And check the equality
    assert np.all(converted_angles == expected_outputs)


def test_convert_to_vectorose_spherical_coordinates_degrees():
    """Test conversion from FLE system to the custom spherical coordinates.

    Unit test for
    :func:`util.convert_math_spherical_coordinates_to_vr_coordinates`.
    """

    # Define our input angles
    input_angles = np.array(
        [[90, 0], [90, 180], [90, 90], [45, 45], [270, 90], [0, 90], [180, 90]]
    )

    # Define the expected outputs
    expected_outputs = np.array(
        [[0, 0], [180, 0], [90, 0], [45, 45], [90, 180], [90, 90], [90, 270]]
    )

    # And now perform the conversion
    converted_angles = vr.util.convert_math_spherical_coordinates_to_vr_coordinates(
        input_angles, use_degrees=True
    )

    # And check the equality
    assert np.all(converted_angles == expected_outputs)


def test_rotate_vectors_no_change():
    """Test vector rotation for no change.

    Unit test for :func:`util.rotate_vectors`.
    """

    # Create a set of vectors
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Rotate them so that the pole is now on the bottom.
    new_pole = np.array([0, 0, 1])

    rotated_vectors = vr.util.rotate_vectors(vectors, new_pole)

    # Now, we would expect the following result.
    expected_result = vectors

    # And now check the equality
    assert np.all(np.isclose(rotated_vectors, expected_result))


def test_rotate_vectors_flip():
    """Test vector rotation for complete flip.

    Unit test for :func:`util.rotate_vectors`.
    """

    # Create a set of vectors
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Rotate them so that the pole is now on the bottom.
    new_pole = np.array([0, 0, -1])

    rotated_vectors = vr.util.rotate_vectors(vectors, new_pole)

    # Now, we would expect the following result.
    expected_result = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # And now check the equality
    assert np.all(np.isclose(rotated_vectors, expected_result))


def test_rotate_vectors_partial_rotation():
    """Test vector rotation for partial rotation.

    Unit test for :func:`util.rotate_vectors`.
    """

    # Create a set of vectors
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Rotate them so that the pole is now on the bottom.
    new_pole = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0])

    rotated_vectors = vr.util.rotate_vectors(vectors, new_pole)

    # Now, we would expect the following result.
    expected_result = np.array(
        [
            [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [0, 0, -1],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
        ]
    )

    # And now check the equality
    assert np.all(np.isclose(rotated_vectors, expected_result))


def test_compute_arc_lengths():
    """Test the arc length calculation.

    Unit test for :func:`util.compute_arc_lengths`.
    """

    my_reference_vector = np.array([1, 0, 0])

    my_collection_of_vectors = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [np.sqrt(3) / 2, 1 / 2, 0],
        ]
    )

    expected_arc_lengths = np.array([0, np.pi / 2, np.pi / 2, np.pi / 4, np.pi / 6])

    calculated_arc_lengths = vr.util.compute_arc_lengths(
        my_reference_vector, my_collection_of_vectors
    )

    # Check the length
    assert len(calculated_arc_lengths) == len(expected_arc_lengths)

    # And now check the values
    assert np.all(np.isclose(calculated_arc_lengths, expected_arc_lengths))


# TODO: Add tests that send in only one vector to the functions that squeeze
